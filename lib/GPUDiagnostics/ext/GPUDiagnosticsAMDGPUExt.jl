module GPUDiagnosticsAMDGPUExt

# AMDGPU.jl implementations of the vendor-GPU API declared in src/device_api.jl. Loaded
# automatically when both GPUDiagnostics and AMDGPU are in the session. Device props
# come from HIP; free/total VRAM from hipMemGetInfo. AMDGPU.jl (2.5) ships no SMI module, so
# power/utilization are read from the amdgpu driver's sysfs (the same source nvtop uses) —
# no rocm-smi/amd-smi needed.

using GPUDiagnostics
using AMDGPU

const GD = GPUDiagnostics

# AMDGPU device ids are already 1-based (HIPDevice(id=1, …)), matching the common API — no offset.
GD.gpu_device_count(::ROCBackend) = length(AMDGPU.devices())
GD.gpu_device(::ROCBackend) = AMDGPU.device_id(AMDGPU.device())
function GD.gpu_device!(::ROCBackend, i::Integer)
    prev = AMDGPU.device_id(AMDGPU.device())
    AMDGPU.device!(AMDGPU.devices()[i])
    return prev
end
GD.gpu_name(::ROCBackend) = AMDGPU.HIP.name(AMDGPU.device())
GD.gpu_sm_count(::ROCBackend) =
    Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)
GD.gpu_max_threads_per_sm(::ROCBackend) =
    Int(AMDGPU.HIP.properties(AMDGPU.device()).maxThreadsPerMultiProcessor)

# Device-event timing on the task-local stream (the one KernelAbstractions launches on).
# HIPEvent disables timing by default (hipEventDisableTiming) — `timing = true` is required.
GD.gpu_event(::ROCBackend) = AMDGPU.HIP.HIPEvent(AMDGPU.stream(); do_record = true, timing = true)
function GD.gpu_elapsed(start::AMDGPU.HIP.HIPEvent, stop::AMDGPU.HIP.HIPEvent)
    AMDGPU.HIP.synchronize(stop)
    return Float64(AMDGPU.HIP.elapsed(start, stop))   # seconds
end

# `gcn_arch` may carry feature suffixes ("gfx942:sramecc+:xnack-"); report the bare name.
_gfx_name(dev = AMDGPU.device()) = String(first(split(AMDGPU.HIP.gcn_arch(dev), ':')))

GD.gpu_arch(::ROCBackend) = _gfx_name()

function GD.gpu_memory_info(::ROCBackend)
    free = Ref{Csize_t}(0)
    total = Ref{Csize_t}(0)
    AMDGPU.HIP.hipMemGetInfo(free, total)
    return (total = Int(total[]), free = Int(free[]), used = Int(total[] - free[]))
end

# Map the current HIP device to its DRM card via PCI address, then read driver sysfs. AMDGPU.jl
# has no SMI, but the amdgpu kernel driver exposes power (hwmon `power1_average`, µW) and the
# engine-busy counter (`gpu_busy_percent`) under /sys/class/drm/cardN/device — what nvtop reads.
function _amd_device_sysfs(dev = AMDGPU.device())
    p = AMDGPU.HIP.properties(dev)
    # DRM card dir is named by PCI address, e.g. "0000:03:00.0" (domain:bus:device.function).
    pci = string(p.pciDomainID; base = 16, pad = 4) * ":" *
        string(p.pciBusID; base = 16, pad = 2) * ":" *
        string(p.pciDeviceID; base = 16, pad = 2) * ".0"
    for c in readdir("/sys/class/drm"; join = true)
        occursin(r"^card\d+$", basename(c)) || continue
        link = joinpath(c, "device")
        islink(link) && basename(realpath(link)) == pci && return link
    end
    error("gpu telemetry: no /sys/class/drm card matches PCI $pci")
end

# The hwmon power file under a card's sysfs dir: `power1_average` where the driver provides
# it (most dGPUs), else the instantaneous `power1_input`. Reports µW.
function _amd_power_file(card::AbstractString)
    hw = first(filter(d -> startswith(basename(d), "hwmon"),
        readdir(joinpath(card, "hwmon"); join = true)))
    f = isfile(joinpath(hw, "power1_average")) ? "power1_average" : "power1_input"
    return joinpath(hw, f)
end

GD.gpu_power(::ROCBackend) =
    parse(Int, strip(read(_amd_power_file(_amd_device_sysfs()), String))) / 1.0e6   # µW → W

function GD.gpu_utilization(::ROCBackend)
    dev = _amd_device_sysfs()
    rd(f) = isfile(joinpath(dev, f)) ? parse(Int, strip(read(joinpath(dev, f), String))) / 100 : NaN
    return (compute = rd("gpu_busy_percent"), memory = rd("mem_busy_percent"))
end

# Telemetry sources: HIP is touched only HERE, in the parent, to resolve each device's sysfs
# paths once; the sampler child (a Julia process that never loads AMDGPU.jl) then reads the amdgpu
# driver's counters (VRAM included, via `mem_info_vram_used` — no hipMemGetInfo) directly, immune
# to this process's HIP locks and Julia's GC/timer coupling. Sysfs paths contain no ':' so the
# spec join is safe. No hardware counters here (`counters` is ignored).
function GD.gpu_sampler_sources(::ROCBackend, device_ids::AbstractVector{<:Integer}, ::Symbol)
    specs = map(device_ids) do i
        card = _amd_device_sysfs(AMDGPU.devices()[i])
        mb = joinpath(card, "mem_busy_percent")   # absent on some devices (e.g. iGPUs) → nan
        "sysfs:" * join([string(i), _amd_power_file(card), joinpath(card, "gpu_busy_percent"),
            isfile(mb) ? mb : "-", joinpath(card, "mem_info_vram_used")], ":")
    end
    return (specs = specs, packages = Base.PkgId[])
end

# ── Compile-time resource report (src/resources.jl hooks) ───────────────────────────────────
# Inventory = AMDGPU.jl's compiled-kernel cache (one HIPKernel per compiled function × argument
# types × device). Attributes come from the driver-API `hipFuncGetAttribute` on the module
# function (the runtime-API `hipFuncGetAttributes` only knows host stubs); NUM_REGS is the VGPR
# count. Occupancy from `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` (a "multiprocessor"
# is what HIP calls one — a WGP on RDNA3, a CU on CDNA), capacities from the CURRENT device's
# properties. The ISA dump (`code_native`, ~0.5 s, no launch) adds the SGPR/VGPR/spill/scratch
# figures and the compiler's own occupancy estimate, which the HIP attributes do not expose.
GD._compiled_kernels(::ROCBackend) = Base.@lock AMDGPU.Compiler.hipfunction_lock begin
    [GD._compiled_kernel(k) for k in values(AMDGPU.Compiler._kernel_instances) if k isa AMDGPU.Runtime.HIPKernel]
end

function _hip_func_attr(fun::AMDGPU.HIP.HIPFunction, attr)
    v = Ref{Cint}(0)
    AMDGPU.HIP.hipFuncGetAttribute(v, attr, fun)
    return Int(v[])
end

function GD._kernel_attributes(::ROCBackend, k::AMDGPU.Runtime.HIPKernel)
    HIP = AMDGPU.HIP
    a(attr) = _hip_func_attr(k.fun, attr)
    return (;
        registers = a(HIP.HIP_FUNC_ATTRIBUTE_NUM_REGS),
        local_mem_bytes = a(HIP.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        shared_mem_bytes = a(HIP.HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
        const_mem_bytes = a(HIP.HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES),
        max_threads_per_block = a(HIP.HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
    )
end

function GD._kernel_occupancy(::ROCBackend, k::AMDGPU.Runtime.HIPKernel, block_size::Int)
    nb = Ref{Cint}(0)
    AMDGPU.HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(nb, k.fun, block_size, 0)
    dev = AMDGPU.device()
    p = AMDGPU.HIP.properties(dev)
    return (;
        active_blocks_per_sm = Int(nb[]),
        warp_size = Int(AMDGPU.HIP.wavefrontsize(dev)),
        max_threads_per_sm = Int(p.maxThreadsPerMultiProcessor),
        shared_mem_per_sm = Int(p.maxSharedMemoryPerMultiProcessor),
    )
end

function GD._kernel_isa_info(::ROCBackend, ck::GD.CompiledKernel{<:AMDGPU.Runtime.HIPKernel})
    k = ck.kernel
    TT = typeof(k).parameters[2]
    try
        io = IOBuffer()
        AMDGPU.code_native(io, k.f, TT; kernel = true, raw = true)
        return GD._parse_amdgpu_kernel_info(String(take!(io)))
    catch err
        @warn "kernel_resources: AMD ISA dump failed — reporting HIP attributes only" exception = err
        return Dict{String, Any}()
    end
end

end
