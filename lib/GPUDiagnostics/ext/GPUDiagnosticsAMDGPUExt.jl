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

# Telemetry child: HIP is touched only HERE to resolve each device's sysfs paths once; the
# spawned bin/gputrace.sh then reads the amdgpu driver's counters (VRAM included, via
# `mem_info_vram_used` — no hipMemGetInfo) from its own process, immune to this process's HIP
# locks and Julia's GC/timer coupling. Sysfs paths contain no ':' so the devspec join is safe.
function GD.gpu_telemetry_child_cmd(::ROCBackend, device_ids::AbstractVector{<:Integer},
        dt::Real, stopfile::AbstractString)
    script = joinpath(pkgdir(GD), "bin", "gputrace.sh")
    specs = map(device_ids) do i
        card = _amd_device_sysfs(AMDGPU.devices()[i])
        mb = joinpath(card, "mem_busy_percent")   # absent on some devices (e.g. iGPUs) → nan
        join([string(i), _amd_power_file(card), joinpath(card, "gpu_busy_percent"),
            isfile(mb) ? mb : "-", joinpath(card, "mem_info_vram_used")], ":")
    end
    return `sh $script $dt $(getpid()) $stopfile $specs`
end

end
