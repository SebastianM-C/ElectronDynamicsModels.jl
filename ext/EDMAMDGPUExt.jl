module EDMAMDGPUExt

# AMDGPU.jl implementations of the vendor-GPU API declared in src/gpu_api.jl. Loaded
# automatically when both ElectronDynamicsModels and AMDGPU are in the session. Device props
# come from HIP; free/total VRAM from hipMemGetInfo. AMDGPU.jl (2.5) ships no SMI module, so
# power/utilization are read from the amdgpu driver's sysfs (the same source nvtop uses) —
# no rocm-smi/amd-smi needed.

using ElectronDynamicsModels
using AMDGPU

const EDM = ElectronDynamicsModels

# AMDGPU device ids are already 1-based (HIPDevice(id=1, …)), matching the common API — no offset.
EDM.gpu_device_count(::ROCBackend) = length(AMDGPU.devices())
EDM.gpu_device(::ROCBackend) = AMDGPU.device_id(AMDGPU.device())
function EDM.gpu_device!(::ROCBackend, i::Integer)
    prev = AMDGPU.device_id(AMDGPU.device())
    AMDGPU.device!(AMDGPU.devices()[i])
    return prev
end
EDM.gpu_name(::ROCBackend) = AMDGPU.HIP.name(AMDGPU.device())
EDM.gpu_sm_count(::ROCBackend) =
    Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)
EDM.gpu_max_threads_per_sm(::ROCBackend) =
    Int(AMDGPU.HIP.properties(AMDGPU.device()).maxThreadsPerMultiProcessor)

# FP64 vector FLOP per clock per HIP "multiprocessor" by gfx architecture (FMA = 2 FLOP).
# HIP's multiProcessorCount is the CU count on CDNA (MI100: 64 lanes at half rate → 64;
# MI200/MI300: full-rate FP64 → 128) but the WORK-GROUP-PROCESSOR count on RDNA (one WGP = two
# CUs: the W7900 reports 48 for its 96 CUs), where FP64 runs at 1/32 of the dual-issue FP32 rate
# — 8 per CU, 16 per WGP. Clock = the device's reported max engine clock (rocminfo "Max Clock"),
# not the marketing boost. Anchors: MI300X 304 CU × 2.1 GHz × 128 = 81.7 TFLOP/s; W7900 48 WGP ×
# 1.76 GHz × 16 = 1.35 TFLOP/s.
const FP64_FLOP_PER_CLK_PER_MP = Dict{String, Int}(
    "gfx908" => 64, "gfx90a" => 128, "gfx940" => 128, "gfx941" => 128, "gfx942" => 128,
    "gfx950" => 128,
    "gfx1100" => 16, "gfx1101" => 16, "gfx1102" => 16,
)
# `gcn_arch` may carry feature suffixes ("gfx942:sramecc+:xnack-"); the table is keyed on the bare name.
_gfx_name(dev = AMDGPU.device()) = String(first(split(AMDGPU.HIP.gcn_arch(dev), ':')))
_fp64_flop_per_clk(arch::AbstractString) = get(FP64_FLOP_PER_CLK_PER_MP, arch, NaN)

EDM.gpu_arch(::ROCBackend) = _gfx_name()
function EDM.gpu_peak_fp64_flops(::ROCBackend)
    dev = AMDGPU.device()
    p = AMDGPU.HIP.properties(dev)
    clock_hz = 1.0e3 * Int(p.clockRate)   # kHz → Hz
    return Int(p.multiProcessorCount) * clock_hz * _fp64_flop_per_clk(_gfx_name(dev))
end

function EDM.gpu_memory_info(::ROCBackend)
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

EDM.gpu_power(::ROCBackend) =
    parse(Int, strip(read(_amd_power_file(_amd_device_sysfs()), String))) / 1.0e6   # µW → W

function EDM.gpu_utilization(::ROCBackend)
    dev = _amd_device_sysfs()
    rd(f) = isfile(joinpath(dev, f)) ? parse(Int, strip(read(joinpath(dev, f), String))) / 100 : NaN
    return (compute = rd("gpu_busy_percent"), memory = rd("mem_busy_percent"))
end

# Telemetry child: HIP is touched only HERE to resolve each device's sysfs paths once; the
# spawned scripts/gputrace.sh then reads the amdgpu driver's counters (VRAM included, via
# `mem_info_vram_used` — no hipMemGetInfo) from its own process, immune to the solver's HIP
# locks and Julia's GC/timer coupling. Sysfs paths contain no ':' so the devspec join is safe.
function EDM.gpu_telemetry_child_cmd(::ROCBackend, device_ids::AbstractVector{<:Integer},
        dt::Real, stopfile::AbstractString)
    script = joinpath(pkgdir(EDM), "scripts", "gputrace.sh")
    specs = map(device_ids) do i
        card = _amd_device_sysfs(AMDGPU.devices()[i])
        mb = joinpath(card, "mem_busy_percent")   # absent on some devices (e.g. iGPUs) → nan
        join([string(i), _amd_power_file(card), joinpath(card, "gpu_busy_percent"),
            isfile(mb) ? mb : "-", joinpath(card, "mem_info_vram_used")], ":")
    end
    return `sh $script $dt $(getpid()) $stopfile $specs`
end

end
