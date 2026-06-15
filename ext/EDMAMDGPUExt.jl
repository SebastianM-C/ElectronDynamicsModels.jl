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

function EDM.gpu_power(::ROCBackend)
    dev = _amd_device_sysfs()
    hw = first(filter(d -> startswith(basename(d), "hwmon"),
        readdir(joinpath(dev, "hwmon"); join = true)))
    f = isfile(joinpath(hw, "power1_average")) ? "power1_average" : "power1_input"
    return parse(Int, strip(read(joinpath(hw, f), String))) / 1.0e6   # µW → W
end

function EDM.gpu_utilization(::ROCBackend)
    dev = _amd_device_sysfs()
    rd(f) = isfile(joinpath(dev, f)) ? parse(Int, strip(read(joinpath(dev, f), String))) / 100 : NaN
    return (compute = rd("gpu_busy_percent"), memory = rd("mem_busy_percent"))
end

end
