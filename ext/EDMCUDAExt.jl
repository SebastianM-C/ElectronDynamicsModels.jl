module EDMCUDAExt

# CUDA.jl implementations of the vendor-GPU API declared in src/gpu_api.jl. Loaded
# automatically when both ElectronDynamicsModels and CUDA are in the session. Telemetry
# (power/utilization/memory) goes through NVML; device props through CUDA attributes.

using ElectronDynamicsModels
using CUDA
using CUDA: NVML

const EDM = ElectronDynamicsModels

# NVML handle for the current CUDA device (NVML indexes by UUID, not the CUDA ordinal).
_nvml() = NVML.Device(CUDA.uuid(CUDA.device()))

EDM.gpu_device_count(::CUDABackend) = length(CUDA.devices())
EDM.gpu_device(::CUDABackend) = CUDA.deviceid(CUDA.device()) + 1          # 0-based CUDA → 1-based API
function EDM.gpu_device!(::CUDABackend, i::Integer)
    prev = CUDA.deviceid(CUDA.device()) + 1
    CUDA.device!(i - 1)
    return prev
end
EDM.gpu_name(::CUDABackend) = CUDA.name(CUDA.device())
EDM.gpu_power(::CUDABackend) = NVML.power_usage(_nvml())                  # Watts (Float64)
EDM.gpu_utilization(::CUDABackend) = NVML.utilization_rates(_nvml())     # (compute, memory) ∈ [0,1]
EDM.gpu_memory_info(::CUDABackend) = NVML.memory_info(_nvml())           # (total, free, used) bytes
EDM.gpu_sm_count(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
EDM.gpu_max_threads_per_sm(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

# Telemetry child: the CUDA runtime is touched only HERE to map our ordinals to NVML uuids
# (stable under CUDA_VISIBLE_DEVICES); the spawned scripts/gputrace_cuda.sh then runs one
# `nvidia-smi -lms` daemon per device from its own process, immune to the solver's CUDA
# locks and Julia's GC/timer coupling.
function EDM.gpu_telemetry_child_cmd(::CUDABackend, device_ids::AbstractVector{<:Integer},
        dt::Real, stopfile::AbstractString)
    script = joinpath(pkgdir(EDM), "scripts", "gputrace_cuda.sh")
    cudevs = collect(CUDA.devices())
    specs = ["GPU-$(CUDA.uuid(cudevs[i]))=$i" for i in device_ids]
    return `sh $script $(round(Int, 1000 * dt)) $(getpid()) $stopfile $specs`
end

end
