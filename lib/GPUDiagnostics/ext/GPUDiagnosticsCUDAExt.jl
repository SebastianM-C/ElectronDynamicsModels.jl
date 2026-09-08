module GPUDiagnosticsCUDAExt

# CUDA.jl implementations of the vendor-GPU API declared in src/device_api.jl. Loaded
# automatically when both GPUDiagnostics and CUDA are in the session. Telemetry
# (power/utilization/memory) goes through NVML; device props through CUDA attributes.

using GPUDiagnostics
using CUDA
using CUDA: NVML

const GD = GPUDiagnostics

# NVML handle for the current CUDA device (NVML indexes by UUID, not the CUDA ordinal).
_nvml() = NVML.Device(CUDA.uuid(CUDA.device()))

GD.gpu_device_count(::CUDABackend) = length(CUDA.devices())
GD.gpu_device(::CUDABackend) = CUDA.deviceid(CUDA.device()) + 1          # 0-based CUDA → 1-based API
function GD.gpu_device!(::CUDABackend, i::Integer)
    prev = CUDA.deviceid(CUDA.device()) + 1
    CUDA.device!(i - 1)
    return prev
end
GD.gpu_name(::CUDABackend) = CUDA.name(CUDA.device())
GD.gpu_power(::CUDABackend) = NVML.power_usage(_nvml())                  # Watts (Float64)
GD.gpu_utilization(::CUDABackend) = NVML.utilization_rates(_nvml())     # (compute, memory) ∈ [0,1]
GD.gpu_memory_info(::CUDABackend) = NVML.memory_info(_nvml())           # (total, free, used) bytes

# Device-event timing on the task-local stream (the one KernelAbstractions launches on). CuEvent's
# default flags keep timing enabled; `elapsed` needs the stop event complete → synchronize it.
GD.gpu_event(::CUDABackend) = (e = CUDA.CuEvent(); CUDA.record(e, CUDA.stream()); e)
function GD.gpu_elapsed(start::CUDA.CuEvent, stop::CUDA.CuEvent)
    CUDA.synchronize(stop)
    return Float64(CUDA.elapsed(start, stop))   # seconds
end
GD.gpu_sm_count(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
GD.gpu_max_threads_per_sm(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

GD.gpu_arch(::CUDABackend) = (cc = CUDA.capability(CUDA.device()); "$(cc.major).$(cc.minor)")

# Telemetry child: the CUDA runtime is touched only HERE to map our ordinals to NVML uuids
# (stable under CUDA_VISIBLE_DEVICES); the spawned bin/gputrace_cuda.sh then runs one
# `nvidia-smi -lms` daemon per device from its own process, immune to this process's CUDA
# locks and Julia's GC/timer coupling.
function GD.gpu_telemetry_child_cmd(::CUDABackend, device_ids::AbstractVector{<:Integer},
        dt::Real, stopfile::AbstractString)
    script = joinpath(pkgdir(GD), "bin", "gputrace_cuda.sh")
    cudevs = collect(CUDA.devices())
    specs = ["GPU-$(CUDA.uuid(cudevs[i]))=$i" for i in device_ids]
    return `sh $script $(round(Int, 1000 * dt)) $(getpid()) $stopfile $specs`
end

end
