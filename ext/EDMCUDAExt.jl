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

# FP64 FMA lanes per SM by compute capability (vector units, not tensor cores). Datacenter
# parts (GV100, GA100, GH100, GB100) run FP64 at 1/2 the FP32 rate; consumer parts at 1/32
# (Turing, Ampere GA10x, Ada, Blackwell GB20x) — 2 lanes of 64 FP32 lanes. Anchors: H100 SXM
# 132 × 1.98 GHz × 64 × 2 = 33.5 TFLOP/s; RTX 5090 170 × 2.41 GHz × 2 × 2 = 1.64 TFLOP/s.
const FP64_LANES_PER_SM = Dict{Tuple{Int, Int}, Int}(
    (7, 0) => 32, (7, 5) => 2,
    (8, 0) => 32, (8, 6) => 2, (8, 7) => 2, (8, 9) => 2,
    (9, 0) => 64,
    (10, 0) => 64, (10, 3) => 64,
    (12, 0) => 2,
)
_fp64_lanes(cc::VersionNumber) = get(FP64_LANES_PER_SM, (Int(cc.major), Int(cc.minor)), NaN)

EDM.gpu_arch(::CUDABackend) = (cc = CUDA.capability(CUDA.device()); "$(cc.major).$(cc.minor)")
function EDM.gpu_peak_fp64_flops(::CUDABackend)
    dev = CUDA.device()
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    clock_hz = 1.0e3 * CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE)   # kHz → Hz
    return sms * clock_hz * _fp64_lanes(CUDA.capability(dev)) * 2
end

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
