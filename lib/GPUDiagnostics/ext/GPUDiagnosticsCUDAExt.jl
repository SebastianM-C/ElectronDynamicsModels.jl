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

# ── Compile-time resource report (src/resources.jl hooks) ───────────────────────────────────
# Inventory = the compiled-kernel cache of CUDA.jl's compiler (CUDACore from CUDA 6.3; the
# same names live in CUDA itself before the split). Attributes via the public `registers` /
# `memory` / `maxthreads` accessors (cuFuncGetAttribute underneath), occupancy via the driver's
# `cuOccupancyMaxActiveBlocksPerMultiprocessor`, capacities from the CURRENT device.
const _CC = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA

GD._compiled_kernels(::CUDABackend) = Base.@lock _CC.cufunction_lock begin
    [GD._compiled_kernel(k) for k in values(_CC._kernel_instances) if k isa CUDA.HostKernel]
end

function GD._kernel_attributes(::CUDABackend, k::CUDA.HostKernel)
    mem = CUDA.memory(k)   # (local, shared, constant) bytes; `local` is a keyword → positional
    return (;
        registers = Int(CUDA.registers(k)),
        local_mem_bytes = Int(mem[1]),
        shared_mem_bytes = Int(mem.shared),
        const_mem_bytes = Int(mem.constant),
        max_threads_per_block = Int(CUDA.maxthreads(k)),
    )
end

function GD._kernel_occupancy(::CUDABackend, k::CUDA.HostKernel, block_size::Int)
    dev = CUDA.device()
    return (;
        active_blocks_per_sm = Int(CUDA.active_blocks(k.fun, block_size)),
        warp_size = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)),
        max_threads_per_sm = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)),
        shared_mem_per_sm = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)),
    )
end

# PTX / SASS binary versions the kernel was built for (CUDA.jl ships no SASS resource-usage
# parser; the driver attributes above already carry registers and spill (local) bytes).
function GD._kernel_isa_info(::CUDABackend, k::CUDA.HostKernel)
    try
        v = _CC.version(k)
        return Dict{String, Any}("ptx_version" => string(v.ptx), "binary_version" => string(v.binary))
    catch err
        @warn "kernel_resources: PTX/binary version query failed" exception = err
        return Dict{String, Any}()
    end
end

end
