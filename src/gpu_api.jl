# Vendor-specific GPU operations not covered by KernelAbstractions: device enumeration +
# selection (the basis for multi-device electron sharding) and telemetry (power / utilization /
# memory + occupancy props, for bottleneck diagnosis and the occupancy bench). Each generic
# below dispatches on the KA `Backend`; the CUDA/AMDGPU package extensions
# (ext/EDM{CUDA,AMDGPU}Ext.jl) supply the methods, loaded on demand when the vendor package is
# in the session. The `::Backend` fallback errors helpfully when neither is loaded.

import KernelAbstractions as KA

"""    gpu_device_count(backend) -> Int

Number of GPUs the vendor runtime exposes for `backend`."""
function gpu_device_count end

"""    gpu_device(backend) -> Int

1-based index of the current device (this common API is 1-based across vendors)."""
function gpu_device end

"""    gpu_device!(backend, i) -> Int

Make GPU `i` (1-based) current; returns the previously-current index. One Julia task per device
+ `gpu_device!` is how multi-device electron sharding pins each shard to a GPU."""
function gpu_device! end

"""    gpu_name(backend) -> String

Marketing name of the current device."""
function gpu_name end

"""    gpu_power(backend) -> Float64

Instantaneous board power draw of the current device, in Watts. A cheap live proxy for compute
*saturation* — NOT occupancy: low power + high SM-utilization ⇒ latency-bound."""
function gpu_power end

"""    gpu_utilization(backend) -> @NamedTuple{compute, memory}

Current-device utilization, each a 0–1 fraction (fraction of recent time the engine was busy)."""
function gpu_utilization end

"""    gpu_memory_info(backend) -> @NamedTuple{total, free, used}

Current-device global memory, in bytes."""
function gpu_memory_info end

"""    gpu_sm_count(backend) -> Int

Streaming-multiprocessor (CU on AMD) count of the current device."""
function gpu_sm_count end

"""    gpu_max_threads_per_sm(backend) -> Int

Max resident threads per SM/CU — with `gpu_sm_count`, the device's total resident-thread
capacity (the denominator of thread-fill occupancy)."""
function gpu_max_threads_per_sm end

# Fallbacks: a KA Backend with no vendor extension loaded → a clear "load the package" error.
# The extensions add more-specific methods (e.g. ::CUDABackend) that win over these.
for f in (
        :gpu_device_count, :gpu_device, :gpu_name, :gpu_power, :gpu_utilization,
        :gpu_memory_info, :gpu_sm_count, :gpu_max_threads_per_sm,
    )
    @eval function $f(b::KA.Backend)
        error(
            $(string(f)), ": no GPU vendor extension loaded for ", typeof(b),
            " — load CUDA.jl or AMDGPU.jl"
        )
    end
end
gpu_device!(b::KA.Backend, ::Integer) = error(
    "gpu_device!: no GPU vendor extension loaded for ", typeof(b), " — load CUDA.jl or AMDGPU.jl"
)

"""
    thread_fill_occupancy(backend, n_threads) -> Float64

Thread-fill occupancy: the fraction of the current device's total resident-thread capacity
(`gpu_sm_count × gpu_max_threads_per_sm`) a launch of `n_threads` can fill. For the
pixel-parallel `accumulate_field` kernel, `n_threads = Nx·Ny` per electron (× electrons-per-
launch if batched). This is an UPPER BOUND on achieved occupancy — per-thread registers /
shared memory cap it further; measure the real number with `ncu`.
"""
function thread_fill_occupancy(backend::KA.Backend, n_threads::Integer)
    capacity = gpu_sm_count(backend) * gpu_max_threads_per_sm(backend)

    return n_threads / capacity
end
