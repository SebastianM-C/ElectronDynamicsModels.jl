# Vendor-specific GPU operations not covered by KernelAbstractions: device enumeration +
# selection (the basis for multi-device work sharding) and telemetry (power / utilization /
# memory + occupancy props, for bottleneck diagnosis). Each generic below dispatches on the KA
# `Backend`; the CUDA/AMDGPU package extensions (ext/GPUDiagnostics{CUDA,AMDGPU}Ext.jl) supply
# the methods, loaded on demand when the vendor package is in the session. The `::Backend`
# fallback errors helpfully when neither is loaded; the KA `CPU` backend gets host fallbacks.

"""    gpu_device_count(backend) -> Int

Number of GPUs the vendor runtime exposes for `backend`."""
function gpu_device_count end

"""    gpu_device(backend) -> Int

1-based index of the current device (this common API is 1-based across vendors)."""
function gpu_device end

"""    gpu_device!(backend, i) -> Int

Make GPU `i` (1-based) current; returns the previously-current index. One Julia task per device
+ `gpu_device!` is how multi-device work sharding pins each shard to a GPU."""
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

"""    gpu_telemetry_child_cmd(backend, device_ids, dt, stopfile) -> Cmd

Build the OUT-OF-PROCESS telemetry sampler command for the (1-based) `device_ids`: a child
that emits one canonical TSV row per device every `dt` seconds on stdout
(`epoch_s  device  power_W  compute_util  mem_util  vram_used_B`; `nan` for counters a
device doesn't expose) and exits when `stopfile` appears or the parent dies. The vendor
runtime is touched only while BUILDING the command (resolving sysfs paths / NVML uuids);
the child itself reads driver sysfs (AMD, bin/gputrace.sh) or runs `nvidia-smi -lms`
(NVIDIA, bin/gputrace_cuda.sh) and shares nothing with this process.

Sampling must live out of process: an in-process tick either wedges on the vendor runtime
behind a backed-up kernel stream, or — even with a runtime-free tick — is suspended wholesale
with the sleeping task by Julia's GC/libuv-timer coupling while the solver's host thread
allocates (measured on a W7900 host: 0 ticks/15 s under pure-CPU alloc churn, 1 tick/98.5 s
over a real hour-scale kernel loop)."""
function gpu_telemetry_child_cmd end

"""    gpu_sm_count(backend) -> Int

Streaming-multiprocessor (CU on AMD) count of the current device."""
function gpu_sm_count end

"""    gpu_max_threads_per_sm(backend) -> Int

Max resident threads per SM/CU — with `gpu_sm_count`, the device's total resident-thread
capacity (the denominator of thread-fill occupancy)."""
function gpu_max_threads_per_sm end

"""    gpu_arch(backend) -> String

Architecture tag of the current device: the compute capability (`"9.0"`) on NVIDIA, the
gfx name without feature suffixes (`"gfx942"`) on AMD. Provenance only."""
function gpu_arch end

"""    gpu_peak_fp64_flops(backend) -> Float64

Attainable vector (non-matrix/tensor) FP64 peak of the current device in FLOP/s, MEASURED on
the device with the dependent-FMA-chain probe of [`measure_peak_fp64_flops`](@ref) (any
KernelAbstractions backend, no per-architecture table) — or, on the CPU backend, BLAS
`LinearAlgebra.peakflops`. The denominator of a percent-of-peak figure for scalar FP64 kernels;
the matrix/tensor peak would be the wrong yardstick. Costs ~1.5 s of device time per call."""
gpu_peak_fp64_flops(backend::KA.Backend) = measure_peak_fp64_flops(backend)

# Fallbacks: a KA Backend with no vendor extension loaded → a clear "load the package" error.
# The extensions add more-specific methods (e.g. ::CUDABackend) that win over these.
for f in (
        :gpu_device_count, :gpu_device, :gpu_name, :gpu_power, :gpu_utilization,
        :gpu_memory_info, :gpu_sm_count, :gpu_max_threads_per_sm, :gpu_arch,
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
# The KA CPU backend is one "device": lets multi-device sharding drivers (and their tests) run
# on the CPU path — e.g. `devices = [1, 1]` shards work over two tasks on the same backend.
gpu_device_count(::KA.CPU) = 1
gpu_device(::KA.CPU) = 1
gpu_device!(::KA.CPU, ::Integer) = 1
gpu_name(::KA.CPU) = "CPU"
gpu_arch(::KA.CPU) = "cpu"
# Host: the SIMD gemm peak over all BLAS threads (best of 3) — what vectorised FP64 code can
# attain; a per-workitem scalar FMA chain would under-report the host by the SIMD width.
gpu_peak_fp64_flops(::KA.CPU) = LinearAlgebra.peakflops(2048; ntrials = 3)

gpu_telemetry_child_cmd(b::KA.Backend, ::AbstractVector{<:Integer}, ::Real, ::AbstractString) = error(
    "gpu_telemetry_child_cmd: no GPU vendor extension loaded for ", typeof(b),
    " — load CUDA.jl or AMDGPU.jl"
)

# ── Device-event kernel timing ──────────────────────────────────────────────────────────────
#
# Launch loops that do not synchronize per launch queue up asynchronously and the host runs
# ahead, so a host clock around a launch measures enqueue latency, not the kernel. Device events are the only instrument that sees kernel time in that regime: an event
# recorded on the launch stream fires when the GPU reaches it in stream order, so a pair around a
# launch brackets exactly that kernel — the START fires after any upload copy queued before it
# on the same stream has completed, the STOP after the kernel finishes. Two barrier
# packets per launch (microseconds) against kernels that run for seconds, no host stall, and no
# change to the kernel body. Both vendors timestamp events on the device (~µs resolution).

"""    gpu_event(backend) -> event

Record a timestamp event on the CURRENT TASK's stream (the one KernelAbstractions launches on)
and return it. Pair two with [`gpu_elapsed`](@ref). The CPU backend returns `time_ns()` — its
kernels are synchronous, so the host clock is the kernel clock."""
function gpu_event end

"""    gpu_elapsed(start, stop) -> Float64

Seconds between two events from [`gpu_event`](@ref). Waits for `stop` to complete first, so
it is safe to call before the stream is otherwise synchronized."""
function gpu_elapsed end

gpu_event(::KA.CPU) = time_ns()
gpu_elapsed(start::UInt64, stop::UInt64) = (stop - start) / 1.0e9
gpu_event(b::KA.Backend) = error(
    "gpu_event: no GPU vendor extension loaded for ", typeof(b), " — load CUDA.jl or AMDGPU.jl"
)

"""
    LaunchTimer()

Collects a device-event pair per kernel launch (see [`gpu_event`](@ref)), keyed by the 1-based
device the launch ran on. A launch loop instruments itself with [`launch_lane`](@ref) once per
call and [`launch_tick`](@ref) / [`launch_tock!`](@ref) around each launch; read back with
[`launch_times`](@ref) once the loop has returned. Safe to share across per-device tasks (pushes
are locked; events are per-stream). Passing `nothing` instead of a timer records nothing and
costs nothing.

    lane = launch_lane(timer, backend)
    for item in work
        e0 = launch_tick(timer, backend)
        launch_kernel!(item, backend)
        launch_tock!(timer, lane, backend, e0)
    end
    launch_times(timer)   # Dict(device => [seconds per launch, …])
"""
struct LaunchTimer
    lanes::Dict{Int, Vector{Tuple{Any, Any}}}   # device id ⇒ [(start, stop), …] in launch order
    lock::ReentrantLock
end
LaunchTimer() = LaunchTimer(Dict{Int, Vector{Tuple{Any, Any}}}(), ReentrantLock())

"""    launch_tick(timer, backend) -> event | nothing

Record the START event of a launch on the current task's stream (`nothing` timer ⇒ no-op)."""
launch_tick(::Nothing, backend) = nothing
launch_tick(::LaunchTimer, backend) = gpu_event(backend)

"""    launch_tock!(timer, lane, backend, e0)

Record the STOP event of a launch and file the pair `(e0, stop)` under device `lane` (from
[`launch_lane`](@ref)). Call on the launching task, right after the launch (`nothing` ⇒ no-op)."""
launch_tock!(::Nothing, dev, backend, e0) = nothing
function launch_tock!(t::LaunchTimer, dev::Integer, backend, e0)
    e1 = gpu_event(backend)
    lock(t.lock) do
        push!(get!(() -> Tuple{Any, Any}[], t.lanes, Int(dev)), (e0, e1))
    end
    return nothing
end
"""    launch_lane(timer, backend) -> Int

Device id the loop's launches are filed under; resolve once per loop, not per launch."""
launch_lane(::Nothing, backend) = 0
launch_lane(::LaunchTimer, backend) = Int(gpu_device(backend))

"""    launch_times(timer::LaunchTimer) -> Dict{Int, Vector{Float64}}

Per-device kernel seconds, one entry per launch in launch order. Waits on each launch's stop
event, so it is safe to call as soon as the loop has returned."""
function launch_times(t::LaunchTimer)
    return Dict{Int, Vector{Float64}}(
        d => Float64[gpu_elapsed(a, b) for (a, b) in pairs] for (d, pairs) in t.lanes
    )
end

"""
    thread_fill_occupancy(backend, n_threads) -> Float64

Thread-fill occupancy: the fraction of the current device's total resident-thread capacity
(`gpu_sm_count × gpu_max_threads_per_sm`) a launch of `n_threads` can fill (e.g. the pixel
count of a pixel-parallel kernel). This is an UPPER BOUND on achieved occupancy — per-thread registers /
shared memory cap it further; [`kernel_resources`](@ref) reports that compile-time bound.
"""
function thread_fill_occupancy(backend::KA.Backend, n_threads::Integer)
    capacity = gpu_sm_count(backend) * gpu_max_threads_per_sm(backend)

    return n_threads / capacity
end
