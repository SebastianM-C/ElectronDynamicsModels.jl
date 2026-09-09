# Multi-device electron sharding for accumulate_field. The radiated field is a LINEAR sum over
# electrons, so partitioning the electrons across D GPUs, accumulating each shard on its own device
# into that device's private buffers, and summing the D partials is EXACT — not an approximation.
# This parallelizes the otherwise-serial electron loop (the H200 is only ~0.59 occupied per launch,
# so a single device underuses it; D devices give a ~D× electron-loop speedup). Built on the vendor
# API: `gpu_device_count`/`gpu_device!` (lib/GPUDiagnostics) pin each shard's task to a GPU.
#
# The reduce. `reduce = :device` (default) sums the partials ON THE DEVICES: the first task to
# finish keeps its buffers resident as the accumulator, every later task folds its own buffers
# into them chunk by chunk over the device-to-device path (`_device_add!`; the vendor runtime
# routes each chunk over the peer link where the two GPUs can address each other and through
# host memory otherwise), and the host downloads ONE cube set at the end, with the
# device-layout → cube-layout permute spread over `reduce_workers` threads. Device memory: one
# staging chunk (1/16 of a buffer) on the accumulator device; host memory: one cube plus
# `reduce_workers`/16 of a cube in staging. The previous design, kept as `reduce = :host`,
# downloaded and permuted every partial on the host under a lock — a single-thread
# `permutedims!` of a full cube per device, which at eight devices took three times longer
# than the kernels themselves.

"""
    ReduceStats()

Wall-clock accounting of the multi-device reduce, filled in by [`accumulate_field_sharded`](@ref)
when passed as `reduce_stats`: `fold_s` is the time the finishing tasks spent folding their
partials into the accumulator (under the lock; device-to-device adds for `reduce = :device`,
host download-permute-add for `:host`), `n_folds` how many partials were folded, and
`download_s` the final download and permute into the host cube (`:device` only; the `:host`
path downloads as it folds). The solver scripts record them as `[timing] reduce_fold` /
`reduce_download`, next to `field` and `kernel`, so a sharded cell's non-kernel time is visible.
"""
mutable struct ReduceStats
    fold_s::Float64
    n_folds::Int
    download_s::Float64
end
ReduceStats() = ReduceStats(0.0, 0, 0.0)

# Near-even contiguous split of 1:n into k index ranges (first `rem` chunks get one extra).
function _shard_indices(n::Integer, k::Integer)
    base, rem = divrem(n, k)
    ranges = UnitRange{Int}[]
    start = 1
    for i in 1:k
        len = base + (i <= rem ? 1 : 0)
        len == 0 && continue
        push!(ranges, start:(start + len - 1))
        start += len
    end
    return ranges
end

"""
    accumulate_field_sharded(trajs, screen, alg, backend;
                             devices = 1:gpu_device_count(backend),
                             reduce = :device, reduce_workers = min(4, Threads.nthreads()),
                             kwargs...)
        -> (; E, B[, E_far, B_far])

Shard `trajs` across `devices` (vendor-native 1-based ids) and run the single-device
[`accumulate_field`](@ref) on each shard CONCURRENTLY — one `Threads.@spawn` task per device, each
pinned with `gpu_device!(backend, d)` so its buffers + kernels land on that GPU — then sum the
per-device partials into ONE host cube set. The sum is exact by linearity; only its summation
order differs from the single-device call (last-bit).

`reduce = :device` (default) sums on the GPUs: the first device to finish keeps its buffers as
the accumulator, the others fold theirs into it over the device-to-device path as they finish,
and one download + permute produces the host cube (the permute runs on `reduce_workers`
threads, each holding a 1/16-cube staging slab). `reduce = :host` is the streamed host reduce:
every partial is downloaded, permuted and added on the host under a lock (one cube resident,
one single-threaded full-cube permute per device). `reduce_stats = ReduceStats()` receives the
wall-clock time of the folds and of the final download (see [`ReduceStats`](@ref)).

Needs ≥`length(devices)` Julia threads (`julia -t`): each per-device task is GPU-bound and blocks its
thread on the final device→host copy, so they only overlap on separate OS threads. Each device holds
a full prod-size buffer set (see the VRAM budget), so this trades device count for memory, not memory
for device count. The same device id may appear more than once (e.g. `devices = [1, 1]`): the
shards then time-share that GPU — pointless for throughput but the exactness check the
CPU-backend test relies on.
"""
function accumulate_field_sharded(
        trajs::Vector{<:TrajectoryInterpolant}, screen::ObserverScreen, alg, backend::KA.Backend;
        devices = 1:gpu_device_count(backend), reduce::Symbol = :device,
        reduce_workers::Integer = min(4, Threads.nthreads()), reduce_stats::Union{Nothing, ReduceStats} = nothing,
        kwargs...
    )
    nd = length(devices)
    nd >= 1 || throw(ArgumentError("accumulate_field_sharded: need ≥1 device, got $nd"))
    reduce in (:device, :host) ||
        throw(ArgumentError("accumulate_field_sharded: reduce must be :device or :host, got $(repr(reduce))"))
    reduce_workers >= 1 || throw(ArgumentError("accumulate_field_sharded: reduce_workers must be ≥ 1"))
    Threads.nthreads() >= nd ||
        @warn "accumulate_field_sharded: $(Threads.nthreads()) Julia thread(s) < $nd devices — \
               per-device tasks will serialize; rerun with julia -t$nd"

    shards = _shard_indices(length(trajs), nd)
    lk = ReentrantLock()
    if reduce == :host
        acc = Ref{Any}(nothing)
        # The sink runs INSIDE accumulate_field, while the device buffers are alive; it returns
        # nothing so the task holds no host copy of its partial.
        sink = (E1, B1, E2, B2, mode) -> lock(lk) do
            t0 = time_ns()
            if acc[] === nothing
                acc[] = _collect_fields(E1, B1, E2, B2, mode)
                reduce_stats === nothing || (reduce_stats.download_s += (time_ns() - t0) / 1e9)
            else
                _add_fields!(acc[], E1, B1, E2, B2, mode)
                reduce_stats === nothing || (reduce_stats.fold_s += (time_ns() - t0) / 1e9; reduce_stats.n_folds += 1)
            end
            nothing
        end
        _run_shards(trajs, shards, devices, screen, alg, backend, sink; kwargs...)
        return acc[]
    end

    # reduce == :device — the first finisher's buffers stay resident (referenced from `part`)
    # and become the accumulator; later finishers add into them on that device.
    part = Ref{Any}(nothing)
    sink = (E1, B1, E2, B2, mode) -> lock(lk) do
        if part[] === nothing
            KA.synchronize(backend)   # the accumulator's own kernels must have landed before others add into it
            part[] = (; dev = gpu_device(backend), E1, B1, E2, B2, mode)
        else
            p = part[]
            p.mode == mode || error("accumulate_field_sharded: shards disagree on the field mode")
            t0 = time_ns()
            _device_add!(backend, p.dev, p.E1, E1)
            _device_add!(backend, p.dev, p.B1, B1)
            if mode == Val(:split)
                _device_add!(backend, p.dev, p.E2, E2)
                _device_add!(backend, p.dev, p.B2, B2)
            end
            reduce_stats === nothing || (reduce_stats.fold_s += (time_ns() - t0) / 1e9; reduce_stats.n_folds += 1)
        end
        nothing
    end
    _run_shards(trajs, shards, devices, screen, alg, backend, sink; kwargs...)
    p = part[]
    p === nothing && return nothing
    gpu_device!(backend, p.dev)
    t0 = time_ns()
    out = _collect_fields(p.E1, p.B1, p.E2, p.B2, p.mode; backend, dev = p.dev, workers = reduce_workers)
    reduce_stats === nothing || (reduce_stats.download_s += (time_ns() - t0) / 1e9)
    return out
end

function _run_shards(trajs, shards, devices, screen, alg, backend, sink; kwargs...)
    @sync for (i, rng) in enumerate(shards)
        d = devices[i]
        Threads.@spawn begin
            gpu_device!(backend, d)
            accumulate_field(trajs[rng], screen, alg, backend; sink, kwargs...)
        end
    end
    return nothing
end

# Chunk length for the device-side reduce: 1/16 of the buffer bounds both the staging
# allocation on the accumulator device and every transfer / broadcast well below 2³¹ elements.
_reduce_chunk(n::Integer) = max(1, cld(n, 16))

"""
    _device_add!(backend, dev, dst, src) -> dst

Fold `src`, resident on the calling task's current device, into `dst`, resident on device
`dev` (1-based vendor id), without a host copy of either. Same device (or the CPU backend): a
chunked in-place broadcast. Different devices: each 1/16 chunk is copied device-to-device into
a staging buffer on `dev` and added there; the vendor runtime routes the copy over the peer
link when the two GPUs can address each other and through host memory otherwise. Should the
direct copy be refused by the runtime, the chunk goes through a host slab instead. The calling
task's device is restored on return.
"""
function _device_add!(backend::KA.Backend, dev::Integer, dst::AbstractArray{T}, src::AbstractArray{T}) where {T}
    length(dst) == length(src) ||
        throw(DimensionMismatch("_device_add!: dst has $(length(dst)) elements, src $(length(src))"))
    n = length(dst)
    n == 0 && return dst
    chunk = _reduce_chunk(n)
    dstv, srcv = vec(dst), vec(src)
    sdev = gpu_device(backend)
    # The vendors order work per task AND per device stream, not across them: every kernel of
    # the calling task must have landed in `src` before it is read, and every chunk copy must
    # have landed in the staging buffer before the add on `dev` reads it. Hence the explicit
    # drains below (KA.synchronize drains the current task's stream on the CURRENT device).
    KA.synchronize(backend)
    if sdev == dev
        for k0 in 1:chunk:n
            r = k0:min(n, k0 + chunk - 1)
            view(dstv, r) .+= view(srcv, r)
        end
        KA.synchronize(backend)
        return dst
    end
    gpu_device!(backend, dev)
    stage = similar(dstv, chunk)
    KA.synchronize(backend)
    gpu_device!(backend, sdev)
    try
        hstage = nothing        # host slab, only if the direct device-to-device copy is refused
        for k0 in 1:chunk:n
            nk = min(chunk, n - k0 + 1)
            if hstage === nothing
                try
                    copyto!(stage, 1, srcv, k0, nk)       # issued from the source device's task stream
                    KA.synchronize(backend)               # … and drained there before `dev` reads it
                catch err
                    @warn "accumulate_field_sharded: device-to-device copy refused; staging the reduce through host memory" exception = (err, catch_backtrace()) maxlog = 1
                    hstage = Vector{T}(undef, chunk)
                end
            end
            if hstage !== nothing
                copyto!(hstage, 1, srcv, k0, nk)
                KA.synchronize(backend)
                gpu_device!(backend, dev)
                copyto!(stage, 1, hstage, 1, nk)
                gpu_device!(backend, sdev)
            end
            gpu_device!(backend, dev)
            view(dstv, k0:(k0 + nk - 1)) .+= view(stage, 1:nk)
            KA.synchronize(backend)                       # the add (and the staging reuse) complete on `dev`
            gpu_device!(backend, sdev)
        end
    finally
        gpu_device!(backend, sdev)
    end
    return dst
end
