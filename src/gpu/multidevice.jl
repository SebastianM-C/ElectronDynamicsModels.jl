# Multi-device electron sharding for accumulate_field. The radiated field is a LINEAR sum over
# electrons, so partitioning the electrons across D GPUs, accumulating each shard on its own device
# into that device's private buffers, and summing the D partials is EXACT — not an approximation.
# This parallelizes the otherwise-serial electron loop (the H200 is only ~0.59 occupied per launch,
# so a single device underuses it; D devices give a ~D× electron-loop speedup). Built on the vendor
# API: `gpu_device_count`/`gpu_device!` (ext/EDM{CUDA,AMDGPU}Ext.jl) pin each shard's task to a GPU.
#
# Host memory: the partials are STREAMED into one host cube set, not collected. Each device task
# hands its still-resident device buffers to a sink that, under a lock, downloads them chunk by
# chunk and adds them into the shared accumulator (`_add_fields!`); the first task to finish
# allocates the accumulator by a plain download (`_collect_fields`). Peak host residency is
# therefore ~1.1× one cube regardless of D — the previous collect-then-sum design held D cubes
# (8 × 97 GB for a 1101² capacity cell), which no node RAM survives. The lock serializes only the
# PCIe download + host add (seconds), not the accumulation.

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
                             devices = 1:gpu_device_count(backend), kwargs...)
        -> (; E, B[, E_far, B_far])

Shard `trajs` across `devices` (vendor-native 1-based ids) and run the single-device
[`accumulate_field`](@ref) on each shard CONCURRENTLY — one `Threads.@spawn` task per device, each
pinned with `gpu_device!(backend, d)` so its buffers + kernels land on that GPU — then sum the
per-device partials into ONE host cube set as each device finishes (streamed reduce; host peak
≈ 1.1 × cube, independent of the device count). `kwargs` (e.g. `mode`, `n_substeps`,
`sync_per_electron`) forward unchanged. The same device id may appear more than once (e.g.
`devices = [1, 1]`): the shards then time-share that GPU — pointless for throughput but the
exactness check the CPU-backend test relies on.

Needs ≥`length(devices)` Julia threads (`julia -t`): each per-device task is GPU-bound and blocks its
thread on the final device→host copy, so they only overlap on separate OS threads. Each device holds
a full prod-size buffer set (see the VRAM budget), so this trades device count for memory, not memory
for device count.
"""
function accumulate_field_sharded(
        trajs::Vector{<:TrajectoryInterpolant}, screen::ObserverScreen, alg, backend::KA.Backend;
        devices = 1:gpu_device_count(backend), kwargs...
    )
    nd = length(devices)
    nd >= 1 || throw(ArgumentError("accumulate_field_sharded: need ≥1 device, got $nd"))
    Threads.nthreads() >= nd ||
        @warn "accumulate_field_sharded: $(Threads.nthreads()) Julia thread(s) < $nd devices — \
               per-device tasks will serialize; rerun with julia -t$nd"

    shards = _shard_indices(length(trajs), nd)
    acc = Ref{Any}(nothing)
    lk = ReentrantLock()
    # The sink runs INSIDE accumulate_field, while the device buffers are alive; it returns
    # nothing so the task holds no host copy of its partial.
    sink = (E1, B1, E2, B2, mode) -> lock(lk) do
        if acc[] === nothing
            acc[] = _collect_fields(E1, B1, E2, B2, mode)
        else
            _add_fields!(acc[], E1, B1, E2, B2, mode)
        end
        nothing
    end
    @sync for (i, rng) in enumerate(shards)
        d = devices[i]
        Threads.@spawn begin
            gpu_device!(backend, d)
            accumulate_field(trajs[rng], screen, alg, backend; sink, kwargs...)
        end
    end

    return acc[]
end
