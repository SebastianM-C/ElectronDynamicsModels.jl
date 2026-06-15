# Multi-device electron sharding for accumulate_field. The radiated field is a LINEAR sum over
# electrons, so partitioning the electrons across D GPUs, accumulating each shard on its own device
# into that device's private buffers, and summing the D partials is EXACT — not an approximation.
# This parallelizes the otherwise-serial electron loop (the H200 is only ~0.59 occupied per launch,
# so a single device underuses it; D devices give a ~D× electron-loop speedup). Built on the vendor
# API: `gpu_device_count`/`gpu_device!` (ext/EDM{CUDA,AMDGPU}Ext.jl) pin each shard's task to a GPU.

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
per-device partials. `kwargs` (e.g. `mode`, `n_substeps`, `sync_per_electron`) forward unchanged.

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
    partials = Vector{Any}(undef, length(shards))
    @sync for (i, rng) in enumerate(shards)
        d = devices[i]
        Threads.@spawn begin
            gpu_device!(backend, d)
            partials[i] = accumulate_field(trajs[rng], screen, alg, backend; kwargs...)
        end
    end

    return _reduce_partials(partials)
end

# Combine the per-device partials (each a NamedTuple `(; E, B[, E_far, B_far])` of host arrays, all
# the same fields/shape) into a single NamedTuple by summing each field across devices.
function _reduce_partials(partials)
    nd = length(partials)
    for i in 2:nd
        for k in propertynames(partials[1])
            getproperty(partials[1], k) .+= getproperty(partials[i], k)
        end
    end

    return partials[1]
end
