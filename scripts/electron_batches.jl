# Electron batching for the field solver scripts (thomson_scattering.jl and its inverse sibling):
# solve a batch of trajectories → accumulate it on the GPU → drop it, instead of solving every
# electron up front and holding all of them for the whole field phase. Included, not a module —
# the includer provides ElectronDynamicsModels; same contract as gpu_telemetry.jl.
#
# WHY. The device side of a field run is bounded by the cube (one buffer set, reused across
# electrons); the HOST side used to scale with N. Each electron's trajectory is kept as a cubic
# spline of its 8-component state plus a 4-component acceleration spline: at the campaign
# convention of 16 knots per proper-time carrier period that is ≈ 24 200 knots ≈ 7.4 MB per
# electron (measured: 17.0 GB of splines for N = 2000 over a 2.2 GB baseline). At N = 16 000 the
# splines alone are ≈ 118 GB, and the download then adds a full cube copy — a 401² × 1666 total-mode
# cube is 12.9 GB — which is how a 44-minute field phase ended in the OOM killer on a 123 GB host.
# Nothing in the accumulation needs all the electrons at once: the kernels upload ONE trajectory at
# a time. `EDM_ELECTRON_BATCH = B` therefore solves and accumulates the electrons B at a time into
# device buffers that persist across the batches (`FieldAccumulator`), and downloads the cube once,
# at the end — after the last batch has been dropped, so the cube copy never lands on top of a
# batch of splines. Host peak becomes max(B × 7.4 MB × (2 while the next batch is solved ahead),
# one cube copy) + the baseline, independent of N.
#
# `EDM_ELECTRON_BATCH = 0` (the default) keeps the single-pass behaviour, bit-identical to before.
# With one device a batched run is bit-identical too — the electron loop, the uploads and the
# launch order are unchanged, so the same contributions are summed in the same order; only the
# multi-device path re-shards per batch, which moves the sum by the last bits.
const ELECTRON_BATCH = parse(Int, get(ENV, "EDM_ELECTRON_BATCH", "0"))
ELECTRON_BATCH >= 0 || error("EDM_ELECTRON_BATCH must be ≥ 0 (0 = single pass), got $ELECTRON_BATCH")
# Solve batch k+1 while batch k's launches run (two batches of splines resident instead of one).
# EDM_ELECTRON_BATCH_OVERLAP=0 serializes them for the tightest possible host peak.
const ELECTRON_BATCH_OVERLAP = get(ENV, "EDM_ELECTRON_BATCH_OVERLAP", "1") == "1"

# Contiguous batch ranges over 1:N (`batch ≤ 0` or `batch ≥ N` ⇒ the single range 1:N).
electron_batch_ranges(N::Integer, batch::Integer) =
    (batch <= 0 || batch >= N) ? [1:N] : [i:min(i + batch - 1, N) for i in 1:batch:N]

# Release a finished batch's splines to the OS. Julia's GC frees them, but glibc keeps the freed
# chunks in its arenas, so RSS — the number the OOM killer reads — stays at the high-water mark
# unless the allocator is asked to give the pages back. `malloc_trim` is glibc-only; anywhere else
# the ccall simply fails and the GC alone has to do.
function release_batch_memory()
    GC.gc()
    Sys.islinux() && try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch
    end
    return nothing
end

"""
    run_electron_batches(ranges, solve_batch, accumulate_batch, finish_accumulation;
                         primed = nothing, primed_s = 0.0, overlap = true)
        -> (result, (; solve_s, overlapped_s, n_batches))

Drive the solve → accumulate → discard loop over `ranges`.

`solve_batch(rng)` returns whatever the accumulation needs for the electrons `rng` (the batch's
`TrajectoryInterpolant`s and any per-batch host-side products); `accumulate_batch(batch, b)` runs
that batch's GPU launches into the persistent device buffers; `finish_accumulation()` downloads the
cube once, at the end. The batch handed to `accumulate_batch` is dropped as soon as it returns and
its pages are returned to the OS before the next one is accumulated, so the host holds one batch
(two with `overlap`, which spawns the solve of batch k+1 before accumulating batch k) — and none at
all during the download, which is where the cube copy lands.

`primed` is a `Ref` holding an already-solved first batch (`primed_s` = the seconds it took): the
caller solves batch 1 outside its own field-phase timer, so `[timing].field` still measures first
launch → finished download. The `Ref` is emptied here, so the caller keeps no handle on batch 1 —
a global binding to it would pin that batch's splines for the whole run and undo the batching.

`solve_s` is the SUM of the batch solve times; `overlapped_s` how much of it the loop never had to
wait for because it ran while the GPU was busy. The launches are enqueued asynchronously, so a
batch small enough to fit whole in the launch queue reports ≈ 0 even though it did overlap; at
production batch sizes the enqueue throttles on the queue and the number is meaningful.
"""
function run_electron_batches(ranges, solve_batch, accumulate_batch, finish_accumulation;
        primed = nothing, primed_s::Real = 0.0, overlap::Bool = true)
    nb = length(ranges)
    solve_s = 0.0
    overlapped_s = 0.0
    spawn_solve(b) = Threads.@spawn begin
        t0 = time_ns()
        v = solve_batch(ranges[b])
        (v, (time_ns() - t0) / 1.0e9)
    end
    pending = if primed === nothing
        spawn_solve(1)
    else
        p = (primed[], Float64(primed_s))
        primed[] = nothing   # the caller's handle on batch 1 goes away with the first fetch
        p
    end
    for b in 1:nb
        t_wait = time_ns()
        spawned = pending isa Task
        batch, s = spawned ? fetch(pending) : pending
        waited = (time_ns() - t_wait) / 1.0e9
        solve_s += s
        # Only a batch solved by this loop can have hidden behind GPU work; the primed batch 1
        # ran before the field phase started, so none of its time was overlapped.
        spawned && (overlapped_s += max(0.0, s - waited))
        pending = (overlap && b < nb) ? spawn_solve(b + 1) : nothing
        accumulate_batch(batch, b)
        batch = nothing
        overlap || b == nb || (pending = spawn_solve(b + 1))
        release_batch_memory()   # the accumulated batch's splines go back to the OS here
    end
    return finish_accumulation(), (; solve_s, overlapped_s, n_batches = nb)
end

# Fold the per-batch `window_coverage` results of a batched run into the single summary the
# manifest's [window] and [flops] sections expect. The per-electron records are exact and
# independent, so concatenating them in batch order and re-reducing reproduces what one
# whole-ensemble check would have returned (`worst_electron` included, since the batches are
# contiguous and in order). `nothing` entries (a failed check) are skipped; all-nothing ⇒ nothing.
function merge_window_coverage(covs)
    live = [c for c in covs if c !== nothing]
    isempty(live) && return nothing
    length(live) == 1 && return live[1]
    per = reduce(vcat, (c.per_electron for c in live))
    isempty(per) && return live[1]
    ref = first(c for c in live if !isempty(c.per_electron))   # slots per electron, from a non-empty batch
    slots_px = ref.slots_nominal ÷ length(ref.per_electron)
    clipped = count(p -> p.slots_executed != slots_px, per)
    known = all(p -> p.slots_executed >= 0, per)
    slots_nominal = length(per) * slots_px
    slots_executed = known ? sum(p -> p.slots_executed, per; init = 0) : missing
    return (;
        ok = clipped == 0,
        slots_nominal,
        slots_executed,
        slot_fill = known ? slots_executed / slots_nominal : missing,
        electrons_clipped = clipped,
        slots_dropped = known ? slots_nominal - slots_executed : missing,
        lead_margin_samples = minimum(p.lead_margin for p in per),
        tail_margin_samples = minimum(p.tail_margin for p in per),
        worst_electron = argmin(p.tail_margin for p in per),
        N_samples = ref.N_samples,
        per_electron = per,
    )
end
