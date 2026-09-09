# Electron batching: the host memory model of a field run

A field run has two very different memory footprints. On the **device** it is the cube: one set of
accumulation buffers, reused by every electron, `Nx × Ny × 3 × N_samples` per bucket. On the
**host** it used to be the trajectories: the driver solved the whole ensemble up front, kept every
electron's splines for the entire field phase, and uploaded them to the kernel one at a time.

Each electron carries a cubic spline of its 8-component state and one of its 4-acceleration, on the
same knots: about `3 + 4·8` doubles per knot for the state and `3 + 4·4` for the acceleration,
≈ 320 bytes per knot. At the campaign convention of 16 knots per proper-time carrier period a
production trajectory has ≈ 24 200 knots — **≈ 7.4 MB per electron** (measured: 17.0 GB of splines
for `N = 2000` above a 2.2 GB Julia baseline). At `N = 16 000` that is ≈ 118 GB of host memory that
must stay resident for the whole field phase, and the download then adds one full copy of the cube
(12.9 GB for a 401² × 1666 total-mode cube). On a 123 GB host the run reached 121 GB RSS and was
killed by the OOM killer at the end of a 44-minute field phase.

Nothing in the accumulation needs all the electrons at once.

## The knob

`EDM_ELECTRON_BATCH = B` makes `scripts/thomson_scattering.jl` and
`scripts/inverse_thomson_scattering.jl` solve and accumulate the electrons `B` at a time: solve a
batch, run its launches into device buffers that persist across batches, drop its splines, repeat;
the cube is downloaded and permuted once, at the end. `EDM_ELECTRON_BATCH = 0` (the default) is the
old single pass. The last batch is dropped (and its pages returned to the OS — the GC frees them
but glibc keeps them in its arenas, where they still count as RSS) *before* the download, so the
two big consumers never coexist:

```
during the batches:  ≈ B × 7.4 MB × (2 with the read-ahead)
at the download:     ≈ one cube copy + 1/16 of a cube in staging
host peak         =  max(the two) + the Julia baseline,   independent of N
```

Both terms are approximate from below: Julia's scheduler roots the last task that ran on each
thread, and glibc keeps freed chunks in its arenas (`malloc_trim`, which the loop calls, only
returns what the arenas can give back), so a batch's pages come back over the following batch
rather than immediately.

`EDM_ELECTRON_BATCH_OVERLAP = 0` turns off the read-ahead (the solve of batch k+1 runs during batch
k's launches) and drops that factor of 2 at the cost of leaving the CPU solve on the critical
path. The manifest records `[config] electron_batch`, keeps `[timing] field` as the
wall time of the accumulation phase (first launch → finished download) and `[timing] kernel` as the
device-event sum, reports `[timing] trajectories` as the SUM of the batch solve times, and adds
`[timing] trajectories_overlapped` — the part of it the loop never had to wait for because it ran
while the GPU was busy. (The launches are enqueued asynchronously, so a batch small enough to fit
whole in the launch queue reports ≈ 0 even though it did overlap; at production batch sizes the
enqueue throttles on the queue and the number is meaningful.)

## The API

Batching is a driver-level feature: the kernels, the per-electron uploads and the launch order are
untouched. [`accumulate_field`](@ref) takes two extra keywords:

```julia
acc = accumulate_field(trajs[1:500],    screen, alg, backend; finish = false)              # keep the buffers
acc = accumulate_field(trajs[501:1000], screen, alg, backend; buffers = acc, finish = false)
fld = accumulate_field(trajs[1001:1500], screen, alg, backend; buffers = acc)              # download once
```

`finish = false` returns the live [`FieldAccumulator`](@ref) — the device buffers, their `mode`,
their device and the running electron count — instead of the downloaded cube; passing it back as
`buffers` accumulates into the same buffers. [`finish_field`](@ref) downloads an accumulator
explicitly, without a further batch of electrons. [`accumulate_field_sharded`](@ref) takes the same
pair and keeps one [`ShardedFieldAccumulator`](@ref) per device: every batch is sharded over the
same devices, and the existing device reduce runs once, on the finishing call.

## Numerics

On one device a batched run is **bit-identical** to the single-pass run. The electron loop, the
uploads and the launch order do not change, and the launches of one stream are serialized, so the
same per-electron contributions are added to the same buffers in the same order.

The multi-device path is not bit-identical, and was not before: batching re-shards each batch
across the devices, so a given device sums a different subset than it would in one pass. Floating-
point addition is not associative, so the cube moves in the last bits — the same tolerance the
sharded path already carries against the single-device call (relative L2 < 1e-12; measured ~1e-16
in `test/gpu_radiation.jl`).

The host-side per-batch products fold exactly: the window-coverage check is per electron, so the
batches' records concatenate; the γ(τ) trace reduces as batch sums and elementwise extrema.

## Measured

The benchmark strong cell scaled down to `N = 2000` (γ = 5, 401², 1666 samples, total mode,
Newton `n_iters = 2`, `EDM_COEF_REUSE=1`, `EDM_SAMPLE_CHUNKS=4`) on a W7900, RSS sampled every 2 s:

| | `EDM_ELECTRON_BATCH=0` | `=500` |
|---|---|---|
| peak RSS | 26.1 GiB | **18.4 GiB** |
| RSS during the field phase | 13.4 GiB (all N splines) | 6.0–9.7 GiB (≤ 2 batches) |
| `[timing] field` | 600.3 s | 605.4 s |
| `[timing] kernel` median per launch | 288.9 ms | 289.3 ms |
| `[timing] trajectories` | 43.1 s | 47.4 s (31.7 s overlapped) |
| wall | 731 s | 709 s |

The plateau is the part that scales with `N`: at `N = 16 000` the unbatched run holds ~92 GB of
splines there, which is what filled a 123 GB host.
