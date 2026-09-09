# The device spline: layout, search and the register-cached interval

The GPU kernels evaluate every electron's worldline through a natural cubic spline of its
state, `GPUCubicSpline`, built once per electron on the host from the trajectory's
`DataInterpolations.CubicSpline` and uploaded with the acceleration spline on the same knots.
This page states exactly what the device implementation does, why, and how it was measured, so
that benchmark manifests and the upstreaming of the interpolant can refer to one description.

## Storage: column-major `N × D`

`GPUCubicSpline{D}` holds the knot times `t` (length `N`), the interval widths `h`, the second
derivatives `z` and the two per-interval linear coefficients `c1`, `c2`, the last three as
`N × D` (`(N-1) × D`) matrices: for a fixed component `d` the knots are contiguous. The value at
`τ` on interval `i` (`t[i] ≤ τ < t[i+1]`) is the standard natural-spline form

```
S(τ) = (z[i]·dt2³ + z[i+1]·dt1³) / (6 h[i+1]) + c1[i]·dt1 + c2[i]·dt2,   dt1 = τ − t[i], dt2 = t[i+1] − τ
```

with `c1[i] = u[i+1]/h[i+1] − z[i+1]·h[i+1]/6` and `c2[i] = u[i]/h[i+1] − z[i]·h[i+1]/6`
precomputed on the host. Every evaluation therefore reads `t[i]`, `t[i+1]`, `h[i+1]` and the
four `D`-vectors `z[i]`, `z[i+1]`, `c1[i]`, `c2[i]`: `3 + 4D` scalars, 35 for the 8-component
state.

The layout is deliberate. The retarded proper time of one pixel advances a small fraction of a
knot per observer sample (about 0.03 knot in the benchmark cell), and adjacent pixels of a
wave sit on the same or the neighbouring knot, so the reuse of a cache line is *along the knot
index*: column-major puts 16 consecutive knots of one component in one 128-byte line, which a
wave's lanes and its next few hundred samples keep hitting. The knot-major alternative
(`D × N`, the array-of-structs layout an upstream `Vector{SVector{D}}` spline has) lets the
compiler merge the component loads into 16-byte accesses and issues 26 fewer load
instructions per pass, but it turns a cache line over every two knots; on the MI300X it
measured 20 % slower for the Newton kernel and 22 % slower for RK4 at the benchmark cell
(39 % more L1 accesses and twice the L2 read requests per sample), and no faster on the
W7900 or the RTX 5090. It is kept on the branch `feat/knot-major-spline` for reference.

## Interval search: warm-started, provably the same interval

`_searchsorted_left(t, x)` is the cold binary search for the largest `i` with `t[i] ≤ x`,
clamped to `[1, N-1]`. `_searchsorted_left(t, x, guess)` returns the same index by
construction: it checks the guessed interval (two reads), otherwise gallops away from it in
the direction of `x` and bisects the bracket it finds. The kernels keep the interval of the
previous evaluation of the same pixel as the guess (`spline(τ, guess) -> (value, idx)`), and
the acceleration spline is evaluated on the interval the state spline found (`to_gpu` refuses
an acceleration spline on different knots). Cubes are bit-identical to the cold search; the
time removed is the search's dependent loads, four to six per sample: 1.44× (MI300X) and
1.47× (H100) for the Newton kernel at the benchmark cell, 1.1× on the W7900 and the RTX 5090.

## The register-cached interval (`coef_reuse`)

The Newton corrections of one sample (and the RK4 stages of one step) evaluate the spline on
the same interval almost always. With `coef_reuse = Val(true)` on `accumulate_potential` /
`accumulate_field`, `_fetch_interval` loads the interval's `IntervalCoefs` once per sample
into registers and `_eval_poly` evaluates the cubic from them; an evaluation whose `τ` has
left the interval (`_in_interval`) refetches through the warm-started search. The cold path,
the warm-started path and the cached path share the one `_eval_poly` expression, so the GPU
cubes are bit-identical across all three (hashed on the W7900, the RTX 5090 and the MI300X for
both kernels); on the CPU the two `Val` specializations may fuse `muladd`s differently, which
`test/coef_reuse.jl` bounds at the last bit.

Measured at the benchmark strong cell, Newton n=2, ms per launch: MI300X 39.8 → 35.3 (1.13×;
load instructions per sample 113 → 55, L2 read requests 569 → 292), W7900 314 → 280 (1.11×),
RTX 5090 148 → 144 (1.03×). The cache costs registers: 128 with a few spills on gfx942 (whose
budget at the kernels' occupancy is 64 architectural + 64 accumulation registers), 140 on the
W7900, 155 on the 5090 (one block per SM instead of two). RK4 holds the interval across four
stages and spills 646 registers on the MI300X (85 ms against 50), so the cache is a loss there.

**Default policy in the solver scripts.** `EDM_COEF_REUSE` unset ⇒ on for the Newton kernel,
off for RK4; `EDM_COEF_REUSE=1|0` forces it either way. The manifest records the value under
`[config] coef_reuse`. The library kwarg defaults to `Val(false)`.

## What the kernels do with it

Both kernels walk one pixel's observer samples in order. The Newton kernel
(`GPUKernelNewton`, `n_iters` corrections from a warm start) solves the light-cone condition
per sample: one spline evaluation per correction plus one for the field, on the cached
interval. The RK4 kernel (`GPUKernelRK4`, `n_substeps`) marches the retarded time: six
evaluations per sample with one substep. The kernel choice is an environment knob
(`EDM_ACCUM_ALG=newton|rk4`, `EDM_NEWTON_ITERS`, `EDM_NSUBSTEPS`); for the γ = 5 benchmark cell
the two kernels' harmonic maps agree to a relative L2 of 5 × 10⁻⁷ at the backscatter line and
2 × 10⁻⁴ at its second harmonic (a uniform scale factor, not a change of structure).
