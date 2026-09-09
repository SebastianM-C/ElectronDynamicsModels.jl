# GPU-compatible cubic spline interpolation
# Extracts precomputed coefficients from DataInterpolations.CubicSpline
# and stores them in flat arrays suitable for GPU kernels.

"""
    GPUCubicSpline{V, M}

A GPU-compatible cubic spline interpolant. Stores precomputed coefficients
in flat arrays (works with both `Vector` and `CuArray`). The coefficient matrices are
column-major over the knots (`N × D`: for one component the knots are contiguous), because
the kernels' reuse of a cache line is along the knot index — consecutive samples of a pixel
and the lanes of a wave sit on the same or the neighbouring knot. The knot-major
alternative (`D × N`) issues fewer, wider loads but turns a cache line over every two knots
and measured 20 % slower on the MI300X; see the manual page "The device spline".

Evaluates the standard natural cubic spline formula:
    S(t) = (z[i]*dt2³ + z[i+1]*dt1³) / (6*h[i+1]) + c1[i]*dt1 + c2[i]*dt2
where dt1 = t - t[i], dt2 = t[i+1] - t.

# Fields
- `t`: knot positions (length N)
- `h`: interval widths, h[i] = t[i] - t[i-1] (length N, h[1] = 0)
- `z`: second derivatives at knots (N × D matrix, D = number of components)
- `c1`: precomputed linear coefficients per interval ((N-1) × D matrix)
- `c2`: precomputed linear coefficients per interval ((N-1) × D matrix)
"""
struct GPUCubicSpline{D, V, M}
    t::V        # knot times, length N
    h::V        # interval widths, length N (h[1] unused padding)
    z::M        # second derivatives, N × D
    c1::M       # linear coefficients, (N-1) × D
    c2::M       # linear coefficients, (N-1) × D
end

"""
    GPUCubicSpline(itp::DataInterpolations.CubicSpline)

Construct a `GPUCubicSpline` by extracting precomputed coefficients from
an existing `DataInterpolations.CubicSpline`.
"""
function GPUCubicSpline(itp::DataInterpolations.CubicSpline)
    t = collect(itp.t)
    h = collect(itp.h)
    N = length(t)
    D = length(first(itp.u))

    # Stack the Vector{SVector{D}} fields into N × D matrices via direct
    # nested writes.  `permutedims(reduce(hcat, …))` is ~5–10× slower at
    # N ≈ 10⁵ due to intermediate allocations and a transpose.
    Tel = eltype(t)
    z_mat = Matrix{Tel}(undef, N, D)
    c1 = Matrix{Tel}(undef, N - 1, D)
    c2 = Matrix{Tel}(undef, N - 1, D)

    @inbounds for i in 1:N
        zi = itp.z[i]
        for d in 1:D
            z_mat[i, d] = zi[d]
        end
    end

    # Precompute c1, c2 for each interval i = 1..N-1
    # c1[i] = u[i+1]/h[i+1] - z[i+1]*h[i+1]/6
    # c2[i] = u[i]/h[i+1]   - z[i]*h[i+1]/6
    # Note: h is 1-indexed with h[1]=0, h[i+1] = t[i+1] - t[i]
    @inbounds for i in 1:(N - 1)
        hi = h[i + 1]
        inv_hi = inv(hi)
        hi_over_6 = hi / 6
        ui = itp.u[i]
        uip1 = itp.u[i + 1]
        zi = itp.z[i]
        zip1 = itp.z[i + 1]
        for d in 1:D
            c1[i, d] = uip1[d] * inv_hi - zip1[d] * hi_over_6
            c2[i, d] = ui[d] * inv_hi - zi[d] * hi_over_6
        end
    end

    return GPUCubicSpline{D, typeof(t), typeof(z_mat)}(t, h, z_mat, c1, c2)
end

"""
    _searchsorted_left(t, x)

Binary search for the interval index: find largest `i` such that `t[i] ≤ x`.
Clamps to `[1, length(t)-1]` for evaluation safety.
GPU-compatible: no allocations, no dynamic dispatch.
"""
function _searchsorted_left(t, x)
    lo, hi = 1, length(t) - 1
    while lo < hi
        mid = (lo + hi + 1) >> 1   # round up to avoid infinite loop when hi = lo + 1
        if t[mid] ≤ x
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

"""
    _searchsorted_left(t, x, guess)

Warm-started variant: returns exactly what `_searchsorted_left(t, x)` returns (the largest
`i ∈ [1, length(t)-1]` with `t[i] ≤ x`, or 1), but starts from `guess`. If `x` lies in
`[t[guess], t[guess+1])` the answer is `guess` after two knot reads; otherwise the search
gallops from `guess` towards `x` (steps 1, 2, 4, …) and finishes with a binary search inside
the bracket it found — about 2·log₂(distance) reads instead of log₂(N). The per-pixel kernels
feed the interval of their previous evaluation back in: consecutive slots move the retarded
time by a fraction of a knot, so the fast path is the common case. The result is identical by
construction (same predicate, same bracket invariants), so kernel output does not change.
"""
function _searchsorted_left(t, x, guess::Integer)
    hi_max = length(t) - 1
    g = clamp(Int(guess), 1, hi_max)
    if t[g] ≤ x
        # answer ≥ g; fast path when x is below the next knot (or g is the last interval)
        (g == hi_max || t[g + 1] > x) && return g
        lo = g + 1      # t[lo] ≤ x
        hi = hi_max     # hi_max, or an index with t[hi + 1] > x (set below)
        step = 1
        while true
            nxt = lo + step
            nxt > hi_max && break
            if t[nxt] ≤ x
                lo = nxt
                step <<= 1
            else
                hi = nxt - 1
                break
            end
        end
    else
        # answer < g (or 1 when x precedes every knot)
        g == 1 && return 1
        hi = g - 1      # t[hi + 1] = t[g] > x
        lo = 1
        step = 1
        while true
            prv = hi - step
            prv < 1 && break
            if t[prv] ≤ x
                lo = prv
                break
            else
                hi = prv - 1
                step <<= 1
            end
        end
    end
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if t[mid] ≤ x
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

"""
    (spline::GPUCubicSpline)(τ)
    (spline::GPUCubicSpline)(τ, guess) -> (value, idx)

Evaluate the spline at time `τ`, returning an `SVector` of interpolated values. The two-argument
form starts the knot search at interval `guess` (see [`_searchsorted_left`](@ref)) and also
returns the interval it used, to be fed back as the next call's guess; its value is bit-identical
to the one-argument form.
"""
(spline::GPUCubicSpline)(τ) = _eval_at(spline, τ, _searchsorted_left(spline.t, τ))

function (spline::GPUCubicSpline)(τ, guess::Integer)
    idx = _searchsorted_left(spline.t, τ, guess)
    return _eval_at(spline, τ, idx), idx
end

# Spline value on a known interval `idx` (1 ≤ idx ≤ length(t) - 1): the arithmetic of the
# evaluation, shared by the cold and the warm-started calls.
_eval_at(spline::GPUCubicSpline, τ, idx) = _eval_poly(_fetch_interval(spline, idx), τ)

"""
    IntervalCoefs{D, T}

Everything a cubic-spline evaluation needs on one knot interval: its end knots, `inv(6h)` and
the four D-vectors `z[i]`, `z[i+1]`, `c1[i]`, `c2[i]` (3 + 4D scalars). [`_fetch_interval`](@ref)
loads it once; [`_eval_poly`](@ref) evaluates the cubic at any `τ` from it. The per-slot solvers
can hold it in registers across the Newton corrections (or RK4 stages) of a slot, which leave
the interval only at a knot crossing, instead of reloading the coefficients for every
evaluation (`coef_reuse = Val(true)` in the accumulate functions: bit-identical values, fewer
loads, more live registers).
"""
struct IntervalCoefs{D, T}
    t_i::T
    t_ip1::T
    inv_6h::T
    z_i::SVector{D, T}
    z_ip1::SVector{D, T}
    c1::SVector{D, T}
    c2::SVector{D, T}
end

@inline function _fetch_interval(spline::GPUCubicSpline{D}, idx) where {D}
    t_i = spline.t[idx]
    t_ip1 = spline.t[idx + 1]
    inv_6h = inv(6 * spline.h[idx + 1])
    z_i = SVector{D}(ntuple(d -> spline.z[idx, d], Val(D)))
    z_ip1 = SVector{D}(ntuple(d -> spline.z[idx + 1, d], Val(D)))
    c1 = SVector{D}(ntuple(d -> spline.c1[idx, d], Val(D)))
    c2 = SVector{D}(ntuple(d -> spline.c2[idx, d], Val(D)))
    return IntervalCoefs(t_i, t_ip1, inv_6h, z_i, z_ip1, c1, c2)
end

# The evaluation arithmetic — one definition for the cold, the warm-started and the
# register-cached paths, so all three produce bit-identical values.
@muladd @inline function _eval_poly(c::IntervalCoefs{D}, τ) where {D}
    dt1 = τ - c.t_i
    dt2 = c.t_ip1 - τ
    return SVector{D}(
        ntuple(Val(D)) do d
            c.z_i[d] * dt2^3 * c.inv_6h +
                c.z_ip1[d] * dt1^3 * c.inv_6h +
                c.c1[d] * dt1 +
                c.c2[d] * dt2
        end
    )
end

# τ inside the fetched interval ⇔ the warm-started search would return the same index.
@inline _in_interval(c::IntervalCoefs, τ) = (c.t_i ≤ τ) & (τ < c.t_ip1)

# ── Adapt.jl integration ─────────────────────────────────────────────

function Adapt.adapt_structure(to, spline::GPUCubicSpline{D}) where {D}
    t = Adapt.adapt(to, spline.t)
    z = Adapt.adapt(to, spline.z)
    return GPUCubicSpline{D, typeof(t), typeof(z)}(
        t,
        Adapt.adapt(to, spline.h),
        z,
        Adapt.adapt(to, spline.c1),
        Adapt.adapt(to, spline.c2),
    )
end

function Adapt.adapt_structure(to, traj::TrajectoryInterpolant)
    return TrajectoryInterpolant(
        Adapt.adapt(to, traj.itp),
        Adapt.adapt(to, traj.a_itp),   # `nothing` on the potential path (see `to_gpu`)
        traj.x_idxs,       # SVector{4,Int} — already isbits
        traj.u_idxs,        # SVector{4,Int} — already isbits
        traj.K,             # Float64 — already isbits
    )
end

# The GPU potential kernel never touches the acceleration spline, so by default
# we skip uploading it: carrying `a_itp = nothing` keeps the (latency- and
# memory-bound) GPUKernelRK4 potential campaign untouched. The field accumulator
# needs 𝔞μ for the Liénard–Wiechert radiation term, so it passes
# `with_acceleration = true` to also upload `GPUCubicSpline(traj.a_itp)` — at the
# cost of a second N×4 spline (≈ doubling the per-trajectory device footprint).
function to_gpu(traj::TrajectoryInterpolant; with_acceleration::Bool = false)
    # The kernels read x⁰…x³ = v[1:4], u⁰…u³ = v[5:8] with literal indices (radiation.jl).
    canonical_state_order(traj) || throw(ArgumentError(
        "to_gpu: the GPU kernels require the canonical state order x_idxs = 1:4, u_idxs = 5:8 " *
        "(got x_idxs = $(traj.x_idxs), u_idxs = $(traj.u_idxs)); build the trajectory with " *
        "TrajectoryInterpolant(sol, x_syms, u_syms) or arrange the spline components in that order"))
    if with_acceleration
        # The field kernels evaluate `a_itp` on the interval the warm-started search found for
        # `itp`, so both splines must sit on the same knots. `TrajectoryInterpolant(sol)` builds
        # them from the same `sol.t`; anything else is rejected up front.
        traj.a_itp.t == traj.itp.t || throw(ArgumentError(
            "to_gpu: the acceleration spline must share the trajectory spline's knots " *
            "($(length(traj.a_itp.t)) vs $(length(traj.itp.t)) knots, or different knot values)"))
    end
    a_itp = with_acceleration ? GPUCubicSpline(traj.a_itp) : nothing
    return TrajectoryInterpolant(GPUCubicSpline(traj.itp), a_itp, traj.x_idxs, traj.u_idxs, traj.K)
end
