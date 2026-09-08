# GPU-compatible cubic spline interpolation
# Extracts precomputed coefficients from DataInterpolations.CubicSpline
# and stores them in flat arrays suitable for GPU kernels.

"""
    GPUCubicSpline{V, M}

A GPU-compatible cubic spline interpolant. Stores precomputed coefficients
in flat arrays (works with both `Vector` and `CuArray`).

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
`i ∈ [1, length(t)-1]` with `t[i] ≤ x`, or 1), but starts from `guess`. It is
FindFirstFunctions' `BracketGallop` search, the same correlated lookup DataInterpolations uses
on the CPU: a bracket is expanded from `guess` towards `x` (steps 1, 2, 4, …) and bisected, so a
query that stays in or next to the previous interval costs two or three knot reads instead of
log₂(N). The per-pixel kernels feed the interval of their previous evaluation back in;
consecutive slots move the retarded time by a fraction of a knot, so that is the common case.
The result is identical by construction (same predicate, `searchsortedlast` semantics clamped
to the evaluation range), so kernel output does not change.
"""
@inline function _searchsorted_left(t, x, guess::Integer)
    return clamp(searchsorted_last(BracketGallop(), t, x, Int(guess)), 1, length(t) - 1)
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
@muladd function _eval_at(spline::GPUCubicSpline{D}, τ, idx) where {D}
    h_idx = spline.h[idx + 1]
    dt1 = τ - spline.t[idx]
    dt2 = spline.t[idx + 1] - τ
    inv_6h = inv(6 * h_idx)

    return SVector{D}(
        ntuple(Val(D)) do d
            spline.z[idx, d] * dt2^3 * inv_6h +
                spline.z[idx + 1, d] * dt1^3 * inv_6h +
                spline.c1[idx, d] * dt1 +
                spline.c2[idx, d] * dt2
        end
    )
end

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
