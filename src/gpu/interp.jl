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
    (spline::GPUCubicSpline)(τ)

Evaluate the spline at time `τ`, returning an `SVector` of interpolated values.
"""
function (spline::GPUCubicSpline{D})(τ) where {D}
    idx = _searchsorted_left(spline.t, τ)
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
        traj.x_idxs,       # SVector{4,Int} — already isbits
        traj.u_idxs,        # SVector{4,Int} — already isbits
        traj.K,             # Float64 — already isbits
    )
end

function to_gpu(traj::TrajectoryInterpolant)
    return TrajectoryInterpolant(GPUCubicSpline(traj.itp), traj.x_idxs, traj.u_idxs, traj.K)
end
