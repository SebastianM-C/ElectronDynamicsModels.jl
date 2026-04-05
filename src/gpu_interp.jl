# GPU-compatible cubic spline interpolation
# Extracts precomputed coefficients from DataInterpolations.CubicSpline
# and stores them in flat arrays suitable for GPU kernels.

import Adapt
import AcceleratedKernels as AK
import KernelAbstractions: Backend

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

    # itp.u is a Vector{SVector{D}} — stack into N × D matrix
    D = length(first(itp.u))
    u_mat = permutedims(reduce(hcat, itp.u))  # N × D

    # itp.z is similar structure to u
    z_mat = permutedims(reduce(hcat, itp.z))  # N × D

    # Precompute c1, c2 for each interval i = 1..N-1
    # c1[i] = u[i+1]/h[i+1] - z[i+1]*h[i+1]/6
    # c2[i] = u[i]/h[i+1]   - z[i]*h[i+1]/6
    # Note: h is 1-indexed with h[1]=0, h[i+1] = t[i+1] - t[i]
    c1 = similar(u_mat, N - 1, D)
    c2 = similar(u_mat, N - 1, D)
    for i in 1:(N - 1)
        hi = h[i + 1]
        inv_hi = inv(hi)
        hi_over_6 = hi / 6
        for d in 1:D
            c1[i, d] = u_mat[i + 1, d] * inv_hi - z_mat[i + 1, d] * hi_over_6
            c2[i, d] = u_mat[i, d] * inv_hi - z_mat[i, d] * hi_over_6
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

    return SVector{D}(ntuple(Val(D)) do d
        spline.z[idx, d] * dt2^3 * inv_6h +
        spline.z[idx + 1, d] * dt1^3 * inv_6h +
        spline.c1[idx, d] * dt1 +
        spline.c2[idx, d] * dt2
    end)
end

# ── Adapt.jl integration ─────────────────────────────────────────────

function Adapt.adapt_structure(to, spline::GPUCubicSpline{D}) where {D}
    t = Adapt.adapt(to, spline.t)
    z = Adapt.adapt(to, spline.z)
    GPUCubicSpline{D, typeof(t), typeof(z)}(
        t,
        Adapt.adapt(to, spline.h),
        z,
        Adapt.adapt(to, spline.c1),
        Adapt.adapt(to, spline.c2),
    )
end

function Adapt.adapt_structure(to, traj::TrajectoryInterpolant)
    TrajectoryInterpolant(
        Adapt.adapt(to, traj.itp),
        traj.x_idxs,       # SVector{4,Int} — already isbits
        traj.u_idxs,        # SVector{4,Int} — already isbits
        traj.K,             # Float64 — already isbits
    )
end

function to_gpu(traj::TrajectoryInterpolant)
    TrajectoryInterpolant(GPUCubicSpline(traj.itp), traj.x_idxs, traj.u_idxs, traj.K)
end

# ── GPU-accelerated accumulation ─────────────────────────────────────

"""
    accumulate_potential(trajs, screen, alg, backend::Backend; solve_kwargs...)

Compute the Liénard-Wiechert 4-potential using CPU retarded-time solve
and GPU-accelerated accumulation via AcceleratedKernels.

`backend` is a KernelAbstractions backend (e.g., `CUDA.CUDABackend()`).
Uses the original `trajs` (CubicSpline-based) for the CPU retarded-time solve,
and converts to `GPUCubicSpline` internally for the GPU accumulation phase.
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen, alg, backend::Backend;
        solve_kwargs...)
    x⁰_samples = screen.x⁰_samples
    N_samples = length(x⁰_samples)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    A = zeros(N_samples, 4, Nx, Ny)

    # Pre-allocate GPU buffers (reused across electrons)
    τ_buf = Adapt.adapt(backend, fill(NaN, N_samples, Nx, Ny))
    A_buf = Adapt.adapt(backend, zeros(N_samples, 4, Nx, Ny))

    # Pre-convert all trajectories to GPU once
    gpu_trajs = [Adapt.adapt(backend, to_gpu(traj)) for traj in trajs]

    τ_all = fill(NaN, N_samples, Nx, Ny)

    # Create typed integrator pool once (reused across all electrons)
    traj0 = first(trajs)
    τi0 = first(traj0.itp.t)
    τf0 = last(traj0.itp.t)
    r_obs_0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
    proto_prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs, τi0,
        (advanced_time(traj0, τi0, r_obs_0), advanced_time(traj0, τf0, r_obs_0)),
        (traj0, r_obs_0)
    )
    proto_integ = init(proto_prob, alg; saveat = x⁰_samples, solve_kwargs...)
    nworkers = Threads.nthreads()
    integ_pool = Channel{typeof(proto_integ)}(nworkers)
    put!(integ_pool, proto_integ)
    for _ in 2:nworkers
        put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, solve_kwargs...))
    end

    for (traj, gpu_traj) in zip(trajs, gpu_trajs)
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)

        # ── Phase 1: CPU retarded-time solve ──
        fill!(τ_all, NaN)

        Threads.@threads for ix in Base.OneTo(Nx)
            integ = take!(integ_pool)
            for iy in Base.OneTo(Ny)
                r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
                x⁰_i = advanced_time(traj, τi, r_obs)
                x⁰_f = advanced_time(traj, τf, r_obs)

                integ.p = (traj, r_obs)
                reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
                solve!(integ)

                sol_u = integ.sol.u::Vector{Float64}
                for k in eachindex(sol_u)
                    @inbounds τ_all[k, ix, iy] = sol_u[k]
                end
            end
            put!(integ_pool, integ)
        end

        # ── Phase 2: GPU accumulation (accumulates in A_buf across electrons) ──
        _gpu_accumulate_kernel!(gpu_traj, screen, τ_all, τ_buf, A_buf, backend)
    end

    # Single D2H transfer at the end
    copyto!(A, Array(A_buf))
    return A
end

function _gpu_accumulate_kernel!(gpu_traj, screen, τ_all_cpu, τ_buf, A_buf, backend)
    # Copy retarded times to pre-allocated GPU buffer (H2D)
    copyto!(τ_buf, τ_all_cpu)

    # Capture screen grids (isbits LinRange — no adaptation needed)
    x_grid = screen.x_grid
    y_grid = screen.y_grid
    z_screen = screen.z

    # Capture trajectory components
    spline = gpu_traj.itp
    x_idxs = gpu_traj.x_idxs
    u_idxs = gpu_traj.u_idxs
    K = gpu_traj.K

    # Pre-compute CartesianIndices outside closure
    CI = CartesianIndices(τ_buf)

    # Kernel accumulates into A_buf (no zeroing — accumulates across electrons)
    AK.foreachindex(τ_buf, backend) do i
        k, ix, iy = Tuple(CI[i])

        τ = τ_buf[k, ix, iy]
        isnan(τ) && return

        v = spline(τ)
        xμ = v[x_idxs]
        uμ = v[u_idxs]

        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)
        disp = r_obs - xμ[SA[2, 3, 4]]
        xr = SVector{4}(norm(disp), disp[1], disp[2], disp[3])

        coeff = K / m_dot(xr, uμ)
        for j in 1:4
            @inbounds A_buf[k, j, ix, iy] += coeff * uμ[j]
        end
    end
    return
end
