# ── Newton light-cone GPU kernel: per-slot root solve + LW accumulation ──
# Lives in the `Experimental` submodule (production staging, like the batched
# Tsit5 path) and extends the parent `accumulate_potential`/`accumulate_field`.
#
# Alternative to the GPUKernelRK4 march (kernel_rk4.jl): instead of
# integrating dτ_r/dx⁰ = 1/(u⁰ − u⃗·n̂) between saveat slots, solve the light-cone
# condition (FFT_v2.pdf eq. 1.2)
#     f(τ) = x⁰_target − x⁰(τ) − |r_obs − x⃗(τ)|  =  0
# at each slot with warm-started Newton corrections.  f is strictly monotone
# decreasing — f′(τ) = −(u⁰ − u⃗·n̂) < 0 since u⁰ ≥ |u⃗| on a timelike worldline —
# so the root is unique and the Newton derivative comes for free from the same
# spline eval that provides the residual.  Unlike the RK4 march, the per-slot
# error does not accumulate along the observer-time grid.
#
# Measured on the W7900 (a₀ = 0.1 production setup, see
# reports/newton_lightcone_retarded_time.typ): n_iters = 1 sits at the accuracy
# floor (9.4e-11 vs tight reference; RK4 n_substeps = 1 is 9.5e-10) and runs
# 1.5–2.0× faster than RK4 n_substeps = 1 on the potential path, time-neutral
# to ~1.3× faster on the field path.  Strongly-relativistic forward-scattering
# geometries need n_iters = 3 (the analog of RK4 needing n_substeps = 8 there).

# Shared deps: experimental.jl (included first in this module) already imports
# Adapt, AK, KernelAbstractions, Backend, StaticArrays, TrajectoryInterpolant,
# ObserverScreen, and accumulate_potential.  This file additionally needs:
using ..ElectronDynamicsModels: to_gpu, lienard_wiechert_F_split, extract_EB
import ..ElectronDynamicsModels: accumulate_field

"""
    GPUKernelNewton

Sentinel solver type selecting the Newton light-cone GPU kernel path: the
retarded proper time at each saveat slot is obtained by solving the light-cone
condition `x⁰_target − x⁰(τ) − |r_obs − x⃗(τ)| = 0` with `n_iters` warm-started
Newton corrections, instead of marching the retarded-time ODE as
[`GPUKernelRK4`](@ref) does.
"""
struct GPUKernelNewton end

# One spline eval → light-cone residual f, Newton factor rhs = −1/f′, and the
# geometric pieces the accumulation reuses.  The Newton update is
# τ ← τ + f·rhs with rhs = 1/(u⁰ − u⃗·n̂) = r_norm/xr_dot_u > 0, and the final
# (converged) eval doubles as the accumulation eval: `v` carries the full
# [xμ; uμ] state and K/xr_dot_u = K·rhs/r_norm.
@inline function _lightcone_eval(τ, gpu_traj, r_obs, x⁰_target)
    v = gpu_traj.itp(τ)
    x⁰ = v[gpu_traj.x_idxs[1]]
    d¹ = r_obs[1] - v[gpu_traj.x_idxs[2]]
    d² = r_obs[2] - v[gpu_traj.x_idxs[3]]
    d³ = r_obs[3] - v[gpu_traj.x_idxs[4]]
    r_norm = sqrt(d¹ * d¹ + d² * d² + d³ * d³)
    u⁰ = v[gpu_traj.u_idxs[1]]
    u¹ = v[gpu_traj.u_idxs[2]]
    u² = v[gpu_traj.u_idxs[3]]
    u³ = v[gpu_traj.u_idxs[4]]
    # m_dot(xr, uμ) with xr = (r_norm, d¹, d², d³)
    xr_dot_u = r_norm * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³)
    rhs = r_norm / xr_dot_u
    f = x⁰_target - x⁰ - r_norm
    return v, f, rhs, r_norm, d¹, d², d³
end

# Per-electron AK.foreachindex pass over (Nx × Ny) pixels; same closure pattern
# as _gpu_unified_one_electron! (see the @kernel note at the top of
# kernel_rk4.jl).  Window computation copied verbatim from the RK4 kernel; the
# RK4 bridge+march is replaced by an Euler predictor + fixed-count Newton
# corrections per slot (uniform trip count → no warp divergence).
function _gpu_newton_one_electron!(
        A_buf, gpu_traj,
        x_grid, y_grid, z_screen,
        x⁰_first, δx⁰, N_samples, Nx, Ny,
        τi, τf, pixel_iter, backend, n_iters
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Column-major unpacking: ix runs fastest, iy outer
        ix = ((i_lin - 1) % Nx) + 1
        iy = ((i_lin - 1) ÷ Nx) + 1
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Pixel-specific advanced-time window x⁰_i, x⁰_f
        v_i = gpu_traj.itp(τi)
        d_i¹ = r_obs[1] - v_i[gpu_traj.x_idxs[2]]
        d_i² = r_obs[2] - v_i[gpu_traj.x_idxs[3]]
        d_i³ = r_obs[3] - v_i[gpu_traj.x_idxs[4]]
        R_i = sqrt(d_i¹^2 + d_i²^2 + d_i³^2)
        x⁰_i_px = v_i[gpu_traj.x_idxs[1]] + R_i

        v_f = gpu_traj.itp(τf)
        d_f¹ = r_obs[1] - v_f[gpu_traj.x_idxs[2]]
        d_f² = r_obs[2] - v_f[gpu_traj.x_idxs[3]]
        d_f³ = r_obs[3] - v_f[gpu_traj.x_idxs[4]]
        x⁰_f_px = v_f[gpu_traj.x_idxs[1]] + sqrt(d_f¹^2 + d_f²^2 + d_f³^2)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))

        if k_start > k_end
            return
        end

        # Warm start at the window edge: τ(x⁰_i_px) = τi exactly, and the v_i
        # eval above already provides rhs(τi) for the first predictor stride —
        # no RK4 bridge needed.
        u⁰_i = v_i[gpu_traj.u_idxs[1]]
        u¹_i = v_i[gpu_traj.u_idxs[2]]
        u²_i = v_i[gpu_traj.u_idxs[3]]
        u³_i = v_i[gpu_traj.u_idxs[4]]
        rhs = R_i / (R_i * u⁰_i - (d_i¹ * u¹_i + d_i² * u²_i + d_i³ * u³_i))
        τ = τi
        x⁰_target = x⁰_first + (k_start - 1) * δx⁰
        Δ = x⁰_target - x⁰_i_px   # ∈ (0, δx⁰] unless k_start clamped to 1

        for k in k_start:k_end
            # Euler predictor from the previous slot's converged state, then
            # fixed-count Newton corrections on the light-cone residual.
            τ = clamp(τ + Δ * rhs, τi, τf)
            v, f, rhs, r_norm, d¹, d², d³ = _lightcone_eval(τ, gpu_traj, r_obs, x⁰_target)
            for _ in 1:n_iters
                τ = clamp(τ + f * rhs, τi, τf)
                v, f, rhs, r_norm, d¹, d², d³ = _lightcone_eval(τ, gpu_traj, r_obs, x⁰_target)
            end

            # Accumulate from the last residual eval — zero extra spline evals.
            coeff = K * rhs / r_norm   # = K / m_dot(xr, uμ)
            @inbounds A_buf[ix, iy, 1, k] += coeff * v[gpu_traj.u_idxs[1]]
            @inbounds A_buf[ix, iy, 2, k] += coeff * v[gpu_traj.u_idxs[2]]
            @inbounds A_buf[ix, iy, 3, k] += coeff * v[gpu_traj.u_idxs[3]]
            @inbounds A_buf[ix, iy, 4, k] += coeff * v[gpu_traj.u_idxs[4]]

            x⁰_target += δx⁰
            Δ = δx⁰
        end
    end
    return
end

"""
    accumulate_potential(trajs, screen, ::GPUKernelNewton, backend; n_iters = 2, sync_per_electron = true)

Newton light-cone GPU path: like the [`GPUKernelRK4`](@ref) unified kernel, but
the retarded proper time at each saveat slot is found by solving the light-cone
condition `x⁰_target − x⁰(τ) − |r_obs − x⃗(τ)| = 0` directly (`n_iters`
warm-started Newton corrections per slot) instead of integrating the
retarded-time ODE between slots.

Spline-eval budget per slot is `n_iters + 1` (the final residual eval doubles
as the accumulation eval), vs `4·n_substeps + 1` for the RK4 march; and the
per-slot error does not accumulate along the march, so accuracy is set by the
convergence of the last Newton step alone.

`sync_per_electron` as in the `GPUKernelRK4` method.
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelNewton,
        backend::Backend;
        n_iters::Int = 2,
        sync_per_electron::Bool = true,
    )
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)

    # Pixel-fastest layout (Nx, Ny leading) so warp-adjacent pixel-threads
    # (consecutive ix) write consecutive addresses → coalesced accumulation.
    A_buf = Adapt.adapt(backend, zeros(Nx, Ny, 4, N_samples))

    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)

    # Iteration target: one element per pixel. Sentinel array; never read.
    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny))

    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        _gpu_newton_one_electron!(
            A_buf, gpu_traj,
            screen.x_grid, screen.y_grid, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_iters,
        )
        sync_per_electron && KernelAbstractions.synchronize(backend)
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    return permutedims(Array(A_buf), (4, 3, 1, 2))
end

# ── Field variant of the Newton light-cone kernel ──
# Identical per-slot Newton solve to `_gpu_newton_one_electron!`; each slot then
# writes the split Liénard–Wiechert (E, B) exactly as
# `_gpu_unified_field_one_electron!` does, reusing the converged residual eval
# (`v`, `d`, `r_norm`) for the geometry, plus the unavoidable `a_itp` eval.
function _gpu_newton_field_one_electron!(
        mode::Val, E1_buf, B1_buf, E2_buf, B2_buf, gpu_traj, c,
        x_grid, y_grid, z_screen,
        x⁰_first, δx⁰, N_samples, Nx, Ny,
        τi, τf, pixel_iter, backend, n_iters
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Column-major unpacking: ix runs fastest, iy outer
        ix = ((i_lin - 1) % Nx) + 1
        iy = ((i_lin - 1) ÷ Nx) + 1
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Pixel-specific advanced-time window x⁰_i, x⁰_f
        v_i = gpu_traj.itp(τi)
        d_i¹ = r_obs[1] - v_i[gpu_traj.x_idxs[2]]
        d_i² = r_obs[2] - v_i[gpu_traj.x_idxs[3]]
        d_i³ = r_obs[3] - v_i[gpu_traj.x_idxs[4]]
        R_i = sqrt(d_i¹^2 + d_i²^2 + d_i³^2)
        x⁰_i_px = v_i[gpu_traj.x_idxs[1]] + R_i

        v_f = gpu_traj.itp(τf)
        d_f¹ = r_obs[1] - v_f[gpu_traj.x_idxs[2]]
        d_f² = r_obs[2] - v_f[gpu_traj.x_idxs[3]]
        d_f³ = r_obs[3] - v_f[gpu_traj.x_idxs[4]]
        x⁰_f_px = v_f[gpu_traj.x_idxs[1]] + sqrt(d_f¹^2 + d_f²^2 + d_f³^2)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))

        if k_start > k_end
            return
        end

        u⁰_i = v_i[gpu_traj.u_idxs[1]]
        u¹_i = v_i[gpu_traj.u_idxs[2]]
        u²_i = v_i[gpu_traj.u_idxs[3]]
        u³_i = v_i[gpu_traj.u_idxs[4]]
        rhs = R_i / (R_i * u⁰_i - (d_i¹ * u¹_i + d_i² * u²_i + d_i³ * u³_i))
        τ = τi
        x⁰_target = x⁰_first + (k_start - 1) * δx⁰
        Δ = x⁰_target - x⁰_i_px

        for k in k_start:k_end
            τ = clamp(τ + Δ * rhs, τi, τf)
            v, f, rhs, r_norm, d¹, d², d³ = _lightcone_eval(τ, gpu_traj, r_obs, x⁰_target)
            for _ in 1:n_iters
                τ = clamp(τ + f * rhs, τi, τf)
                v, f, rhs, r_norm, d¹, d², d³ = _lightcone_eval(τ, gpu_traj, r_obs, x⁰_target)
            end

            # Field write from the converged eval (X reuses r_norm and d).
            uμ = v[gpu_traj.u_idxs]
            𝔞μ = gpu_traj.a_itp(τ)
            X = SVector{4}(r_norm, d¹, d², d³)
            F_near, F_far = lienard_wiechert_F_split(X, uμ, 𝔞μ, K, c)
            E_near, B_near = extract_EB(F_near, c)
            E_far, B_far = extract_EB(F_far, c)

            for j in 1:3
                if mode === Val(:split)
                    @inbounds E1_buf[ix, iy, j, k] += E_far[j]
                    @inbounds B1_buf[ix, iy, j, k] += B_far[j]
                    @inbounds E2_buf[ix, iy, j, k] += E_near[j]
                    @inbounds B2_buf[ix, iy, j, k] += B_near[j]
                else
                    @inbounds E1_buf[ix, iy, j, k] += E_far[j] + E_near[j]
                    @inbounds B1_buf[ix, iy, j, k] += B_far[j] + B_near[j]
                end
            end

            x⁰_target += δx⁰
            Δ = δx⁰
        end
    end
    return
end

"""
    accumulate_field(trajs, screen, ::GPUKernelNewton, backend; n_iters = 2, mode = Val(:split), sync_per_electron = true)

Field counterpart of the `GPUKernelNewton` [`accumulate_potential`](@ref)
method: per-slot Newton light-cone solve instead of the RK4 retarded-time
march, otherwise identical in buffers, `mode`, and streaming to the
`GPUKernelRK4` [`accumulate_field`](@ref) method.
"""
function accumulate_field(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelNewton,
        backend::Backend;
        n_iters::Int = 2,
        mode::Val = Val(:split),
        sync_per_electron::Bool = true,
    )
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)
    c = screen.c

    E1_buf = Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples))
    B1_buf = Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples))
    if mode == Val(:split)
        E2_buf = Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples))
        B2_buf = Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples))
    else
        E2_buf = E1_buf
        B2_buf = B1_buf
    end

    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)

    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny))

    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj; with_acceleration = true))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        _gpu_newton_field_one_electron!(
            mode, E1_buf, B1_buf, E2_buf, B2_buf, gpu_traj, c,
            screen.x_grid, screen.y_grid, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_iters,
        )
        sync_per_electron && KernelAbstractions.synchronize(backend)
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
        finalize(gpu_traj.a_itp.t)
        finalize(gpu_traj.a_itp.h)
        finalize(gpu_traj.a_itp.z)
        finalize(gpu_traj.a_itp.c1)
        finalize(gpu_traj.a_itp.c2)
    end

    if mode == Val(:split)
        E_far = permutedims(Array(E1_buf), (4, 3, 1, 2))
        B_far = permutedims(Array(B1_buf), (4, 3, 1, 2))
        E_near = permutedims(Array(E2_buf), (4, 3, 1, 2))
        B_near = permutedims(Array(B2_buf), (4, 3, 1, 2))
        E = E_far .+ E_near
        B = B_far .+ B_near
        return (; E, B, E_far, B_far)
    else
        E = permutedims(Array(E1_buf), (4, 3, 1, 2))
        B = permutedims(Array(B1_buf), (4, 3, 1, 2))
        return (; E, B)
    end
end
