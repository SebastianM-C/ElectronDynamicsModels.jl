# V1 — host-side check of the Newton light-cone solve (kernel_newton.jl math)
# against a tight-tolerance ODE solve of the retarded-time equation.
#
# Replicates the per-pixel loop of `_gpu_newton_one_electron!` on the host
# (same `_lightcone_eval`, same window/predictor/correction logic), records the
# per-slot τ_r, and compares against Vern9 at reltol=abstol=1e-12 on
# `retarded_time_rhs`. Uses the analytic worldline from test/gpu_radiation.jl —
# no MTK solve, runs in seconds, no GPU.
#
# Run from the worktree root: julia --project=scripts scripts/v1_newton_root_check.jl

using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen, to_gpu,
    _lightcone_eval, retarded_time_rhs, advanced_time
using DataInterpolations
using StaticArrays
using OrdinaryDiffEqVerner
using SciMLBase
using LinearAlgebra
using Printf

# ── analytic worldline (copied from test/gpu_radiation.jl) ──────────────────
function analytic_traj(; g, A, Ω, vz, τspan, N, K = 1.0)
    @assert g > sqrt(vz^2 + (A * Ω)^2) "worldline must stay timelike"
    ts = collect(range(τspan[1], τspan[2], length = N))
    us = [SVector{8}(
        g * τ, A * sin(Ω * τ), 0.0, vz * τ,
        g, A * Ω * cos(Ω * τ), 0.0, vz,
    ) for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    as = [SVector{4}(0.0, -A * Ω^2 * sin(Ω * τ), 0.0, 0.0) for τ in ts]
    a_itp = CubicSpline(as, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4),
        SVector{4, Int}(5, 6, 7, 8), K)
end

# ── host replica of the kernel's per-pixel Newton march ─────────────────────
# Same math and control flow as _gpu_newton_one_electron!, but records τ and
# the final residual f at every slot instead of accumulating the potential.
function newton_march(gpu_traj, r_obs, x⁰_samples, τi, τf; n_iters)
    x⁰_first = first(x⁰_samples)
    δx⁰ = step(x⁰_samples)
    N_samples = length(x⁰_samples)

    v_i = gpu_traj.itp(τi)
    d_i = SVector{3}(r_obs[j] - v_i[gpu_traj.x_idxs[1 + j]] for j in 1:3)
    R_i = norm(d_i)
    x⁰_i_px = v_i[gpu_traj.x_idxs[1]] + R_i
    v_f = gpu_traj.itp(τf)
    x⁰_f_px = v_f[gpu_traj.x_idxs[1]] +
        norm(SVector{3}(r_obs[j] - v_f[gpu_traj.x_idxs[1 + j]] for j in 1:3))

    inv_δ = inv(δx⁰)
    k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
    k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))

    τs = fill(NaN, N_samples)
    fs = fill(NaN, N_samples)
    k_start > k_end && return τs, fs, k_start, k_end

    u_i = SVector{4}(v_i[gpu_traj.u_idxs[j]] for j in 1:4)
    rhs = R_i / (R_i * u_i[1] - d_i ⋅ u_i[SA[2, 3, 4]])
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
        τs[k] = τ
        fs[k] = f
        x⁰_target += δx⁰
        Δ = δx⁰
    end
    return τs, fs, k_start, k_end
end

# ── tight-tolerance ODE reference, scattered onto the same slots ────────────
function ode_reference(traj, r_obs, x⁰_samples, τi, τf)
    x⁰_i = advanced_time(traj, τi, r_obs)
    x⁰_f = advanced_time(traj, τf, r_obs)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs, τi, (x⁰_i, x⁰_f), (traj, r_obs))
    sol = solve(prob, Vern9(); reltol = 1.0e-12, abstol = 1.0e-12,
        saveat = x⁰_samples, save_start = false, save_end = false)
    τ_ref = fill(NaN, length(x⁰_samples))
    δ = step(x⁰_samples)
    for (t, τ) in zip(sol.t, sol.u)
        idx = round(Int, (t - first(x⁰_samples)) / δ) + 1
        1 <= idx <= length(τ_ref) && (τ_ref[idx] = τ)
    end
    return τ_ref
end

function run_case(name, traj, screen; pixels)
    τi, τf = first(traj.itp.t), last(traj.itp.t)
    gpu_traj = to_gpu(traj)   # host-side GPUCubicSpline
    println("\n═══ $name ═══  (τ span = $(τf - τi), δx⁰ = $(round(step(screen.x⁰_samples), sigdigits=3)))")
    for (ix, iy) in pixels
        r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
        τ_ref = ode_reference(traj, r_obs, screen.x⁰_samples, τi, τf)
        @printf("  pixel (%2d,%2d):  ", ix, iy)
        for n_iters in 0:3
            τs, fs, k_lo, k_hi = newton_march(gpu_traj, r_obs, screen.x⁰_samples, τi, τf; n_iters)
            both = findall(k -> !isnan(τs[k]) && !isnan(τ_ref[k]), 1:length(τ_ref))
            # drop the window-edge slots where the strict-interior conventions differ
            interior = filter(k -> k_lo + 1 <= k <= k_hi - 1, both)
            Δτ = maximum(abs.(τs[interior] .- τ_ref[interior])) / (τf - τi)
            fmax = maximum(abs.(fs[interior]))
            @printf("it=%d: Δτ=%.1e |f|=%.1e   ", n_iters, Δτ, fmax)
        end
        println()
    end
end

# ── regime 1: transverse (test/gpu_radiation.jl bulk-pattern setup) ─────────
traj_t = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
    τspan = (0.0, 20.0), N = 6000)
τi, τf = 0.0, 20.0
z = 50.0
grid = LinRange(-8.0, 8.0, 9)
screen_t = ObserverScreen(grid, grid, z,
    LinRange(1.2τi + (z - 16.0), 1.2τf + (z + 16.0), 240); c = 1.0)
run_case("transverse (vz = 0)", traj_t, screen_t;
    pixels = [(1, 1), (5, 5), (9, 1), (3, 7), (9, 9)])

# ── regime 2: forward Doppler (vz ≈ c — Newton stress test) ─────────────────
g, vz = 1.05, 0.95
traj_d = analytic_traj(; g, A = 0.15, Ω = 2.0, vz,
    τspan = (0.0, 20.0), N = 8000)
z = 60.0
grid_d = LinRange(-6.0, 6.0, 7)
screen_d = ObserverScreen(grid_d, grid_d, z,
    LinRange(g * τi + (z - vz * 20.0 - 6.0), g * 20.0 + (z + 6.0), 200); c = 1.0)
run_case("forward Doppler (vz = 0.95)", traj_d, screen_d;
    pixels = [(1, 1), (4, 4), (7, 1), (2, 6), (7, 7)])

println("\nΔτ is relative to the τ span; |f| is the raw light-cone residual (same units as x⁰).")
println("it=0 column = Euler predictor alone (no correction) — each added iteration")
println("should shrink the error quadratically until the spline/cancellation floor.")
