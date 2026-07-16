# V2 — the actual GPUKernelNewton kernel on the CPU backend vs the adaptive
# Vern9 CPU reference, in both analytic regimes of test/gpu_radiation.jl,
# alongside GPUKernelRK4 for context. A-level (accumulated potential) errors.
#
# Run from the worktree root: julia --project=scripts scripts/v2_newton_kernel_check.jl

using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen
using DataInterpolations
using StaticArrays
using KernelAbstractions: CPU
using OrdinaryDiffEqVerner
using LinearAlgebra
using Printf

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

rel_l2(a, b) = norm(a .- b) / norm(b)

function run_case(name, traj, screen)
    trajs = [traj]
    A_ref = accumulate_potential(trajs, screen, Vern9())
    println("\n═══ $name ═══")
    for n in (1, 8)
        e = rel_l2(accumulate_potential(trajs, screen, GPUKernelRK4(), CPU(); n_substeps = n), A_ref)
        @printf("  RK4    n_substeps=%d : rel_l2 = %.2e\n", n, e)
    end
    for n in (1, 2, 3, 4)
        e = rel_l2(accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = n), A_ref)
        @printf("  Newton n_iters=%d   : rel_l2 = %.2e\n", n, e)
    end
end

# regime 1: transverse
traj_t = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
    τspan = (0.0, 20.0), N = 6000)
z = 50.0
grid = LinRange(-8.0, 8.0, 9)
screen_t = ObserverScreen(grid, grid, z,
    LinRange(0.0 + (z - 16.0), 1.2 * 20.0 + (z + 16.0), 240); c = 1.0)
run_case("transverse (vz = 0)", traj_t, screen_t)

# regime 2: forward Doppler
g, vz = 1.05, 0.95
traj_d = analytic_traj(; g, A = 0.15, Ω = 2.0, vz,
    τspan = (0.0, 20.0), N = 8000)
z = 60.0
grid_d = LinRange(-6.0, 6.0, 7)
screen_d = ObserverScreen(grid_d, grid_d, z,
    LinRange(g * 0.0 + (z - vz * 20.0 - 6.0), g * 20.0 + (z + 6.0), 200); c = 1.0)
run_case("forward Doppler (vz = 0.95)", traj_d, screen_d)
