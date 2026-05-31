using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen
using DataInterpolations
using StaticArrays
using KernelAbstractions: CPU
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqVerner
using LinearAlgebra
using Test

# ── Analytic worldline (no ODE solve) ───────────────────────────────────────
# Proper-time-parameterized charge: constant-rate time + z drift `vz` toward the
# screen, plus a transverse sinusoidal wiggle along x.  Future-directed timelike
# (u⁰ > |u⃗|) as long as `g > sqrt(vz² + (A·Ω)²)`, so the retarded-time RHS
# 1/(u⁰ - u⃗·n̂) stays positive.  Two regimes are used below:
#   - `vz = 0`  (transverse): integrand ≈ const, clean L2 agreement — a
#     correctness check for the kernel (geometry, slot mapping, spline eval).
#   - `vz ≈ c`  (forward Doppler): u⃗·n̂ ≈ u⁰, so the integrand swings hard and
#     fixed-step RK4 is genuinely sensitive to `n_substeps` — a convergence check.
function analytic_traj(; g, A, Ω, vz, τspan, N, K = 1.0)
    @assert g > sqrt(vz^2 + (A * Ω)^2) "worldline must stay timelike"
    ts = collect(range(τspan[1], τspan[2], length = N))
    us = [SVector{8}(
        g * τ,                # x⁰
        A * sin(Ω * τ),       # x¹
        0.0,                  # x²
        vz * τ,               # x³
        g,                    # u⁰
        A * Ω * cos(Ω * τ),   # u¹
        0.0,                  # u²
        vz,                   # u³
    ) for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, SVector{4, Int}(1, 2, 3, 4),
        SVector{4, Int}(5, 6, 7, 8), K)
end

# Relative Frobenius error — robust to the single-slot boundary ambiguity at the
# edge of each pixel's arrival window (where even Tsit5 vs Vern9 disagree by a
# few %).  A max-abs metric would report that edge slot instead of the bulk fit.
rel_l2(a, b) = norm(a .- b) / norm(b)

@testset "GPU radiation accumulation" begin

    @testset "GPUKernelRK4 matches reference (bulk pattern)" begin
        # Mild transverse regime: integrand nearly constant, so the kernel's
        # geometry / slot-mapping / spline evaluation is what's under test.
        traj = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
            τspan = (0.0, 20.0), N = 6000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)

        z = 50.0
        Nx = Ny = 9
        half = 8.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 240)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        A_gpu = accumulate_potential(trajs, screen, GPUKernelRK4(), CPU(); n_substeps = 8)

        @test size(A_gpu) == size(A_ref)
        @test all(isfinite, A_gpu)
        @test maximum(abs, A_ref) > 0                 # the reference actually radiates
        @test rel_l2(A_gpu, A_ref) < 5.0e-3           # bulk pattern matches the reference

        # The two-phase CPU-solve + AcceleratedKernels accumulation path
        # (src/gpu/accumulate.jl) should track the same reference.
        A_ak = accumulate_potential(trajs, screen, Tsit5(), CPU())
        @test rel_l2(A_ak, A_ref) < 5.0e-3
    end

    @testset "n_substeps drives RK4 convergence (forward Doppler)" begin
        # Relativistic drift toward the screen ⇒ u⃗·n̂ ≈ u⁰ ⇒ the retarded-time
        # integrand swings hard, so n_substeps = 1 is visibly under-resolved.
        traj = analytic_traj(; g = 1.05, A = 0.15, Ω = 2.0, vz = 0.95,
            τspan = (0.0, 20.0), N = 8000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)
        g, vz = 1.05, 0.95

        z = 60.0
        Nx = Ny = 7
        half = 6.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        # Forward emission compresses the arrival window (dx⁰/dτ ≈ u⁰ - u³).
        x⁰ = LinRange(g * τi + (z - vz * τf - half), g * τf + (z + half), 200)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        err(n) = rel_l2(accumulate_potential(trajs, screen, GPUKernelRK4(), CPU(); n_substeps = n), A_ref)

        e1, e8 = err(1), err(8)
        @test e1 > 5.0e-2          # single-step RK4 is genuinely under-resolved here
        @test e8 < 5.0e-3          # enough sub-steps recovers reference agreement
        @test e8 < e1              # adding sub-steps reduces the discrepancy
    end
end
