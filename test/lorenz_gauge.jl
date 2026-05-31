using ElectronDynamicsModels
using ElectronDynamicsModels: ObserverScreen, TrajectoryInterpolant
using DataInterpolations
using StaticArrays
using LinearAlgebra
using Statistics
using KernelAbstractions: CPU
using Test

# ── Lorenz-gauge self-consistency: ∂_μ A^μ = 0 ──────────────────────────────
# The Liénard-Wiechert 4-potential satisfies the Lorenz condition identically,
# so checking it on the *computed* field is a reference-free, end-to-end test of
# the whole pipeline (retarded-time solve + observer-time binning + the LW
# accumulation in GPUKernelRK4). It holds for ANY worldline — and Aᵘ = K uᵘ/(xr·u)
# is reparameterization-invariant, so the analytic worldline below need not be
# proper-time-normalized. Uniform circular motion keeps both Aˣ and Aʸ active.
function circular_charge(; c = 1.0, Ω = 2.0, A = 0.1, g = 1.0, nper = 30, knots_per = 60, K = 1.0)
    T = 2π / Ω
    ts = collect(range(0.0, nper * T, length = round(Int, nper * knots_per)))
    us = [
        SVector{8}(
                g * t, A * cos(Ω * t), A * sin(Ω * t), 0.0,        # xμ
                g, -A * Ω * sin(Ω * t), A * Ω * cos(Ω * t), 0.0,   # uμ = dxμ/dt
            ) for t in ts
    ]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, SVector{4, Int}(1, 2, 3, 4), SVector{4, Int}(5, 6, 7, 8), K)
end

_anorm(A, t, i, j) = sqrt(A[t, 1, i, j]^2 + A[t, 2, i, j]^2 + A[t, 3, i, j]^2 + A[t, 4, i, j]^2)

@testset "Lorenz gauge ∂_μ A^μ = 0" begin
    c = 1.0
    Ω = 2.0
    k = Ω / c                      # wavenumber for the field-derivative scale k‖A‖
    traj = circular_charge(; c, Ω)
    τf = last(traj.itp.t)

    # Off-axis box on three screens at z and z±δz; centered 2nd-order FD in all of
    # (x⁰, x, y, z). The time step δx⁰ = c·δt sets the dominant truncation ~(Ω·δt)²/6.
    Z = 40.0
    xc, yc = 8.0, 6.0
    Δ = 0.2
    δz = 0.1
    Npix = 9
    SPP = 16
    n_substeps = 4
    half = (Npix ÷ 2) * Δ
    x_grid = range(xc - half, xc + half, length = Npix)
    y_grid = range(yc - half, yc + half, length = Npix)
    δt = (2π / Ω) / SPP
    δx⁰ = c * δt
    x⁰_start = sqrt(Z^2 + (xc + half)^2 + (yc + half)^2) - 2half
    N_samples = round(Int, τf / δx⁰) + 20
    x⁰ = range(start = x⁰_start, step = δx⁰, length = N_samples)

    s0 = ObserverScreen(x_grid, y_grid, Z, x⁰; c)
    sm = ObserverScreen(x_grid, y_grid, Z - δz, x⁰; c)
    sp = ObserverScreen(x_grid, y_grid, Z + δz, x⁰; c)
    A0 = accumulate_potential([traj], s0, GPUKernelRK4(), CPU(); n_substeps)
    Am = accumulate_potential([traj], sm, GPUKernelRK4(), CPU(); n_substeps)
    Ap = accumulate_potential([traj], sp, GPUKernelRK4(), CPU(); n_substeps)

    # Pointwise ∂_μ A^μ on samples whose full time-FD stencil sits inside the active
    # arrival window (so the centered difference doesn't straddle the zero margin at
    # a pixel's first/last arrival slot).
    maxA = maximum(abs, A0)
    thr = 0.3 * maxA
    rel = Float64[]          # |∂_μ A^μ| / (k‖A‖) — relative Lorenz residual
    term_scale = Float64[]   # max|individual term| / (k‖A‖) — depth of the cancellation
    for j in 2:(Npix - 1), i in 2:(Npix - 1), t in 2:(N_samples - 1)
        (
            _anorm(A0, t - 1, i, j) < thr || _anorm(A0, t, i, j) < thr ||
                _anorm(A0, t + 1, i, j) < thr
        ) && continue
        an = _anorm(A0, t, i, j)
        ∂t = (A0[t + 1, 1, i, j] - A0[t - 1, 1, i, j]) / (2δx⁰)    # ∂A⁰/∂x⁰
        ∂x = (A0[t, 2, i + 1, j] - A0[t, 2, i - 1, j]) / (2Δ)      # ∂Aˣ/∂x
        ∂y = (A0[t, 3, i, j + 1] - A0[t, 3, i, j - 1]) / (2Δ)      # ∂Aʸ/∂y
        ∂z = (Ap[t, 4, i, j] - Am[t, 4, i, j]) / (2δz)             # ∂Aᶻ/∂z
        push!(rel, abs(∂t + ∂x + ∂y + ∂z) / (k * an))
        push!(term_scale, maximum(abs, (∂t, ∂x, ∂y, ∂z)) / (k * an))
    end

    @test length(rel) > 1000
    @test median(rel) < 1.0e-3
    @test maximum(rel) < 5.0e-3
    @test median(term_scale) > 0.01
end
