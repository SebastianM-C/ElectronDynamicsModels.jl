using ElectronDynamicsModels
using ElectronDynamicsModels: m_dot   # Minkowski product, for the reference grouped LW tensor
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using StaticArrays
using Test
using SciMLBase
using LinearAlgebra
using FFTW

@testset "Thomson Scattering — Dipole Pattern" begin
    # Non-relativistic Thomson scattering from a single electron in a weak plane wave.
    # For a₀ ≪ 1 (linear polarization along x), the scattered radiation at the
    # fundamental frequency follows the dipole pattern: |A_ω|² ∝ sin²(θ) / R²
    # where θ is the angle from the polarization (x) axis.
    # Higher harmonics should be negligible.

    # Use natural units (c = 1)
    @named world = Worldline(:τ,:natural)

    a₀ = 0.01
    @named laser = PlaneWave(; amplitude = a₀, frequency = 1.0, world)
    @named elec = ClassicalElectron(; laser)
    sys = mtkcompile(elec)

    ω = 1.0
    T = 2π / ω
    c = 1.0

    # Simulate for several periods to get a clean FFT
    n_periods = 20
    τi = -n_periods * T / 2
    τf = n_periods * T / 2
    tspan = (τi, τf)

    x⁰ = [τi * c, 0.0, 0.0, 0.0]
    u⁰ = [c, 0.0, 0.0, 0.0]  # electron at rest

    u0 = [sys.x => x⁰, sys.u => u⁰]
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
    )
    sol = solve(prob, Vern9(), reltol = 1.0e-14, abstol = 1.0e-14)

    @test SciMLBase.successful_retcode(sol)

    # Build trajectory interpolant
    trajs = [TrajectoryInterpolant(sol, sys.x, sys.u)]

    # Screen setup: observe from far field at distance Z along z
    Z = 1.0e4  # far field: Z ≫ λ = 2π
    δt = T / 8  # 8 samples per period (Nyquist up to 4th harmonic)

    Nx, Ny = 15, 15
    L = Z * 0.1  # small screen so all pixels have similar arrival times

    # Cover the full range: earliest signal (center pixel) to latest end (corner pixel)
    x⁰_earliest = c * τi + Z
    x⁰_latest = c * τf + hypot(Z, L * sqrt(2))
    N_samples = ceil(Int, (x⁰_latest - x⁰_earliest) / (c * δt))
    x⁰_samples = range(start = x⁰_earliest, step = c * δt, length = N_samples)

    screen = ObserverScreen(
        LinRange(-L, L, Nx),
        LinRange(-L, L, Ny),
        Z,
        x⁰_samples;
        c,
    )

    A_s = accumulate_potential(trajs, screen, Tsit5())
    A_ω = rfft(A_s, 1)

    # Find fundamental frequency bin
    freqs = rfftfreq(N_samples, 1 / δt)
    idx_f1 = findmin(f -> abs(f - ω / 2π), freqs)[2]

    # Extract |A_ω|² at fundamental frequency (all 3 spatial components)
    intensity = zeros(Nx, Ny)
    A_ω_expected = zeros(Nx, Ny)
    for ix in 1:Nx, iy in 1:Ny
        intensity[ix, iy] = sum(abs2, A_ω[idx_f1, 2:4, ix, iy])
        r_obs = [screen.x_grid[ix], screen.y_grid[iy], Z]
        R = norm(r_obs)
        n̂ = r_obs / norm(r_obs)
        A_ω_expected[ix, iy] = (1 - n̂[1]^2) / R^2
    end

    @testset "Dipole angular pattern" begin
        for ix in 1:Nx, iy in 1:Ny
            expected = A_ω_expected[ix, iy] / maximum(A_ω_expected)
            @test intensity[ix, iy] / maximum(intensity) ≈ expected rtol = 0.05
        end
    end

    @testset "Higher harmonics suppressed" begin
        # For a₀ ≪ 1, the 2nd harmonic should be much weaker than fundamental
        idx_f2 = findmin(f -> abs(f - 2ω / 2π), freqs)[2]
        power_f1 = sum(abs2, A_ω[idx_f1, 2:4, :, :])
        power_f2 = sum(abs2, A_ω[idx_f2, 2:4, :, :])

        @test power_f2 / power_f1 < 1.0e-4
    end
end

@testset "Liénard–Wiechert radiation/velocity split" begin
    # `lienard_wiechert_F_split` regroups eq:lw by power of R into F_vel (1/R²) and
    # F_rad (1/R). Validate (a) it sums to the original grouped closed form, (b) the
    # R-scaling of each piece, and (c) that the split is better conditioned than the
    # cancellation-prone (c² − X𝔞) extraction at small a₀.

    # Independent reference: the ORIGINAL grouped form with the (c² − X𝔞) factor.
    lw_grouped(X, u, 𝔞, K, c) = @SMatrix [
        K * (
                (X[μ] * 𝔞[ν] - X[ν] * 𝔞[μ]) * inv(m_dot(X, u)^2) +
                (X[μ] * u[ν] - X[ν] * u[μ]) * (c^2 - m_dot(X, 𝔞)) * inv(m_dot(X, u)^3)
            ) for μ in 1:4, ν in 1:4
    ]

    cc = 1.0
    K = 0.7
    v = 0.1
    γ = 1 / sqrt(1 - v^2)
    u = SVector{4}(γ * cc, γ * v, 0.0, 0.0)
    R = 100.0
    n̂ = SVector{3}(0.6, 0.0, 0.8)                       # unit line of sight
    X = SVector{4}(R, R * n̂[1], R * n̂[2], R * n̂[3])     # Xᵘ = (R, R n̂)

    @testset "F_vel + F_rad reproduces the grouped form" begin
        𝔞 = SVector{4}(0.0, 0.3, -0.2, 0.1)             # moderate 4-acceleration
        F_vel, F_rad = lienard_wiechert_F_split(X, u, 𝔞, K, cc)
        @test F_vel + F_rad ≈ lw_grouped(X, u, 𝔞, K, cc) rtol = 1.0e-12
        @test lienard_wiechert_F(X, u, 𝔞, K, cc) ≈ lw_grouped(X, u, 𝔞, K, cc) rtol = 1.0e-12
    end

    @testset "F_rad ∝ 1/R and F_vel ∝ 1/R²" begin
        𝔞 = SVector{4}(0.0, 0.3, -0.2, 0.1)
        λ = 8.0                                          # observer λ× farther along n̂
        Fv1, Fr1 = lienard_wiechert_F_split(X, u, 𝔞, K, cc)
        Fv2, Fr2 = lienard_wiechert_F_split(λ .* X, u, 𝔞, K, cc)
        @test Fv2 ≈ Fv1 ./ λ^2 rtol = 1.0e-10
        @test Fr2 ≈ Fr1 ./ λ rtol = 1.0e-10
    end

    @testset "split avoids the (c² − X𝔞) cancellation at small a₀" begin
        # Tiny acceleration ⇒ X𝔞 ≪ c², so forming (c² − X𝔞) in Float64 loses the
        # low bits of the radiation contribution carried by X𝔞. The direct −X𝔞 term
        # in `lienard_wiechert_F_split` never forms that difference.
        ε = 1.0e-10
        𝔞 = SVector{4}(0.0, 0.3, -0.2, 0.1) .* ε
        # BigFloat ground truth for the radiation tensor (clean split formula).
        _, Fr_big = lienard_wiechert_F_split(BigFloat.(X), BigFloat.(u), BigFloat.(𝔞), BigFloat(K), BigFloat(cc))
        Fr_big64 = Float64.(Fr_big)
        # Our Float64 split, vs the naive grouped-minus-velocity extraction.
        Fv64, Fr_split = lienard_wiechert_F_split(X, u, 𝔞, K, cc)
        Fr_naive = lw_grouped(X, u, 𝔞, K, cc) - Fv64
        err_split = maximum(abs, Fr_split - Fr_big64)
        err_naive = maximum(abs, Fr_naive - Fr_big64)
        @test err_split < err_naive                      # better conditioned
        @test err_naive > 1.0e3 * err_split              # by orders of magnitude at this a₀
    end
end
