using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen
using DataInterpolations
using StaticArrays
using OrdinaryDiffEqVerner
using LinearAlgebra
using Test

# Exact worldline of an electron born at rest in a plane wave along +z with a Gaussian envelope
# (c = 1): light-front phase φ = τ, u⊥ = a(φ), u⁰ − u³ = 1, w = |a|²/2 ⇒ u = (1 + w, aₓ, a_y, w).
# a(φ) = a₀ g(φ − φc)(cos φ, δ sin φ), δ = 0 linear, 1 circular. 𝔞 = du/dτ in closed form; x = ∫u dτ
# to 1e-13 by Vern9, so x and u agree for the retarded-time mapping of accumulate_field.
function pulse_traj(; a₀, δ = 0.0, T = 12.0, φc = 0.0, span = 4T, N = 6001, K = 1.0)
    g(φ) = exp(-((φ - φc) / T)^2)
    a(φ) = a₀ * g(φ) * SVector(cos(φ), δ * sin(φ))
    a′(φ) = a₀ * g(φ) * SVector(-2(φ - φc) / T^2 * cos(φ) - sin(φ), δ * (-2(φ - φc) / T^2 * sin(φ) + cos(φ)))
    u(φ) = (v = a(φ); w = (v ⋅ v) / 2; SVector(1 + w, v[1], v[2], w))
    τs = collect(range(φc - span, φc + span, length = N))
    prob = ODEProblem((x, _, τ) -> u(τ), SVector(0.0, 0.0, 0.0, 0.0), (first(τs), last(τs)))
    xs = solve(prob, Vern9(); reltol = 1e-13, abstol = 1e-13, saveat = τs).u
    states = [SVector{8}(x..., u(τ)...) for (x, τ) in zip(xs, τs)]
    accs = [(v = a(τ); d = a′(τ); dw = v ⋅ d; SVector(dw, d[1], d[2], dw)) for τ in τs]
    itp = CubicSpline(states, τs; extrapolation = ExtrapolationType.Extension)
    a_itp = CubicSpline(accs, τs; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4), SVector{4, Int}(5, 6, 7, 8), K)
end

const ε₀ = 1 / (4π)   # the atomic-unit value; with c = K = 1 the prefactor ε₀c²K² is 1/(4π)

@testset "angular_energy" begin
    @testset "matches R² ε₀ c ∫|E_far|² dt from accumulate_field" begin
        traj = pulse_traj(; a₀ = 0.5)
        R = 1.0e4
        θ = [-0.5, 0.0, 0.5]
        # screen pixels at the far-field directions (x, y) = R (tan θx, tan θy); a single window covers
        # every pixel's arrival (R/cos θ offsets) at ~16 samples per period of the 4th harmonic
        x⁰ = range(first(traj.itp.t) + R - 5, last(traj.itp.t) + R / cos(0.5)^2 + 20; step = 0.08)
        screen = ObserverScreen(R .* tan.(θ), R .* tan.(θ), R, x⁰; c = 1.0)
        fld = accumulate_field([traj], screen, Vern9(); reltol = 1e-12, abstol = 1e-12)
        δt = step(x⁰)
        W_ref = [sum(abs2, fld.E_far[:, :, i, j]) * δt * ε₀ * (R^2 * (tan(θ[i])^2 + tan(θ[j])^2 + 1))
            for i in 1:3, j in 1:3]
        W = angular_energy([traj], θ, θ; c = 1.0, ε₀, oversample = 4)
        @test all(>(0), W_ref)
        @test maximum(abs.(W ./ W_ref .- 1)) < 3e-4   # measured 9.4e-5, ∝ 1/R (9.4e-4 at R = 1e3, 9.4e-6 at 1e5)
        # residual: finite R (the worldline spans ~1 in x/z against R = 1e4 ⇒ ~1e-4 in R² and n̂)
        # plus observer-sample quadrature; see the report in the PR description
    end

    @testset "full sphere integrates to the Liénard/Larmor energy" begin
        traj = pulse_traj(; a₀ = 0.3, N = 4001)
        nμ, nφ = 400, 200
        μs = [-1 + (2i - 1) / nμ for i in 1:nμ]
        φs = [2π * (j - 0.5) / nφ for j in 1:nφ]
        dirs = [SVector(sqrt(1 - μ^2) * cos(φ), sqrt(1 - μ^2) * sin(φ), μ) for μ in μs, φ in φs]
        W = angular_energy([traj], dirs; c = 1.0, ε₀)
        W_sphere = sum(W) * (2 / nμ) * (2π / nφ)
        # Larmor/Liénard: dW/dt′ = (q²/(6πε₀c³)) (−𝔞·𝔞), q = 4πε₀cK ⇒ (8π/3) ε₀ K² (−𝔞·𝔞) with c = 1
        τs = traj.itp.t
        integrand = [(a = traj.a_itp(τ); u = traj.itp(τ)[5]; -(a[1]^2 - a[2]^2 - a[3]^2 - a[4]^2) * u) for τ in τs]
        W_larmor = (8π / 3) * ε₀ * sum((integrand[1:end-1] .+ integrand[2:end]) ./ 2 .* diff(τs))
        @test W_sphere ≈ W_larmor rtol = 1e-4
    end

    @testset "weak linear field ⇒ dipole pattern ∝ 1 − (n̂·x̂)²" begin
        traj = pulse_traj(; a₀ = 1e-3)
        θ = collect(range(-1.0, 1.0, 9))
        dirs = far_field_directions(θ, θ)
        W = angular_energy([traj], dirs; c = 1.0, ε₀)
        dip = [1 - n[1]^2 for n in dirs]
        @test maximum(abs.(W ./ W[5, 5] .- dip ./ dip[5, 5])) < 1e-3   # drift ~a₀² and Doppler ~a₀²
    end

    @testset "circular polarization ⇒ azimuthally symmetric about the axis" begin
        traj = pulse_traj(; a₀ = 0.5, δ = 1.0, T = 40.0, N = 16001)   # long envelope: rotation ≡ a CEP shift, ∝ e^(−T²/4k)
        for θp in (0.2, 0.7, 1.3)
            dirs = [SVector(sin(θp) * cos(φ), sin(θp) * sin(φ), cos(θp)) for φ in range(0, 2π, 9)[1:8]]
            W = angular_energy([traj], dirs; c = 1.0, ε₀)
            @test maximum(W) / minimum(W) - 1 < 1e-6
        end
    end

    @testset "incoherent: the ensemble is the sum of its electrons" begin
        trajs = [pulse_traj(; a₀ = 0.4, φc = φc) for φc in (0.0, 1.3, -2.1)]
        θ = collect(range(-0.6, 0.6, 7))
        W = angular_energy(trajs, θ, θ; c = 1.0, ε₀, oversample = 2)
        W1 = sum(angular_energy([t], θ, θ; c = 1.0, ε₀, oversample = 2) for t in trajs)
        @test W ≈ W1 rtol = 1e-12
        @test angular_energy(trajs, θ, θ; c = 1.0, ε₀, zsign = -1) ≉ W   # the other side differs (drift along +z)
    end

    @testset "oversampling converges; τspan restricts" begin
        traj = pulse_traj(; a₀ = 1.0, N = 1501)
        θ = collect(range(-0.4, 0.4, 5))
        W1, W4, W8 = (angular_energy([traj], θ, θ; c = 1.0, ε₀, oversample = o) for o in (1, 4, 8))
        @test maximum(abs.(W8 ./ W4 .- 1)) < maximum(abs.(W8 ./ W1 .- 1))
        @test maximum(abs.(W8 ./ W4 .- 1)) < 1e-6
        half = angular_energy([traj], θ, θ; c = 1.0, ε₀, oversample = 4, τspan = (-Inf, 0.0))
        @test all(0 .< half .< W4)
    end

    @testset "one_over_e_halfwidths" begin
        θx = collect(range(-10, 10, 401))
        θy = collect(range(-10, 10, 401))
        sx, sy = 2.0, 3.0
        W = [exp(-(a^2 / (2sx^2) + b^2 / (2sy^2))) for a in θx, b in θy]
        w = one_over_e_halfwidths(W, θx, θy)
        @test w.x ≈ √2 * sx rtol = 1e-3
        @test w.y ≈ √2 * sy rtol = 1e-3
        @test w.r_eq ≈ √2 * sqrt(sx * sy) rtol = 2e-2
        @test isnan(one_over_e_halfwidths(W[190:212, :], θx[190:212], θy).x)   # level outside the grid
    end
end
