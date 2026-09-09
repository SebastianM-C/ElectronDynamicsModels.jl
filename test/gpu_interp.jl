using ElectronDynamicsModels
using ElectronDynamicsModels: GPUCubicSpline, _searchsorted_left, _eval_at,
    advanced_time, retarded_time_rhs
using Random
using DataInterpolations
using StaticArrays
using Test
using LinearAlgebra
using SciMLBase
using OrdinaryDiffEqTsit5

@testset "GPU Interpolation" begin
    @testset "_searchsorted_left" begin
        t = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Interior points
        @test _searchsorted_left(t, 2.5) == 2
        @test _searchsorted_left(t, 1.5) == 1
        @test _searchsorted_left(t, 4.9) == 4

        # Exact knot hits
        @test _searchsorted_left(t, 1.0) == 1
        @test _searchsorted_left(t, 3.0) == 3
        @test _searchsorted_left(t, 5.0) == 4  # clamped to N-1

        # Edge: just past first knot
        @test _searchsorted_left(t, 1.001) == 1
    end

    @testset "warm-started _searchsorted_left == plain search" begin
        rng = MersenneTwister(20260908)
        for _ in 1:200
            N = rand(rng, 2:60)
            # sorted knots, uniform or not, sometimes with duplicates
            t = sort(rand(rng, Bool) ? rand(rng, N) .* 10 : round.(rand(rng, N) .* 10; digits = 1))
            for _ in 1:200
                x = rand(rng, (-1.0, 11.0, rand(rng) * 12 - 1, t[rand(rng, 1:N)], t[rand(rng, 1:N)] + eps()))
                g = rand(rng, -2:(N + 2))   # out-of-range guesses are clamped
                @test _searchsorted_left(t, x, g) == _searchsorted_left(t, x)
            end
        end
        # a monotone walk, the kernels' access pattern
        t = collect(range(0.0, 1.0; length = 1001))
        idx = 1
        for x in range(-0.1, 1.1; length = 5000)
            idx = _searchsorted_left(t, x, idx)
            @test idx == _searchsorted_left(t, x)
        end
        # spline evaluation through a guess is bit-identical to the cold evaluation
        ts = collect(range(0.0, 10.0; length = 50))
        us = [SVector{8}(sin(τ), cos(τ), τ^2 / 100, τ, cos(2τ), sin(2τ), exp(-τ / 10), 1.0 + τ / 10) for τ in ts]
        sp = GPUCubicSpline(CubicSpline(us, ts; extrapolation = DataInterpolations.ExtrapolationType.Extension))
        for τ in range(-0.5, 10.5; length = 300), g in (1, 7, 25, 49, 60)
            v, i = sp(τ, g)
            @test i == _searchsorted_left(sp.t, τ)
            @test v === sp(τ)
            @test _eval_at(sp, τ, i) === sp(τ)
        end
    end

    @testset "to_gpu rejects an acceleration spline on other knots" begin
        ts = collect(range(0.0, 10.0; length = 40))
        us = [SVector{8}(sin(t), cos(t), t, 0.0, -sin(t), -cos(t), 1.0, 0.0) for t in ts]
        as = [SVector{4}(cos(t), -sin(t), 0.0, 0.0) for t in ts]
        E = DataInterpolations.ExtrapolationType.Extension
        itp = CubicSpline(us, ts; extrapolation = E)
        same = CubicSpline(as, ts; extrapolation = E)
        other = CubicSpline(as[1:2:end], ts[1:2:end]; extrapolation = E)
        x_idxs = SVector{4, Int}(1, 2, 3, 4)
        u_idxs = SVector{4, Int}(5, 6, 7, 8)
        gt = ElectronDynamicsModels.to_gpu(TrajectoryInterpolant(itp, same, x_idxs, u_idxs, 1.0); with_acceleration = true)
        @test gt isa TrajectoryInterpolant
        @test_throws ArgumentError ElectronDynamicsModels.to_gpu(TrajectoryInterpolant(itp, other, x_idxs, u_idxs, 1.0); with_acceleration = true)
    end

    @testset "GPUCubicSpline accuracy" begin
        # Create test data: interpolate a known vector-valued function
        N = 50
        ts = range(0.0, 10.0, length = N)
        # 8 components like our trajectory (mix of sin, cos, polynomials)
        us = [SVector{8}(
            sin(t), cos(t), t^2 / 100, t,
            cos(2t), sin(2t), exp(-t / 10), 1.0 + t / 10
        ) for t in ts]

        itp_ref = CubicSpline(us, collect(ts);
            extrapolation = DataInterpolations.ExtrapolationType.Extension)
        gpu_spline = GPUCubicSpline(itp_ref)

        # Test at many interior points
        test_ts = range(0.1, 9.9, length = 200)
        max_err = 0.0
        for τ in test_ts
            ref = itp_ref(τ)
            gpu = gpu_spline(τ)
            err = norm(ref - gpu)
            max_err = max(max_err, err)
        end
        @test max_err < 1e-12

        # Test at knot points
        for τ in ts[2:end-1]
            ref = itp_ref(τ)
            gpu = gpu_spline(τ)
            @test norm(ref - gpu) < 1e-12
        end
    end

    @testset "GPUCubicSpline with TrajectoryInterpolant indexing" begin
        # Simulate what TrajectoryInterpolant does: interpolate, then index
        N = 30
        ts = range(0.0, 5.0, length = N)
        us = [SVector{8}(sin(t), cos(t), t, 0.0, -sin(t), -cos(t), 1.0, 0.0) for t in ts]

        itp_ref = CubicSpline(us, collect(ts);
            extrapolation = DataInterpolations.ExtrapolationType.Extension)
        gpu_spline = GPUCubicSpline(itp_ref)

        x_idxs = SVector{4, Int}(1, 2, 3, 4)
        u_idxs = SVector{4, Int}(5, 6, 7, 8)

        τ = 2.7
        v_ref = itp_ref(τ)
        v_gpu = gpu_spline(τ)

        # Check that indexing with SVector indices works
        @test v_ref[x_idxs] ≈ v_gpu[x_idxs]
        @test v_ref[u_idxs] ≈ v_gpu[u_idxs]
    end

    @testset "GPUCubicSpline in retarded_time_rhs" begin
        # Verify GPUCubicSpline gives same RHS values as CubicSpline
        N = 40
        ts = range(0.0, 10.0, length = N)
        v = 0.1
        γ = 1.0 / sqrt(1 - v^2)
        us = [SVector{8}(γ * t, v * γ * t, 0.0, 0.0, γ, v * γ, 0.0, 0.0) for t in ts]

        itp = CubicSpline(us, collect(ts);
            extrapolation = DataInterpolations.ExtrapolationType.Extension)
        gpu_spline = GPUCubicSpline(itp)

        x_idxs = SVector{4, Int}(1, 2, 3, 4)
        u_idxs = SVector{4, Int}(5, 6, 7, 8)
        K = 1.0

        traj_cpu = TrajectoryInterpolant(itp, x_idxs, u_idxs, K)
        traj_gpu = TrajectoryInterpolant(gpu_spline, x_idxs, u_idxs, K)
        @test canonical_state_order(traj_cpu)
        # the GPU kernels index the state with literals: a non-canonical layout is rejected up front
        @test_throws ArgumentError ElectronDynamicsModels.to_gpu(TrajectoryInterpolant(itp, u_idxs, x_idxs, K))

        r_obs = SVector{3}(0.0, 5.0, 10.0)
        τ_test = 3.0
        t_obs = 15.0

        val_cpu = retarded_time_rhs(τ_test, (traj_cpu, r_obs), t_obs)
        val_gpu = retarded_time_rhs(τ_test, (traj_gpu, r_obs), t_obs)

        @test val_cpu ≈ val_gpu rtol = 1e-10
    end
end
