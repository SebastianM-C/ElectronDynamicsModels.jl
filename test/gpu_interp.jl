using ElectronDynamicsModels
using ElectronDynamicsModels: GPUCubicSpline, _searchsorted_left,
    advanced_time, retarded_time_rhs
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

        r_obs = SVector{3}(0.0, 5.0, 10.0)
        τ_test = 3.0
        t_obs = 15.0

        val_cpu = retarded_time_rhs(τ_test, (traj_cpu, r_obs), t_obs)
        val_gpu = retarded_time_rhs(τ_test, (traj_gpu, r_obs), t_obs)

        @test val_cpu ≈ val_gpu rtol = 1e-10
    end
end
