# `coef_reuse = Val(true)` (spline-interval coefficients held in registers across a slot's
# evaluations) reproduces the default path: the same arithmetic on the same operands. On the GPU
# backends that is bit-identical (verified by cube hashes in the A/B protocol: both backends fuse
# every `muladd`). On the CPU backend the two `Val` specializations are separately inlined loops
# and Julia's `muladd` may fuse or not depending on the surrounding code, so results can differ
# at the last bit there (2e-16 absolute here) — hence a tight tolerance below, and an exact `===`
# check on the evaluation primitives, which compile once.
using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen, GPUCubicSpline,
    IntervalCoefs, _fetch_interval, _eval_poly, _eval_at, _in_interval, _searchsorted_left
using DataInterpolations
using StaticArrays
using KernelAbstractions: CPU
using Test

bits_equal(a, b) = size(a) == size(b) && all(x === y for (x, y) in zip(a, b))
# CPU-backend agreement (see the header): a last-bit τ difference propagates through the Newton
# correction to a few 1e-16 on values of order 1e-2, i.e. ~1e-14 of the array's scale; 1e-12 is
# 30× above the observed maximum and 1e5× below any physical difference.
lastbit_equal(a, b) = size(a) == size(b) && maximum(abs, a .- b) ≤ 1e-12 * maximum(abs, a)

function _reuse_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.3, τspan = (0.0, 20.0), N = 3000, K = 1.0)
    ts = collect(range(τspan[1], τspan[2], length = N))
    us = [SVector{8}(g * τ, A * sin(Ω * τ), 0.0, vz * τ, g, A * Ω * cos(Ω * τ), 0.0, vz) for τ in ts]
    as = [SVector{4}(0.0, -A * Ω^2 * sin(Ω * τ), 0.0, 0.0) for τ in ts]
    E = ExtrapolationType.Extension
    return TrajectoryInterpolant(CubicSpline(us, ts; extrapolation = E), CubicSpline(as, ts; extrapolation = E),
        SVector{4, Int}(1, 2, 3, 4), SVector{4, Int}(5, 6, 7, 8), K)
end

@testset "coefficient reuse is bit-identical" begin
    @testset "_fetch_interval / _eval_poly == _eval_at" begin
        traj = _reuse_traj()
        gt = ElectronDynamicsModels.to_gpu(traj; with_acceleration = true)
        for sp in (gt.itp, gt.a_itp), τ in range(-0.5, 20.5; length = 400)
            idx = _searchsorted_left(sp.t, τ)
            c = _fetch_interval(sp, idx)
            @test c isa IntervalCoefs
            @test _eval_poly(c, τ) === _eval_at(sp, τ, idx) === sp(τ)
            if 1 < idx < length(sp.t) - 1   # membership matches the search away from the clamped ends
                @test _in_interval(c, τ)
            end
        end
    end

    traj = _reuse_traj()
    trajs = [traj]
    τi, τf = first(traj.itp.t), last(traj.itp.t)
    z = 50.0; half = 8.0
    x_grid = LinRange(-half, half, 7); y_grid = LinRange(-half, half, 7)
    x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 160)
    screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

    @testset "potential, both kernels" begin
        for (alg, kw) in ((GPUKernelNewton(), (; n_iters = 2)), (GPUKernelNewton(), (; n_iters = 1)),
                          (GPUKernelRK4(), (; n_substeps = 1)), (GPUKernelRK4(), (; n_substeps = 3)))
            A0 = accumulate_potential(trajs, screen, alg, CPU(); kw..., coef_reuse = Val(false))
            A1 = accumulate_potential(trajs, screen, alg, CPU(); kw..., coef_reuse = Val(true))
            @test lastbit_equal(A0, A1)
            @test any(!iszero, A1)
        end
    end

    @testset "field, both kernels, both modes" begin
        for (alg, kw) in ((GPUKernelNewton(), (; n_iters = 2)), (GPUKernelRK4(), (; n_substeps = 2))),
            mode in (Val(:split), Val(:total))
            F0 = accumulate_field(trajs, screen, alg, CPU(); kw..., mode, coef_reuse = Val(false))
            F1 = accumulate_field(trajs, screen, alg, CPU(); kw..., mode, coef_reuse = Val(true))
            for k in keys(F0)
                @test lastbit_equal(F0[k], F1[k])
            end
            @test any(!iszero, F1.E)
        end
    end
end
