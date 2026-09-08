# Algorithmic FLOP profile of the field kernels (src/diagnostics/flops.jl) — CPU only.
using ElectronDynamicsModels
using ElectronDynamicsModels: _lightcone_eval, _rt_rhs_kernel, _rk4_step, _window_edge, m_dot, to_gpu,
    _profile_trajectory, _counted
using CountedFloats
using ElectronDynamicsModels.KernelAbstractions: CPU
using StaticArrays
using LinearAlgebra
using Test

const CF64 = Counted{Float64}

traj = _profile_trajectory()
gt = _counted(to_gpu(traj; with_acceleration = true))
r_obs = SVector{3}(CF64(1.0), CF64(-2.0), CF64(50.0))
τ = CF64(3.3)

@testset "helper counts reproduce the hand-derived numbers" begin
    x = SVector{4}(CF64.((1.0, 2.0, 3.0, 4.0)))
    y = SVector{4}(CF64.((0.5, 0.25, 2.0, 1.0)))
    # `@muladd` on m_dot: 1 mul + 3 fma (each fma counts 2 FLOP) — same 7 FLOP as 4 mul + 3 add
    c = @count m_dot(x, y)
    @test c[:mul] == 1 && c[:fma] == 3 && c[:add] == 0 && flops(c) == 7
    v = SVector{3}(CF64.((1.0, 2.0, 3.0)))
    c = @count norm(v)
    @test c[:mul] == 3 && c[:add] == 2 && c[:sqrt] == 1 && flops(c) == 6
    # cubic spline, D = 8, under `@muladd`: per component 7 mul + 3 fma (dt³ recomputed; the three
    # sums of products fuse), plus dt1, dt2 (2 sub) and inv(6h) (1 mul, 1 div) — 108 FLOP as before
    cs = @count gt.itp(τ)
    @test cs[:mul] == 57 && cs[:fma] == 24 && cs[:add] == 2 && cs[:div] == 1 && cs[:sqrt] == 0 && flops(cs) == 108
    ca = @count gt.a_itp(τ)
    @test ca[:mul] == 29 && ca[:fma] == 12 && ca[:add] == 2 && ca[:div] == 1
    # light-cone residual eval = spline + 2 mul / 5 fma / 7 add / 2 div / 1 sqrt (the ρ², R and X·u
    # sums of products fuse under `@muladd`)
    cl = @count _lightcone_eval(τ, gt, r_obs, CF64(1.0))
    @test cl[:mul] == cs[:mul] + 2 && cl[:fma] == cs[:fma] + 5 && cl[:add] == cs[:add] + 7 && cl[:div] == cs[:div] + 2 && cl[:sqrt] == 1
    # retarded-time RHS = spline + 2 mul / 5 fma / 3 add / 2 div / 1 sqrt
    cr = @count _rt_rhs_kernel(τ, gt, r_obs)
    @test cr[:mul] == cs[:mul] + 2 && cr[:fma] == cs[:fma] + 5 && cr[:add] == cs[:add] + 3 && cr[:div] == cs[:div] + 2 && cr[:sqrt] == 1
    # RK4 step = 4 RHS evals + 8 mul + 7 add
    ck = @count _rk4_step(τ, CF64(0.1), gt, r_obs)
    @test flops(ck) == 4 * flops(cr) + 15
    @test cl[:trans] == 0 && cl[:pow] == 0
end

@testset "flop_profile: deterministic, linear in the accuracy knob, mode-independent" begin
    pN1 = flop_profile(GPUKernelNewton(); n_iters = 1)
    pN2 = flop_profile(GPUKernelNewton(); n_iters = 2)
    pN3 = flop_profile(GPUKernelNewton(); n_iters = 3)
    @test pN2 == flop_profile(GPUKernelNewton(); n_iters = 2)
    @test pN2.alg == "GPUKernelNewton" && pN2.mode == :split && pN2.n_name == :n_iters && pN2.n == 2
    # one more Newton correction per slot = one light-cone eval + proposal (mul, add) + midpoint (add, div)
    cl = flops(@count _lightcone_eval(τ, gt, r_obs, CF64(1.0)))
    @test pN2.flop_per_slot - pN1.flop_per_slot == cl + 4
    @test pN3.flop_per_slot - pN2.flop_per_slot == cl + 4
    @test pN2.per_slot.fma > 0 && pN2.per_slot.pow == 0 && pN2.per_slot.trans == 0   # fused under `@muladd`
    @test pN2.per_slot.sqrt == 3 + 0   # one per light-cone eval (predictor + 2 corrections)
    @test pN2.flop_per_slot > 500 && pN2.flop_per_pixel_launch > 0
    @test pN2.per_pixel_launch.sqrt == 2   # the two window edges

    pR1 = flop_profile(GPUKernelRK4(); n_substeps = 1)
    pR2 = flop_profile(GPUKernelRK4(); n_substeps = 2)
    pR4 = flop_profile(GPUKernelRK4(); n_substeps = 4)
    step = flops(@count _rk4_step(τ, CF64(0.1), gt, r_obs))
    @test pR2.flop_per_slot - pR1.flop_per_slot == step
    @test pR4.flop_per_slot - pR2.flop_per_slot == 2 * step
    @test pR1.n_name == :n_substeps && pR1.alg == "GPUKernelRK4"

    # :total sums far + near in-kernel: same FLOPs per slot, half the buffer traffic
    tN = flop_profile(GPUKernelNewton(); mode = :total, n_iters = 2)
    @test tN.flop_per_slot == pN2.flop_per_slot && tN.per_slot == pN2.per_slot
    @test tN.bytes_per_slot == 96 && pN2.bytes_per_slot == 192
    @test tN.arithmetic_intensity == 2 * pN2.arithmetic_intensity
    @test flop_profile(GPUKernelRK4(); mode = Val(:total)).flop_per_slot == pR1.flop_per_slot

    @test_throws ArgumentError flop_profile(GPUKernelNewton(); mode = :bogus)
    @test_throws ArgumentError flop_profile(GPUKernelNewton(); n_iters = 0)
    @test occursin("CountedFloats", pN2.convention)
end

@testset "measured FP64 peak reachable through the package (probe tested in lib/GPUDiagnostics)" begin
    h = gpu_peak_fp64_flops(CPU())
    @test isfinite(h) && h > 1.0e8
end
