# V3 — GPU precision of the Newton light-cone kernel at the a₀=0.1 production
# setup, against the tight-tolerance CPU reference (diag_precision.jl layout
# with GPUKernelNewton rows added).

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using OrdinaryDiffEqNonlinearSolve
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using AcceleratedKernels
using AMDGPU
using Printf

const backend = AMDGPU.ROCBackend()
const c = 137.03599908330932

const ω = 0.057
const τ = 150 / ω
const λ = 2π * c / ω
const w₀ = 75λ
const Rmax = 3.25w₀
const a₀ = 0.1

@named world = Worldline(:τ, :atomic)
@named laser = LaguerreGaussLaser(;
    wavelength = λ, a0 = a₀, beam_waist = w₀, radial_index = 2, azimuthal_index = -2,
    world, temporal_profile = :gaussian, temporal_width = τ, focus_position = 0.0,
    polarization = :circular,
)
@named elec = ClassicalElectron(; laser)
const sys = mtkcompile(elec)

const τi = -8τ
const τf = 8τ
const prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, [sys.x => [τi * c, 0.0, 0.0, 0.0], sys.u => [c, 0.0, 0.0, 0.0]], (τi, τf);
    u0_constructor = SVector{8}, fully_determined = true,
)
const set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

const ϕ = (1 + √5) / 2
radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
function sunflower(n, α)
    pts = Vector{Vector{Float64}}()
    stride = 2π / ϕ^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        push!(pts, [r * cos(k * stride), r * sin(k * stride)])
    end
    return pts
end
function abserr(a₀)
    amp = log10(a₀)
    return 10^(-amp^2 / 27 + 32amp / 27 - 220 / 27)
end

function build_trajs(N)
    xμ = [SVector{4, Float64}(τi * c, r[1], r[2], 0.0) for r in Rmax * sunflower(N, 2)]
    function prob_func(prob, ctx)
        u0_new, p = set_x(prob, SVector{8}(xμ[ctx.sim_id]..., c, 0.0, 0.0, 0.0))
        return remake(prob; u0 = u0_new, p)
    end
    ens = EnsembleProblem(prob; prob_func, safetycopy = false)
    sol = solve(ens, Vern9(), EnsembleThreads(); reltol = 1.0e-12, abstol = abserr(a₀), trajectories = N)
    return trajectory_interpolants(sol)
end

const Z = 2.0e5λ
const δt = 2π / ω / 5
const x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)
build_screen(Nx, Ns) = ObserverScreen(LinRange(-25w₀, 25w₀, Nx), LinRange(-25w₀, 25w₀, Nx), Z, range(start = x⁰_start, step = c * δt, length = Ns); c)

relerr(A, Aref) = norm(A .- Aref) / norm(Aref)

function main()
    trajs = build_trajs(100)
    screen = build_screen(50, 1000)

    A_tight = accumulate_potential(trajs, screen, Tsit5(); reltol = 1.0e-12, abstol = 1.0e-12)
    A_rk4_1 = accumulate_potential(trajs, screen, GPUKernelRK4(), backend; n_substeps = 1)
    A_rk4_8 = accumulate_potential(trajs, screen, GPUKernelRK4(), backend; n_substeps = 8)

    @printf("RK4 n_substeps=1  vs tight-cpu-ref : %.3e\n", relerr(A_rk4_1, A_tight))
    @printf("RK4 n_substeps=8  vs tight-cpu-ref : %.3e\n", relerr(A_rk4_8, A_tight))
    for n in (1, 2, 3)
        A_newton = accumulate_potential(trajs, screen, GPUKernelNewton(), backend; n_iters = n)
        @printf("Newton n_iters=%d  vs tight-cpu-ref : %.3e\n", n, relerr(A_newton, A_tight))
    end
    return nothing
end

main()
