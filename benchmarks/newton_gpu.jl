# Benchmark: GPUKernelNewton (per-slot light-cone Newton solve) vs GPUKernelRK4
# (retarded-time ODE march) on the potential path.  Physics identical to
# benchmark_experimental_gpu.jl / thomson_scattering.jl (a₀ = 0.1 circular LG).
#
# Accuracy anchor per rung is RK4 n_substeps = 8 (converged; absolute accuracy
# vs the tight CPU reference is established separately by
# diag_newton_precision.jl at the (100, 50) size).  The RK4 ns=1 vs ns=8 pair
# doubles as a sanity check that the timing is kernel-bound: if those two run
# at the same speed, the rung is upload/write-bound and eval savings can't show.
#
# Run from the worktree root:
#   julia --project=benchmarks --threads=auto benchmarks/newton_gpu.jl

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

function warmup()
    t = build_trajs(2)
    s = build_screen(4, 16)
    accumulate_potential(t, s, GPUKernelRK4(), backend; n_substeps = 2)
    accumulate_potential(t, s, GPUKernelNewton(), backend; n_iters = 2)
    return nothing
end

# min-of-k wall time; returns (t_min, last result)
function time_min(f, k)
    best = Inf
    local A
    for _ in 1:k
        t = @elapsed A = f()
        best = min(best, t)
    end
    return best, A
end

const configs = [
    (N = 100, Nx = 50, k = 3),
    (N = 300, Nx = 50, k = 3),
    (N = 1000, Nx = 200, k = 2),
]
const N_samples = 1000

function main()
    println("Backend: ", backend)
    println("Warming up (JIT)…")
    warmup()

    trajs_cache = Dict{Int, Any}()

    for cfg in configs
        N, Nx, k = cfg.N, cfg.Nx, cfg.k
        @printf("\n========== N=%d  screen=%d×%d  N_samples=%d  (min of %d) ==========\n",
            N, Nx, Nx, N_samples, k)
        flush(stdout)

        trajs = get!(() -> build_trajs(N), trajs_cache, N)
        screen = build_screen(Nx, N_samples)

        # Accuracy anchor: converged RK4 (absolute accuracy of this anchor is
        # pinned by diag_newton_precision.jl against the tight CPU reference).
        t_rk48, A_anchor = time_min(() -> accumulate_potential(
            trajs, screen, GPUKernelRK4(), backend; n_substeps = 8), k)

        @printf("%-22s %10s %10s %10s %10s\n", "method", "relerr", "time(s)", "x rk4ns1", "x rk4ns8")

        t_rk41, A = time_min(() -> accumulate_potential(
            trajs, screen, GPUKernelRK4(), backend; n_substeps = 1), k)
        @printf("%-22s %10.2e %10.3f %10.2f %10.2f\n",
            "RK4 n_substeps=1", relerr(A, A_anchor), t_rk41, 1.0, t_rk48 / t_rk41)
        @printf("%-22s %10s %10.3f %10.2f %10.2f\n",
            "RK4 n_substeps=8", "anchor", t_rk48, t_rk41 / t_rk48, 1.0)
        A = nothing
        flush(stdout)

        for n in (1, 2, 3)
            t, A = time_min(() -> accumulate_potential(
                trajs, screen, GPUKernelNewton(), backend; n_iters = n), k)
            @printf("%-22s %10.2e %10.3f %10.2f %10.2f\n",
                "Newton n_iters=$n", relerr(A, A_anchor), t, t_rk41 / t, t_rk48 / t)
            A = nothing
            flush(stdout)
        end

        A_anchor = nothing
        GC.gc()
    end
    return nothing
end

main()
