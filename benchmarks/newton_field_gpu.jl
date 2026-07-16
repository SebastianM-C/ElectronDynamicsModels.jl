# Benchmark: GPUKernelNewton vs GPUKernelRK4 on the FIELD path (the production
# path; register-pressure risk case — see the findfirst-spline experiment).
# Split mode (production default). Physics identical to benchmark_newton_gpu.jl.
#
# Run from the worktree root:
#   julia --project=benchmarks --threads=auto benchmarks/newton_field_gpu.jl

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
relerr_fields(F, Fref) = max(relerr(F.E, Fref.E), relerr(F.B, Fref.B))

function warmup()
    t = build_trajs(2)
    s = build_screen(4, 16)
    accumulate_field(t, s, GPUKernelRK4(), backend; n_substeps = 2)
    accumulate_field(t, s, GPUKernelNewton(), backend; n_iters = 2)
    return nothing
end

function time_min(f, k)
    best = Inf
    local A
    for _ in 1:k
        t = @elapsed A = f()
        best = min(best, t)
    end
    return best, A
end

# (300, 400²) split-mode buffers ≈ 4 × 3.84 GB ≈ 15.4 GB — fits the W7900.
const configs = [
    (N = 300, Nx = 200, k = 2, rk4_sweep = (1, 8), newton_sweep = (1, 2, 3)),
    (N = 300, Nx = 400, k = 1, rk4_sweep = (1,), newton_sweep = (2,)),
]
const N_samples = 1000

function main()
    println("Backend: ", backend)
    println("Warming up (JIT)…")
    warmup()

    trajs_cache = Dict{Int, Any}()

    for cfg in configs
        N, Nx, k = cfg.N, cfg.Nx, cfg.k
        @printf("\n========== FIELD (split)  N=%d  screen=%d×%d  N_samples=%d  (min of %d) ==========\n",
            N, Nx, Nx, N_samples, k)
        flush(stdout)

        trajs = get!(() -> build_trajs(N), trajs_cache, N)
        screen = build_screen(Nx, N_samples)

        # Anchor: converged RK4 at ns=8 when it's in the sweep, else ns=1.
        ns_anchor = maximum(cfg.rk4_sweep)
        t_anchor, F_anchor = time_min(() -> accumulate_field(
            trajs, screen, GPUKernelRK4(), backend; n_substeps = ns_anchor), k)

        @printf("%-22s %10s %10s %10s\n", "method", "relerr", "time(s)", "x anchor")
        @printf("%-22s %10s %10.3f %10.2f\n", "RK4 n_substeps=$ns_anchor", "anchor", t_anchor, 1.0)
        flush(stdout)

        for ns in cfg.rk4_sweep
            ns == ns_anchor && continue
            t, F = time_min(() -> accumulate_field(
                trajs, screen, GPUKernelRK4(), backend; n_substeps = ns), k)
            @printf("%-22s %10.2e %10.3f %10.2f\n",
                "RK4 n_substeps=$ns", relerr_fields(F, F_anchor), t, t_anchor / t)
            F = nothing
            flush(stdout)
        end

        for n in cfg.newton_sweep
            t, F = time_min(() -> accumulate_field(
                trajs, screen, GPUKernelNewton(), backend; n_iters = n), k)
            @printf("%-22s %10.2e %10.3f %10.2f\n",
                "Newton n_iters=$n", relerr_fields(F, F_anchor), t, t_anchor / t)
            F = nothing
            flush(stdout)
        end

        F_anchor = nothing
        GC.gc()
    end
    return nothing
end

main()
