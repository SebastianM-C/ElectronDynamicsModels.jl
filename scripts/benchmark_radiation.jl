using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using CUDA
using AcceleratedKernels

# ── Setup (same as thomson_scattering.jl but single electron) ───────

const c = 137.03599908330932
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ

a₀ = 0.01  # small a₀ for quick solve

@named world = Worldline(:τ,:atomic)
@named laser = PlaneWave(; frequency = ω, amplitude = a₀, world)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

τi = -4τ
τf = 4τ
tspan = (τi, τf)

x⁰ = [τi * c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]

u0 = [sys.x => x⁰, sys.u => u⁰]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
)
sol = solve(prob, Vern9(), reltol = 1e-12, abstol = 1e-12)
trajs = trajectory_interpolants(SciMLBase.EnsembleSolution([sol], 0.0, true, sol))

# Screen parameters
const Z = 2.0e5λ
const δt = 2π / ω / 4
const N_samples = floor(Int, (τf - τi) / δt)
const x⁰_start = c * τi + hypot(Z, 25w₀)

x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)

# ── Run benchmarks ──────────────────────────────────────────────────

backend = CUDA.CUDABackend()

println("Threads: $(Threads.nthreads())")
println("Time samples: $(N_samples)")
println("GPU: $(CUDA.name(CUDA.device()))")
println()

# ── Scaling with screen size (1 electron) ───────────────────────────
println("=== Screen size scaling (1 electron) ===")
for Npx in [10, 50, 100, 200, 300]
    sc = ObserverScreen(
        LinRange(-25w₀, 25w₀, Npx),
        LinRange(-25w₀, 25w₀, Npx),
        Z,
        x⁰_samples
    )

    # Warmup
    accumulate_potential(trajs, sc, Tsit5())
    accumulate_potential(trajs, sc, Tsit5(), backend)

    nruns = Npx ≤ 100 ? 3 : 1

    t_cpu = @elapsed for _ in 1:nruns
        accumulate_potential(trajs, sc, Tsit5())
    end
    t_cpu /= nruns

    t_gpu = @elapsed for _ in 1:nruns
        accumulate_potential(trajs, sc, Tsit5(), backend)
    end
    t_gpu /= nruns

    speedup = round(t_cpu / t_gpu, digits=1)
    println("$(lpad(Npx, 3))×$(rpad(Npx, 3))  cpu=$(lpad(round(t_cpu, digits=3), 7))s  gpu=$(lpad(round(t_gpu, digits=3), 7))s  speedup=$(lpad(speedup, 5))×")
end

# ── Scaling with electron count (50×50 screen) ──────────────────────
println()
println("=== Electron count scaling (50×50 screen) ===")
sc = ObserverScreen(
    LinRange(-25w₀, 25w₀, 50),
    LinRange(-25w₀, 25w₀, 50),
    Z,
    x⁰_samples
)

# Create multiple copies of the trajectory to simulate many electrons
for Ne in [1, 10, 50, 100, 300]
    multi_trajs = repeat(trajs, Ne)

    # Warmup (small)
    if Ne ≤ 10
        accumulate_potential(multi_trajs, sc, Tsit5())
        accumulate_potential(multi_trajs, sc, Tsit5(), backend)
    end

    nruns = Ne ≤ 50 ? 2 : 1

    t_cpu = @elapsed for _ in 1:nruns
        accumulate_potential(multi_trajs, sc, Tsit5())
    end
    t_cpu /= nruns

    t_gpu = @elapsed for _ in 1:nruns
        accumulate_potential(multi_trajs, sc, Tsit5(), backend)
    end
    t_gpu /= nruns

    speedup = round(t_cpu / t_gpu, digits=1)
    println("$(lpad(Ne, 3)) e⁻  cpu=$(lpad(round(t_cpu, digits=3), 7))s  gpu=$(lpad(round(t_gpu, digits=3), 7))s  speedup=$(lpad(speedup, 5))×")
end
