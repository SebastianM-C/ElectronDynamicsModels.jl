using ElectronDynamicsModels
using ElectronDynamicsModels: retarded_time_rhs, advanced_time, _accumulate_pixel!
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra

# ── Setup (same as thomson_scattering.jl but smaller) ──────────────

const c = 137.03599908330932
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ

a₀ = 0.01  # small a₀ for quick solve

@named ref_frame = ProperFrame(:atomic)
@named laser = PlaneWave(; frequency = ω, amplitude = a₀, ref_frame)
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

# Screen
const Z = 2.0e5λ
const δt = 2π / ω / 4
const N_samples = floor(Int, (τf - τi) / δt)
const x⁰_start = c * τi + hypot(Z, 25w₀)

Nx = 50
Ny = 50

x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)
screen = ObserverScreen(
    LinRange(-25w₀, 25w₀, Nx),
    LinRange(-25w₀, 25w₀, Ny),
    Z,
    x⁰_samples
)

# ── Benchmark: retarded-time solve only ─────────────────────────────

function bench_retarded_time_solve(traj, screen, alg; solve_kwargs...)
    x⁰_samples = screen.x⁰_samples
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    τi = first(traj.itp.t)
    τf = last(traj.itp.t)

    r_obs_0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
    x⁰_i_0 = advanced_time(traj, τi, r_obs_0)
    x⁰_f_0 = advanced_time(traj, τf, r_obs_0)
    proto_prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs, τi, (x⁰_i_0, x⁰_f_0), (traj, r_obs_0)
    )

    # Store retarded time solutions
    τ_solutions = Matrix{Vector{Float64}}(undef, Nx, Ny)

    nworkers = Threads.nthreads()
    integ_pool = Channel{Any}(nworkers)
    for _ in 1:nworkers
        put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, solve_kwargs...))
    end

    Threads.@threads for ix in Base.OneTo(Nx)
        integ = take!(integ_pool)
        for iy in Base.OneTo(Ny)
            r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
            x⁰_i = advanced_time(traj, τi, r_obs)
            x⁰_f = advanced_time(traj, τf, r_obs)

            integ.p = (traj, r_obs)
            reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
            solve!(integ)

            τ_solutions[ix, iy] = copy(integ.sol.u)
        end
        put!(integ_pool, integ)
    end

    return τ_solutions
end

# ── Benchmark: accumulation only ────────────────────────────────────

function bench_accumulate_only(traj, screen, τ_solutions)
    N_samples = length(screen.x⁰_samples)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    A = zeros(N_samples, 4, Nx, Ny)

    Threads.@threads for ix in Base.OneTo(Nx)
        for iy in Base.OneTo(Ny)
            _accumulate_pixel!(A, traj, screen, τ_solutions[ix, iy], ix, iy)
        end
    end

    return A
end

# ── Run benchmarks ──────────────────────────────────────────────────

traj = trajs[1]

println("Threads: $(Threads.nthreads())")
println("Time samples: $(N_samples)")
println()

for Npx in [10, 50, 100, 200, 300]
    sc = ObserverScreen(
        LinRange(-25w₀, 25w₀, Npx),
        LinRange(-25w₀, 25w₀, Npx),
        Z,
        x⁰_samples
    )

    # Warmup
    τ_sol = bench_retarded_time_solve(traj, sc, Tsit5())
    bench_accumulate_only(traj, sc, τ_sol)

    nruns = Npx ≤ 100 ? 3 : 1

    t_solve = @elapsed for _ in 1:nruns
        bench_retarded_time_solve(traj, sc, Tsit5())
    end
    t_solve /= nruns

    τ_sol = bench_retarded_time_solve(traj, sc, Tsit5())

    t_accum = @elapsed for _ in 1:nruns
        bench_accumulate_only(traj, sc, τ_sol)
    end
    t_accum /= nruns

    pct = round(t_accum / (t_solve + t_accum) * 100, digits=1)
    println("$(lpad(Npx, 3))×$(rpad(Npx, 3))  solve=$(lpad(round(t_solve, digits=4), 8))s  accum=$(lpad(round(t_accum, digits=4), 8))s  accum=$(lpad(pct, 5))%")
end
