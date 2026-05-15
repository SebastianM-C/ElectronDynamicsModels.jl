# Standalone smoke test for the unified GPU kernel path.
# Compares accumulate_potential(..., Tsit5())  (CPU adaptive reference)
# against accumulate_potential(..., GPUKernelRK4(), CUDABackend())  on a
# tiny problem (~5 electrons, 21×21 pixels, 500 saveat slots).

using ElectronDynamicsModels
using ModelingToolkit
using ModelingToolkit: getdefault
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using Unitful, UnitfulAtomic
using CUDA

@named world = Worldline(:τ, :atomic)

c = getdefault(world.c)
m_e = getdefault(world.m_e)

# ── Minimal laser (matches OAM script geometry, lower a0 for fast solve) ──
τ_pulse = austrip(2u"ps")
λ = austrip(800u"nm")
ω = 2π * c / λ
w₀ = austrip(40u"μm")
W = austrip(0.01u"J")     # 100× smaller pulse energy than OAM script — keeps electron mostly drifting (cheap solve)
a0 = a0_from_pulse_energy(W, w₀, τ_pulse, ω; world, mode = (p = 0, m = 2))

@named laser = LaguerreGaussLaser(;
    wavelength = λ,
    a0,
    beam_waist = w₀,
    k_direction = [0, 0, -1],
    radial_index = 0,
    azimuthal_index = 2,
    world,
    temporal_profile = :gaussian,
    temporal_width = τ_pulse,
    focus_position = 0.0,
    polarization = :circular,
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

# ── Tiny electron set: deterministic ring of 5 ──
KE = austrip(25u"MeV")
γ_e = 1 + KE / (m_e * c^2)
β_e = sqrt(1 - 1 / γ_e^2)

N_macro = 5

t_initial_lab = -8 * τ_pulse
z_centroid_lab = β_e * c * t_initial_lab
τ_span = -t_initial_lab / γ_e * 2

xμ = SVector{4, Float64}[]
for i in 1:N_macro
    θ = 2π * (i - 1) / N_macro
    push!(xμ, SVector{4, Float64}(c * t_initial_lab, w₀ * cos(θ), w₀ * sin(θ), z_centroid_lab))
end

x_init = xμ[1]
u_init = c * [γ_e, 0, 0, γ_e * β_e]

u0 = [sys.x => x_init, sys.u => u_init]
prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, (zero(τ_span), τ_span);
    u0_constructor = SVector{8}, fully_determined = true,
)
set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, i, repeat)
    u0_new, p = set_x(prob, SVector{8}(xμ[i]..., u_init...))
    return remake(prob; u0 = u0_new, p)
end

ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
println("Solving $N_macro trajectories...")
@time solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    saveat = LinRange(0, τ_span, 10^4),
    reltol = 1.0e-9, abstol = 1.0e-9,
    trajectories = N_macro,
)

trajs = trajectory_interpolants(solution)

# ── Tiny screen ──
L_screen = austrip(1u"m")
half_extent = austrip(5u"μm")
Nx = Ny = 21
x_grid = LinRange(-half_extent, half_extent, Nx)
y_grid = LinRange(-half_extent, half_extent, Ny)

emit_window = c * (-t_initial_lab) / γ_e^2
margin = emit_window
x⁰_lo = L_screen - margin
x⁰_hi = L_screen + margin
N_samples = 500
x⁰_samples = LinRange(x⁰_lo, x⁰_hi, N_samples)
screen = ObserverScreen(x_grid, y_grid, L_screen, x⁰_samples)

println("\nProblem size: Nx=$Nx, Ny=$Ny, N_macro=$N_macro, N_samples=$N_samples")

# ── CPU reference ──
println("\n── CPU reference (Tsit5 adaptive) ──")
@time A_cpu = accumulate_potential(trajs, screen, Tsit5())

# ── GPU unified kernel ──
println("\n── GPU unified kernel (RK4 fixed-step) ──")
@time A_gpu = accumulate_potential(trajs, screen, GPUKernelRK4(), CUDA.CUDABackend())

# ── Compare ──
abs_err = maximum(abs, A_cpu .- A_gpu)
ref_max = maximum(abs, A_cpu)
rel_err = abs_err / max(ref_max, eps())

println("\n── Results ──")
println("  CPU peak |A|       : $ref_max")
println("  GPU peak |A|       : $(maximum(abs, A_gpu))")
println("  max |A_cpu - A_gpu|: $abs_err")
println("  relative error     : $rel_err")

if rel_err < 1.0e-4
    println("\n✓ GPU path agrees with CPU within 1e-4 (RK4 vs adaptive-Tsit5 tolerance acceptable)")
else
    println("\n✗ GPU path disagrees with CPU — investigate before trusting on the full problem")
end
