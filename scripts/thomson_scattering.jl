using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using OrdinaryDiffEqNonlinearSolve
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using FFTW
using CairoMakie
using AcceleratedKernels
# using CUDA
using AMDGPU
using Serialization
using Printf

# Atomic units
const c = 137.03599908330932

# Laser parameters
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ
Rmax = 3.25w₀

a₀ = 0.1

@named world = Worldline(:τ, :atomic)

@named laser = LaguerreGaussLaser(;
    wavelength = λ,
    a0 = a₀,
    beam_waist = w₀,
    radial_index = 2,
    azimuthal_index = -2,
    world,
    temporal_profile = :gaussian,
    temporal_width = τ,
    focus_position = 0.0,
    polarization = :circular
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

# Time span
τi = -8τ
τf = 8τ
tspan = (τi, τf)

# Single electron solve (for parameter access)
x⁰ = [τi * c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]

u0 = [
    sys.x => x⁰,
    sys.u => u⁰,
]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
)
sol0 = solve(prob, Vern9(), reltol = 1.0e-15, abstol = 1.0e-12)

# Sunflower distribution for electron positions
const ϕ = (1 + √5) / 2

function radius(k, n, b)
    if k > n - b
        return 1.0
    else
        return sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
    end
end

function sunflower(n, α)
    points = Vector{Vector{Float64}}()
    angle_stride = 2π / ϕ^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        θ = k * angle_stride
        push!(points, [r * cos(θ), r * sin(θ)])
    end
    return points
end

# Ensemble solve
N = 10_000
R₀ = Rmax * sunflower(N, 2)
xμ = [[τi * c, r..., 0.0] for r in R₀]

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, ctx)
    i = ctx.sim_id
    x_new = SVector{4}(xμ[i]...)
    u_new = SVector{4}(c, 0.0, 0.0, 0.0)
    u0, p = set_x(prob, SVector{8}(x_new..., u_new...))
    return remake(prob; u0, p)
end

function abserr(a₀)
    amp = log10(a₀)
    expo = -amp^2 / 27 + 32amp / 27 - 220 / 27
    return 10^expo
end

ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    reltol = 1.0e-12, abstol = abserr(a₀), trajectories = N
)

# Radiation computation

trajs = trajectory_interpolants(solution)

# Screen parameters
const Z = 2.0e5λ
const samples_per_period = 16          # Nyquist = 8× fundamental
const δt = 2π / ω / samples_per_period
const N_samples = 8000                 # ≈500λ window
const x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)

Nx = 400
Ny = 400

x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)

screen = ObserverScreen(
    LinRange(-25w₀, 25w₀, Nx),
    LinRange(-25w₀, 25w₀, Ny),
    Z,
    x⁰_samples
)

# @time A_cpu = accumulate_potential(trajs, screen, Tsit5());

# GPU (fully-on-GPU GPUKernelRK4 path, n_substeps=1)
@time A_rk4 = accumulate_potential(trajs, screen, GPUKernelRK4(), AMDGPU.ROCBackend(); n_substeps = 1);

# @info norm(A_cpu - A_rk4) / norm(A_cpu)

A_s = A_rk4

# Serialize the raw 4-potential so the offline scripts (plot_harmonic_ladder.jl,
# plot_harmonics.jl, plot_power_spectrum.jl) can read this run directly.
datafile = "A_rk4_$(Nx)_N$(N)_Ns$(N_samples)_spp$(samples_per_period).jls"
serialize(datafile, A_s)
println("serialized → $datafile")

# ── Harmonic maps ──
# Extract the first two harmonics of ω₁ for all four 4-potential components.
# rfft one component at a time (the full complex spectrum is 4× the raw array),
# mirroring plot_harmonic_ladder.jl's memory-conscious slicing.
const complabels = ("A⁰", "Aˣ", "Aʸ", "Aᶻ")
const harmonics = (1, 2)

freqs = rfftfreq(N_samples, 1 / δt)
harmonic_bins = [findmin(x -> abs(x - n * ω / 2π), freqs)[2] for n in harmonics]

fields = Array{ComplexF64, 4}(undef, length(harmonics), 4, Nx, Ny)
for μ in 1:4
    A_ω_c = rfft(A_s[:, μ, :, :], 1)
    for (k, idx) in enumerate(harmonic_bins)
        fields[k, μ, :, :] = A_ω_c[idx, :, :]
    end
    A_ω_c = nothing
    GC.gc()
end

# One figure per harmonic, 2×2 over the four components; each panel scaled to its
# own peak since component amplitudes differ by orders.
function plot_harmonic(k, n)
    idx = harmonic_bins[k]
    fig = Figure()
    Label(fig[0, :], @sprintf("Thomson scattering — %dω₁ (%.3f× fundamental)",
            n, freqs[idx] / (ω / 2π)), fontsize = 16, font = :bold)
    for μ in 1:4
        field = real.(fields[k, μ, :, :])
        cr = maximum(abs, field)
        gl = fig[cld(μ, 2), (μ - 1) % 2 + 1] = GridLayout()
        ax = Axis(gl[1, 1], width = 340, height = 340, xlabel = "x", ylabel = "y",
            title = @sprintf("%s  (peak %.2e)", complabels[μ], cr))
        hm = heatmap!(ax, collect(screen.x_grid), collect(screen.y_grid), field,
            colorrange = (-cr, cr), colormap = :seismic)
        Colorbar(gl[1, 2], hm, width = 12, height = 340)
    end
    resize_to_layout!(fig)
    out = @sprintf("thomson_scattering_h%d.png", n)
    save(out, fig)
    println("saved → $out")
    return fig
end

for (k, n) in enumerate(harmonics)
    plot_harmonic(k, n)
end
