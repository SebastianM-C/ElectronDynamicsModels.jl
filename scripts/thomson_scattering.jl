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

# Atomic units
const c = 137.03599908330932

# Laser parameters
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ
Rmax = 3.25w₀

a₀ = 10.0

@named ref_frame = ProperFrame(:atomic)

@named laser = LaguerreGaussLaser(;
    wavelength = λ,
    a0 = a₀,
    beam_waist = w₀,
    radial_index = 2,
    azimuthal_index = -2,
    ref_frame,
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
N = 300
R₀ = Rmax * sunflower(N, 2)
xμ = [[τi * c, r..., 0.0] for r in R₀]

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, i, repeat)
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
const δt = 2π / ω / 4
const N_samples = floor(Int, (τf - τi) / δt)
const x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)

Nx = 50
Ny = 50

x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)

screen = ObserverScreen(
    LinRange(-25w₀, 25w₀, Nx),
    LinRange(-25w₀, 25w₀, Ny),
    Z,
    x⁰_samples
)

K = sol0.ps[ref_frame.q_e / (4π * ref_frame.ε₀ * c)]

A_s = accumulate_potential(trajs, screen, K, alg = Tsit5())
A_ω = rfft(A_s, 1)

# Find fundamental frequency bin
freqs = rfftfreq(N_samples, 1 / δt)
idx_f1 = findmin(x -> abs(x - ω / 2π), freqs)[2]

# Plot the y-component of the potential at the fundamental frequency
fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, xlabel = "x", ylabel = "y", title = "Thomson scattering (ω₁)")
field = real.(A_ω[idx_f1, 3, :, :])
heatmap!(
    ax, collect(screen.x_grid), collect(screen.y_grid), field,
    colorrange = maximum(abs, field) .* (-1, 1), colormap = :seismic
)
fig
