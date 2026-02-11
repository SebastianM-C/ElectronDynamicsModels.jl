using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using GLMakie
using LaTeXStrings

const inch = 96
const pt = 4/3
const cm = inch / 2.54

# set_theme!(
#     fonts = (;
#         regular = "CMU Serif Roman",
#         bold = "CMU Serif Roman Bold"),
#     fontsize = 16pt,
# )

# Code is using Atomic Units !!!
# natural constants
c = 137.03599908330932 # speed of light
qme = -1. # specific charge

h = 2π
α = 1/c
ε₀=qme^2/(2α*h*c)
μ₀=1/(ε₀*c^2)

# derived
ω = 0.057
τ = 150/ω
λ = 2π*c/ω
w₀ = 75λ
Rmax = 3.25w₀

ξx, ξy = (1/√2, im/√2) .|> complex

# Laser parameters in atomic units
λ_au = λ
a₀ = 0.1
w₀_au = w₀
p_index = 2
m_index = -2

# Convert temporal profile to ElectronDynamicsModels format
τ_fwhm = τ
z₀ = 0.0

# Create spacetime and laser using ElectronDynamicsModels
@named ref_frame = ProperFrame(:atomic)
# @named ref_frame = LabFrame(:atomic)

@named laser = LaguerreGaussLaser(
    wavelength=λ_au,
    a0=a₀,
    beam_waist=w₀_au,
    radial_index=p_index,
    azimuthal_index=m_index,
    ref_frame=ref_frame,
    temporal_profile=:gaussian,  # Using Gaussian profile
    temporal_width=τ_fwhm,
    focus_position=z₀,
    polarization=:circular
)

# Create electron system
@named lg_elec = ClassicalElectron(; laser, ref_frame)


# Compile the system
sys = mtkcompile(lg_elec)

# Time span
τi = -8τ
τf = 8τ
tspan = (τi, τf)

# Create base problem with placeholder initial position
x⁰ = [τi*c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]

u0 = [
    (sys.x) => x⁰,
    (sys.u) => u⁰
]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(sys, u0, tspan, u0_constructor=SVector{8}, fully_determined=true)
sol0 = solve(prob, Vern9(), reltol = 1e-15, abstol = 1e-12)

# Sunflower pattern for initial positions
N = 900

const ϕ = (1 + √5)/2

function radius(k, n, b)
    if k > n - b
        return 1.0
    else
        return sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
    end
end

function sunflower(n, α)
    points = []
    angle_stride = 2π/ϕ^2 # geodesic ? 360 * ϕ :
    b = round(Int, α * sqrt(n))  # number of boundary points

    for k in 1:n
        r = radius(k, n, b)
        θ = k * angle_stride
        append!(points, ([r * cos(θ), r * sin(θ)], ))
    end

    return points
end

# Generate initial positions in sunflower pattern
R₀ = Rmax*sunflower(N, 2)
xμ = [[τi*c, r..., 0.] for r in R₀]

# Use SymbolicIndexingInterface to set positions
set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)]);

# Problem function to set different initial positions for each electron
function prob_func(prob, i, repeat)
    # Get position for this electron
    x_new = SVector{4}(xμ[i]...)
    # γ₀ = 1.0 / sqrt(1 - (vz/c)^2)

    # Initial momentum - electron at rest
    u_new = SVector{4}(c, 0.0, 0.0, 0.0)

    # Set new initial conditions
    u0, p = set_x(prob, SVector{8}(x_new..., u_new...))

    remake(prob; u0, p)
end

# Absolute error tolerance function
function abserr(a₀)
    amp = log10(a₀)
    expo = -amp^2/27 + 32amp/27 - 220/27
    10^expo
end

# Create ensemble problem
ensemble = EnsembleProblem(prob; prob_func, safetycopy=false)

# Solve ensemble
solution = solve(ensemble, Vern9(), EnsembleThreads();
                    reltol=1e-12, abstol=abserr(a₀),
                    trajectories=N)

# ru = solution.u[1](range(τi, τf, 1001), idxs = [sys.x; sys.u])

# Solve single trajectory for visualization (electron #1)
x_single = SVector{8}(xμ[1]..., u⁰...)
u0_single, p_single = set_x(prob, x_single)
prob_single = remake(prob; u0=u0_single, p=p_single)

sol = solve(prob_single, Vern9(),
            reltol=1e-12,
            abstol=1e-20)

#### eval field

_t = 0
_x = sol[sys.x, 500]

x_sub = map(x->EvalAt(_t)(x[1])=>x[2], collect(sys.x .=> _x))
eval_point = [laser.τ=>0; x_sub; sys.t => EvalAt(_t)(sys.x[1]) / c]

all_eqs = Symbolics.fixpoint_sub(equations(laser), merge(initial_conditions(laser), initial_conditions(eval_point)))
eq_dict = Dict(map(eq->eq.lhs=>eq.rhs, all_eqs[setdiff(1:19, 10:15)]))
Symbolics.fixpoint_sub(all_eqs, eq_dict)

# using CairoMakie
# Visualization
fig = Figure(fontsize=14pt)
# ax = Axis3(fig[1, 1], aspect=:data)
ax = Axis3(fig[1, 1], aspect=(1, 1, 1))


# Extract trajectory
t_range = range(τi, τf, length=10001)
x_traj = [sol(t, idxs=sys.x[2]) for t in t_range]
y_traj = [sol(t, idxs=sys.x[3]) for t in t_range]
z_traj = [sol(t, idxs=sys.x[4]) for t in t_range]

lines!(ax, x_traj, y_traj, z_traj)

fig
