using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using CairoMakie

# Create systems with and without radiation reaction
a₀ = 1.0  # Lower normalized amplitude to ensure physical behavior

# Use plane wave for both
@named plane_wave = PlaneWave(amplitude=a₀, frequency=1.0, k_vector=[0,0,1])

# Classical electron (no radiation)
@named electron_classical = ChargedParticle(
    radiation_model = nothing,
    external_field = plane_wave
)

# Radiating electron (Landau-Lifshitz)
@named electron_radiating = ChargedParticle(
    radiation_model = :landau_lifshitz,
    external_field = plane_wave
)

sys_classical = mtkcompile(electron_classical)
sys_radiating = mtkcompile(electron_radiating)

# Initial conditions - electron initially moving slowly
γ₀ = 1.1
v₀ = sqrt(1 - 1/γ₀^2)

u0_classical = [
    sys_classical.x => [0.0, 0.0, 0.0, 0.0],
    sys_classical.u => [γ₀, 0.0, 0.0, v₀]
]

u0_radiating = [
    sys_radiating.x => [0.0, 0.0, 0.0, 0.0],
    sys_radiating.u => [γ₀, 0.0, 0.0, v₀]
]

# Time span 
tspan = (0.0, 50.0)

# Solve both systems
prob_classical = ODEProblem(sys_classical, u0_classical, tspan)
sol_classical = solve(prob_classical, Vern9(), abstol=1e-9, reltol=1e-9)

prob_radiating = ODEProblem(sys_radiating, u0_radiating, tspan)
sol_radiating = solve(prob_radiating, Vern9(), abstol=1e-9, reltol=1e-9)

# Create figure showing radiation effects
fig = Figure(resolution = (900, 700), backgroundcolor = :white)

# Top panel: Trajectories
ax1 = Axis(fig[1, 1],
    xlabel = "z",
    ylabel = "x",
    title = "Electron Trajectory in Plane Wave (a₀ = $a₀)",
    titlesize = 24,
    xlabelsize = 20,
    ylabelsize = 20
)

# Extract trajectories
x_classical = sol_classical[sys_classical.x[2], :]
z_classical = sol_classical[sys_classical.x[4], :]

x_radiating = sol_radiating[sys_radiating.x[2], :]
z_radiating = sol_radiating[sys_radiating.x[4], :]

lines!(ax1, z_classical, x_classical,
    color = :blue,
    linewidth = 3,
    label = "Classical"
)

lines!(ax1, z_radiating, x_radiating,
    color = :red,
    linewidth = 2,
    linestyle = :dash,
    label = "Landau-Lifshitz"
)

axislegend(ax1, position = :lt)

# Bottom panel: Energy evolution
ax2 = Axis(fig[2, 1],
    xlabel = "Time",
    ylabel = "γ",
    title = "Energy Evolution",
    titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20
)

t_classical = sol_classical.t
γ_classical = sol_classical[sys_classical.u[1], :]

t_radiating = sol_radiating.t
γ_radiating = sol_radiating[sys_radiating.u[1], :]

lines!(ax2, t_classical, γ_classical,
    color = :blue,
    linewidth = 3,
    label = "Classical"
)

lines!(ax2, t_radiating, γ_radiating,
    color = :red,
    linewidth = 2,
    linestyle = :dash,
    label = "Landau-Lifshitz"
)

axislegend(ax2, position = :rt)

# Add grid
ax1.xgridvisible = true
ax1.ygridvisible = true
ax2.xgridvisible = true
ax2.ygridvisible = true

# Save
save("radiation_reaction.svg", fig)
save("radiation_reaction.png", fig, px_per_unit = 2)

println("Radiation reaction figure saved")
println("Final energy (classical): γ = $(γ_classical[end])")
println("Final energy (radiating): γ = $(γ_radiating[end])")
println("Energy loss due to radiation: $(round((γ_classical[end] - γ_radiating[end]), digits=3))")