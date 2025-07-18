using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using OrdinaryDiffEqDefault
using OrdinaryDiffEqTsit5
using CairoMakie

# Parameters
k = 1
c = 1
a₀ = 1.0

# Create the system
@named plane_wave = PlaneWave(; k_vector = [0, 0, k])
@named electron = ChargedParticle(; external_field = plane_wave)
sys = mtkcompile(electron)

# Initial velocity for figure-8 orbit
v₀_z = -c / ((2 / a₀)^2 + 1)  # velocity in z-direction
γ₀ = 1.0 / sqrt(1 - (v₀_z/c)^2)

# Initial conditions
u0 = [
    sys.x => [0.0, 0.0, 0.0, 0.0],
    sys.u => [γ₀, 0.0, 0.0, γ₀*v₀_z]
]

# Create problem - even longer timespan to show solver differences
prob = ODEProblem(sys, u0, (0.0, 500.0))

# Solve with different methods
println("Solving with Tsit5 (1e-6 tolerance)...")
sol_tsit5 = solve(prob, Tsit5(), abstol = 1e-6, reltol=1e-3)

println("Solving with Vern9 (1e-6 tolerance)...")
sol_vern9_med = solve(prob, Vern9(), abstol = 1e-6, reltol=1e-3)

println("Solving with Vern9 (1e-9 tolerance)...")
sol_vern9 = solve(prob, Vern9(), abstol=1e-9, reltol=1e-12)

# Create a comprehensive figure
fig = Figure(size = (1200, 1400), backgroundcolor = :white)

# Top panel: Trajectory
ax1 = Axis(fig[1:3, 1:2],
    xlabel = "x",
    ylabel = "z",
    title = "Figure-8 Electron Trajectory (uniform tolerances)",
    titlesize = 26,
    xlabelsize = 22,
    ylabelsize = 22,
    xticklabelsize = 18,
    yticklabelsize = 18
)

# Plot trajectories
lines!(ax1, sol_tsit5, idxs=(sys.x[2], sys.x[4]),
    color = :blue,
    linewidth = 3,
    linestyle = :solid,
    label = "Tsit5 (a: 1e-6, r: 1e-3)"
)

lines!(ax1, sol_vern9_med, idxs=(sys.x[2], sys.x[4]),
    color = :orange,
    linewidth = 2.5,
    linestyle = :dash,
    label = "Vern9 (a: 1e-6, r: 1e-3)"
)

lines!(ax1, sol_vern9, idxs=(sys.x[2], sys.x[4]),
    color = :red,
    linewidth = 2,
    linestyle = :dot,
    label = "Vern9 (a: 1e-9, r: 1e-12)"
)

# Add legend
Legend(fig[4, 1:2], ax1, nbanks = 1, orientation = :horizontal)

# Add grid and styling
ax1.xgridvisible = true
ax1.ygridvisible = true
ax1.xgridcolor = (:gray, 0.3)
ax1.ygridcolor = (:gray, 0.3)

# Common setup for conservation metrics
u_norm = sys.u[1]^2 - sys.u[2]^2 - sys.u[3]^2 - sys.u[4]^2
ts = range(1e-3, 10, length=200)

u_norm_tsit5 = sol_tsit5(ts, idxs=u_norm).u
u_norm_vern9_med = sol_vern9_med(ts, idxs=u_norm).u
u_norm_vern9 = sol_vern9(ts, idxs=u_norm).u

γ_tsit5 = sol_tsit5(ts, idxs=sys.u[1]).u
γ_vern9_med = sol_vern9_med(ts, idxs=sys.u[1]).u
γ_vern9 = sol_vern9(ts, idxs=sys.u[1]).u

# Row 2: Tsit5 (a: 1e-6, r: 1e-3)
ax2a = Axis(fig[5, 1],
    # xlabel = "Time",
    ylabel = "|u² - c²|",
    title = "4-Velocity Norm Conservation",
    titlesize = 22,
    # xlabelsize = 20,
    xticklabelsvisible = false,
    ylabelsize = 20,
    # xticklabelsize = 16,
    yticklabelsize = 16,
    yscale = Makie.pseudolog10
)

lines!(ax2a, ts, abs.(u_norm_tsit5 .- c^2),
    color = :blue,
    linewidth = 2,
    label = "Tsit5 (a: 1e-6, r: 1e-3)"
)

axislegend(ax2a, position = :lt)

ax3a = Axis(fig[5, 2],
    # xlabel = "Time",
    ylabel = "|γ(t) - γ(0)|",
    title = "Energy Conservation",
    titlesize = 22,
    # xlabelsize = 20,
    xticklabelsvisible = false,
    ylabelsize = 20,
    # xticklabelsize = 16,
    yticklabelsize = 16,
    yscale = Makie.log10
)

lines!(ax3a, ts, abs.(γ_tsit5 .- γ₀),
    color = :blue,
    linewidth = 2,
    label = "Tsit5 (a: 1e-6, r: 1e-3)"
)

axislegend(ax3a, position=:rb)

# Row 3: Vern9 (a: 1e-6, r: 1e-3)
ax2b = Axis(fig[6, 1],
    # xlabel = "Time",
    ylabel = "|u² - c²|",
    # title = "4-Velocity Norm Conservation",
    # titlesize = 22,
    # xlabelsize = 20,
    ylabelsize = 20,
    # xticklabelsize = 16,
    xticklabelsvisible = false,
    yticklabelsize = 16,
    yscale = Makie.pseudolog10
)

lines!(ax2b, ts, abs.(u_norm_vern9_med .- c^2),
    color = :orange,
    linewidth = 2,
    label = "Vern9 (a: 1e-6, r: 1e-3)"
)

axislegend(ax2b, position = :lt)

ax3b = Axis(fig[6, 2],
    # xlabel = "Time",
    ylabel = "|γ(t) - γ(0)|",
    # title = "Energy Conservation",
    # titlesize = 22,
    # xlabelsize = 20,
    ylabelsize = 20,
    # xticklabelsize = 16,
    xticklabelsvisible = false,
    yticklabelsize = 16,
    yscale = log10
)

lines!(ax3b, ts, abs.(γ_vern9_med .- γ₀),
    color = :orange,
    linewidth = 2,
    label = "Vern9 (a: 1e-6, r: 1e-3)"
)

axislegend(ax3b, position = :rb)

# Row 4: Vern9 (1e-9)
ax2c = Axis(fig[7, 1],
    xlabel = "Time",
    ylabel = "|u² - c²|",
    # title = "4-Velocity Norm Conservation",
    # titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 16,
    yticklabelsize = 16,
    yscale = Makie.pseudolog10
)

lines!(ax2c, ts, abs.(u_norm_vern9 .- c^2),
    color = :red,
    linewidth = 2,
    label = "Vern9 (a: 1e-9, r: 1e-12)"
)

axislegend(ax2c, position = :lt)

ax3c = Axis(fig[7, 2],
    xlabel = "Time",
    ylabel = "|γ(t) - γ(0)|",
    # title = "Energy Conservation",
    # titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 16,
    yticklabelsize = 16,
    yscale = log10
)

lines!(ax3c, ts, abs.(γ_vern9 .- γ₀),
    color = :red,
    linewidth = 2,
    label = "Vern9 (a: 1e-9, r: 1e-12)"
)

axislegend(ax3c, position = :rb)


# Save as vector graphics (SVG only)
save("figure8.svg", fig)

# Also save PNG for preview
save("figure8.png", fig, px_per_unit = 2)

println("Figure-8 plots saved as figure8.svg and figure8.png")

# Print some statistics
println("\nSolver statistics:")
println("Tsit5 (1e-6):")
println("  Time steps: $(length(sol_tsit5.t))")
println("  Function evaluations: $(sol_tsit5.stats.nf)")
println("  Accepted steps: $(sol_tsit5.stats.naccept)")
println("Vern9 (1e-6):")
println("  Time steps: $(length(sol_vern9_med.t))")
println("  Function evaluations: $(sol_vern9_med.stats.nf)")
println("  Accepted steps: $(sol_vern9_med.stats.naccept)")
println("Vern9 (1e-9):")
println("  Time steps: $(length(sol_vern9.t))")
println("  Function evaluations: $(sol_vern9.stats.nf)")
println("  Accepted steps: $(sol_vern9.stats.naccept)")

# Calculate difference at final time
x_diff = abs(sol_tsit5[sys.x[2], end] - sol_vern9[sys.x[2], end])
z_diff = abs(sol_tsit5[sys.x[4], end] - sol_vern9[sys.x[4], end])
println("\nDifference at t=$(prob.tspan[2]):")
println("Δx = $x_diff")
println("Δz = $z_diff")

# Conservation errors
println("\nMax 4-velocity norm error:")
println("Tsit5 (1e-6): $(maximum(abs.(u_norm_tsit5 .- c)))")
println("Vern9 (1e-6): $(maximum(abs.(u_norm_vern9_med .- c)))")
println("Vern9 (1e-9): $(maximum(abs.(u_norm_vern9 .- c)))")

println("\nMax energy error:")
println("Tsit5: $(maximum(abs.(γ_tsit5 .- γ₀)))")
println("Vern9: $(maximum(abs.(γ_vern9 .- γ₀)))")


############ Non-uniform tolerance

low_abstols = [fill(1e-3, 7); 1e-5]
low_reltols = 1e-3

sol_vern7_low_tol = solve(prob, Vern7(), abstol = low_abstols, reltol=low_reltols)

u_norm_vern9_low = sol_vern7_low_tol(ts, idxs=u_norm).u
u_norm_vern9 = sol_vern9(ts, idxs=u_norm).u
