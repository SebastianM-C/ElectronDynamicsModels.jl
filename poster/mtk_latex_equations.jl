using Pkg
Pkg.activate(@__DIR__)

using ElectronDynamicsModels
using ModelingToolkit
using Latexify

# Create a simple system to show the transformation
@named plane_wave = PlaneWave(
    amplitude = 1.0,
    frequency = 1.0,
    k_vector = [0, 0, 1]
)

@named electron = ChargedParticle(
    external_field = plane_wave,
    radiation_model = :landau_lifshitz
)

# Get the system before simplification
sys_raw = electron

# Get the system after mtkcompile
sys_compiled = mtkcompile(electron)

# LaTeXify the equations
println("=== Raw System Equations (before mtkcompile) ===")
println("Number of equations: ", length(equations(sys_raw)))
println()

# Get a subset of interesting equations to show
eqs_raw = equations(sys_raw)
# Show position evolution, velocity evolution, and constraint
interesting_indices = [1, 5, 9]  # dx/dτ, du/dτ, constraint
for i in interesting_indices
    if i <= length(eqs_raw)
        println(latexify(eqs_raw[i]))
        println()
    end
end

println("\n=== Compiled System Equations (after mtkcompile) ===")
println("Number of equations: ", length(equations(sys_compiled)))
println()

# Show simplified equations
eqs_compiled = equations(sys_compiled)
# Show a few key simplified equations
for i in 1:min(3, length(eqs_compiled))
    println(latexify(eqs_compiled[i]))
    println()
end

# Also generate the full system info
println("\n=== System Structure ===")
println("Raw system states: ", length(unknowns(sys_raw)))
println("Compiled system states: ", length(unknowns(sys_compiled)))
println("Raw system parameters: ", length(parameters(sys_raw)))
println("Compiled system parameters: ", length(parameters(sys_compiled)))