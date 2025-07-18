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
    radiation_model = nothing
)

# Get the system before simplification
sys_raw = electron

# Get the system after mtkcompile
sys_compiled = mtkcompile(electron)

# Get all equations
println("=== Raw System Equations (before mtkcompile) ===")
println("Number of equations: ", length(equations(sys_raw)))
println()

eqs_raw = equations(sys_raw)
# Group equations by type for better display
position_eqs = []
velocity_eqs = []
force_eqs = []
field_eqs = []
other_eqs = []

for eq in eqs_raw
    eq_str = string(eq)
    if occursin("Differential(τ)(x", eq_str) || occursin("Differential(τ)(particle.x", eq_str)
        push!(position_eqs, eq)
    elseif occursin("Differential(τ)(u", eq_str) || occursin("Differential(τ)(particle.u", eq_str)
        push!(velocity_eqs, eq)
    elseif occursin("F_", eq_str) || occursin("force", eq_str)
        push!(force_eqs, eq)
    elseif occursin("E[", eq_str) || occursin("B[", eq_str) || occursin("field", eq_str)
        push!(field_eqs, eq)
    else
        push!(other_eqs, eq)
    end
end

println("Position equations (", length(position_eqs), "):")
for eq in position_eqs[1:min(2, length(position_eqs))]
    println(latexify(eq))
end
if length(position_eqs) > 2
    println("... and ", length(position_eqs) - 2, " more")
end

println("\nVelocity equations (", length(velocity_eqs), "):")
for eq in velocity_eqs[1:min(2, length(velocity_eqs))]
    println(latexify(eq))
end
if length(velocity_eqs) > 2
    println("... and ", length(velocity_eqs) - 2, " more")
end

println("\nForce equations (", length(force_eqs), "):")
for eq in force_eqs[1:min(3, length(force_eqs))]
    println(latexify(eq))
end

println("\nField equations (", length(field_eqs), "):")
for eq in field_eqs[1:min(3, length(field_eqs))]
    println(latexify(eq))
end

println("\nOther equations (", length(other_eqs), "):")
for eq in other_eqs[1:min(3, length(other_eqs))]
    println(latexify(eq))
end

println("\n\n=== Compiled System Equations (after mtkcompile) ===")
println("Number of equations: ", length(equations(sys_compiled)))
println()

eqs_compiled = equations(sys_compiled)
println("All compiled equations:")
for (i, eq) in enumerate(eqs_compiled)
    println("$i. ", latexify(eq))
end

# Also show the states
println("\n=== System States ===")
println("Raw system unknowns (", length(unknowns(sys_raw)), "):")
for (i, var) in enumerate(unknowns(sys_raw))
    if i <= 5
        println("  ", var)
    elseif i == 6
        println("  ... and ", length(unknowns(sys_raw)) - 5, " more")
        break
    end
end

println("\nCompiled system unknowns (", length(unknowns(sys_compiled)), "):")
for var in unknowns(sys_compiled)
    println("  ", var)
end
