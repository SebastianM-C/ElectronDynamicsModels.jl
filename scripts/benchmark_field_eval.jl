using ElectronDynamicsModels
using ModelingToolkit
using LaserTypes: LaserTypes
using BenchmarkTools
using StaticArrays

# Common parameters (atomic units)
c = 137.03599908330932
ω = 0.057
T₀ = 2π / ω
λ_val = c * T₀
w₀_val = 75 * λ_val
a₀_val = 2.0
p_val = 1
m_val = 1

# --- FieldEvaluator setup ---
@named ref_frame = ProperFrame(:atomic)
@named laser = LaguerreGaussLaser(
    wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val,
    radial_index=p_val, azimuthal_index=m_val,
    ref_frame=ref_frame, temporal_profile=:constant)

fe = FieldEvaluator(laser, ref_frame)

# --- LaserTypes setup ---
lt_laser = LaserTypes.LaguerreGaussLaser(:atomic;
    λ=λ_val, a₀=a₀_val, w₀=w₀_val, p=p_val, m=m_val)

# Test point
x, y, z = 0.3w₀_val, 0.1w₀_val, 0.5w₀_val
t_val = 0.25T₀
pos = SVector(x, y, z)

# Verify agreement before benchmarking
result_fe = fe([t_val, x, y, z])
lt_E = LaserTypes.E(pos, t_val, lt_laser)
lt_B = LaserTypes.B(pos, t_val, lt_laser)

println("=== Sanity check ===")
println("FieldEvaluator E: ", result_fe.E)
println("LaserTypes     E: ", lt_E)
println("FieldEvaluator B: ", result_fe.B)
println("LaserTypes     B: ", lt_B)
println()

# --- Benchmarks ---
txyz = SVector(t_val, x, y, z)

println("=== FieldEvaluator (E + B in one call) ===")
display(@benchmark $fe($txyz))
println()

println("=== LaserTypes E only ===")
display(@benchmark LaserTypes.E($pos, $t_val, $lt_laser))
println()

println("=== LaserTypes B only ===")
display(@benchmark LaserTypes.B($pos, $t_val, $lt_laser))
println()

println("=== LaserTypes E + B ===")
display(@benchmark begin
    LaserTypes.E($pos, $t_val, $lt_laser)
    LaserTypes.B($pos, $t_val, $lt_laser)
end)
println()
