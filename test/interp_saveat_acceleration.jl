# With `saveat`, the ODE solution is NOT dense: `sol(t, Val{1})` silently falls back to the
# stored-value linear interpolation and returns left-segment slopes — i.e. acceleration
# delayed by half a saveat knot. Fed into TrajectoryInterpolant's a_itp, that delay stamped
# a −n·π/16 global phase on every EDM_INTERP_SAVEAT-era harmonic map (root-caused
# 2026-07-24). The fix differentiates the trajectory's own cubic spline on non-dense
# solutions; this test pins it by demanding the saveat-built acceleration match the dense
# solution's model-consistent derivative inside the pulse.
using Test
using ElectronDynamicsModels
using ModelingToolkit, OrdinaryDiffEqVerner, SciMLBase, StaticArrays, LinearAlgebra
using SymbolicIndexingInterface: variable_index
import ElectronDynamicsModels as EDM

let
    c = 137.03599908330932
    ω = 0.057
    τp = 150 / ω
    λ = 2π * c / ω
    w₀ = 75λ
    τi, τf = -8τp, 8τp
    @named world = Worldline(:τ, :atomic)
    @named laser = LaguerreGaussLaser(;
        wavelength = λ, a0 = 0.1, beam_waist = w₀,
        radial_index = 2, azimuthal_index = -2, world, temporal_profile = :gaussian,
        temporal_width = τp, focus_position = 0.0, polarization = :circular_minus,
        initial_phase = -π / 2)
    @named elec = ClassicalElectron(; laser)
    sys = mtkcompile(elec)
    u0 = [sys.x => SVector{4}(τi * c, 1.2w₀, 0.7w₀, 0.0),
        sys.u => SVector{4}(c, 0.0, 0.0, 0.0)]
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        sys, u0, (τi, τf), u0_constructor = SVector{8}, fully_determined = true)

    sol_dense = solve(prob, Vern9(); reltol = 1.0e-12, abstol = 1.0e-13)
    sol_sv = solve(prob, Vern9(); reltol = 1.0e-12, abstol = 1.0e-13,
        saveat = collect(τi:((2π / ω) / 16):τf))

    tr_sv = EDM.TrajectoryInterpolant(sol_sv, sys.x, sys.u)
    uidx = SVector{4, Int}(variable_index.((sol_dense,), collect(sys.u)))

    # Probe between knots across the pulse core. The pre-fix half-knot delay shifts the
    # ω-oscillating acceleration by π/16 ⇒ ~20% pointwise error; the spline derivative is
    # accurate to ≪1% at 16 knots/period.
    worst = maximum(range(-τp, τp; length = 41)) do τ
        a_ref = sol_dense(τ, Val{1})[uidx]
        a_sv = EDM.state_with_acceleration(tr_sv, τ)[3]
        norm(a_sv - a_ref) / max(norm(a_ref), eps())
    end
    @test worst < 0.02
end
