using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using OrdinaryDiffEqNonlinearSolve
using Test
using SciMLBase

@testset "GaussLaser wavelength/frequency interface" begin
    c = 137.03599908330932  # speed of light in atomic units

    @named ref_frame = ProperFrame(:atomic)

    # Choose a test wavelength and compute the corresponding frequency
    λ_val = 1.0
    ω_val = 2π * c / λ_val

    # Build system using wavelength kwarg
    @named laser_λ = GaussLaser(; wavelength=λ_val, a0=1.0, ref_frame)
    @named elec_λ = ClassicalElectron(; laser=laser_λ, ref_frame)
    sys_λ = mtkcompile(elec_λ)

    # Build system using frequency kwarg
    @named laser_ω = GaussLaser(; frequency=ω_val, a0=1.0, ref_frame)
    @named elec_ω = ClassicalElectron(; laser=laser_ω, ref_frame)
    sys_ω = mtkcompile(elec_ω)

    # Same initial conditions
    tspan = (0.0, 1.0)
    u0_λ = [sys_λ.x => [0.0, 0.0, 0.0, 0.0], sys_λ.u => [c, 0.0, 0.0, 0.0]]
    u0_ω = [sys_ω.x => [0.0, 0.0, 0.0, 0.0], sys_ω.u => [c, 0.0, 0.0, 0.0]]

    prob_λ = ODEProblem(sys_λ, u0_λ, tspan)
    prob_ω = ODEProblem(sys_ω, u0_ω, tspan)

    # Verify that the derived parameters are consistent
    @test prob_λ.ps[sys_λ.laser_λ.ω] ≈ ω_val
    @test prob_ω.ps[sys_ω.laser_ω.λ] ≈ λ_val

    # Solve both and compare trajectories
    sol_λ = solve(prob_λ, Vern9(), reltol=1e-12, abstol=1e-12)
    sol_ω = solve(prob_ω, Vern9(), reltol=1e-12, abstol=1e-12)

    @test SciMLBase.successful_retcode(sol_λ)
    @test SciMLBase.successful_retcode(sol_ω)

    # Both should produce the same final state
    for i in 1:4
        @test sol_λ[sys_λ.x[i], end] ≈ sol_ω[sys_ω.x[i], end] rtol=1e-8
        @test sol_λ[sys_λ.u[i], end] ≈ sol_ω[sys_ω.u[i], end] rtol=1e-8
    end

    # Test error when both are specified
    @test_throws ErrorException GaussLaser(; wavelength=1.0, frequency=1.0, ref_frame, name=:bad)
end

@testset "GaussLaser field values at origin" begin
    # Atomic units
    c = 137.03599908330932
    m_e = 1.0
    q_e = -1.0

    λ_val = 1.0
    a0_val = 1.0
    ω_val = 2π * c / λ_val

    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=a0_val, ref_frame)

    fe = FieldEvaluator(laser, ref_frame)

    # Read initialized parameters from the problem
    E₀ = fe.prob.ps[fe.prob.f.sys.laser.E₀]
    τ0 = fe.prob.ps[fe.prob.f.sys.laser.τ0]
    T0_val = fe.prob.ps[fe.prob.f.sys.laser.T0]

    # Verify E₀ is computed correctly: a₀ * m_e * c * ω / |q_e|
    E₀_expected = a0_val * m_e * c * ω_val / abs(q_e)
    @test E₀ ≈ E₀_expected rtol=1e-10

    # Evaluate at origin, at the pulse center time t₀ = 5*T0
    t₀ = 5 * T0_val

    result = fe([t₀, 0.0, 0.0, 0.0])

    # At origin (r=0, z=0):
    # - wz = w₀ (beam width at z=0)
    # - Gaussian beam factor: w₀/wz * exp(-r²/wz²) = 1
    # - Spatial phase: atan(0, z_R) - k*0 = 0
    # - Temporal envelope: exp(-((t₀ - t₀)/τ0)²) = 1
    # So E[1] = E₀ * cos(ω*t₀) for linear polarization (ξx=1, ξy=0)
    phase = ω_val * t₀
    E_expected = E₀ * cos(phase)

    @test result.E[1] ≈ E_expected rtol=1e-10
    @test result.E[2] ≈ 0 atol=1e-10
    @test result.E[3] ≈ 0 atol=1e-10

    # B field: B[1] = -E[2]/c = 0, B[2] = E[1]/c, B[3] depends on Ez (0 at origin)
    @test result.B[1] ≈ 0 atol=1e-10
    @test result.B[2] ≈ E_expected / c rtol=1e-10
    @test result.B[3] ≈ 0 atol=1e-10
end