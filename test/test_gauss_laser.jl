using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using OrdinaryDiffEqNonlinearSolve
using Test
using SciMLBase
using LaserTypes: LaserTypes
using LinearAlgebra
using Random

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

@testset "GaussLaser vs LaserTypes" begin
    # Compare against LaserTypes.GaussLaser (independent reference implementation)
    # LaserTypes uses ConstantProfile (no temporal envelope), so we divide out
    # our Gaussian envelope to compare.
    c = 137.03599908330932
    ω = 0.057
    T₀ = 2π / ω
    λ_val = c * T₀
    w₀_val = 75 * λ_val
    a₀_val = 2.0

    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val, ref_frame)

    fe = FieldEvaluator(laser, ref_frame)
    sys = fe.prob.f.sys
    τ0 = fe.prob.ps[sys.laser.τ0]
    T0_val = fe.prob.ps[sys.laser.T0]
    t₀ = 5 * T0_val

    # LaserTypes reference (constant temporal profile)
    lt_gauss = LaserTypes.GaussLaser(:atomic; λ=λ_val, a₀=a₀_val, w₀=w₀_val)

    # Test at 50 random points in the beam volume
    rng = Xoshiro(123)
    for _ in 1:50
        x = w₀_val * 0.5 * (2rand(rng) - 1)
        y = w₀_val * 0.5 * (2rand(rng) - 1)
        z = w₀_val * 0.5 * (2rand(rng) - 1)
        t_offset = T₀ * rand(rng)  # offset from pulse center

        t_eval = t₀ + t_offset
        result = fe([t_eval, x, y, z])

        # Our envelope at this point
        env = exp(-((t_offset - z / c) / τ0)^2)

        # LaserTypes at same spacetime point (no envelope)
        lt_E = LaserTypes.E([x, y, z], t_eval, lt_gauss)
        lt_B = LaserTypes.B([x, y, z], t_eval, lt_gauss)

        # Our field = LaserTypes field × envelope
        for i in 1:3
            if abs(lt_E[i]) > 1e-10
                @test isapprox(result.E[i], lt_E[i] * env, rtol=1e-7)
            end
            if abs(lt_B[i]) > 1e-10
                @test isapprox(result.B[i], lt_B[i] * env, rtol=1e-7)
            end
        end
    end
end

@testset "GaussLaser matches LG(0,0)" begin
    # LaguerreGauss with p=0, m=0 should produce identical fields to GaussLaser
    # at any spacetime point (with matched temporal envelope).
    c = 137.03599908330932
    ω = 0.057
    T₀ = 2π / ω
    λ_val = c * T₀
    w₀_val = 75 * λ_val
    a₀_val = 2.0

    @named ref_frame = ProperFrame(:atomic)

    @named gauss = GaussLaser(; wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val, ref_frame)
    fe_gauss = FieldEvaluator(gauss, ref_frame)
    sys_g = fe_gauss.prob.f.sys
    T0_val = fe_gauss.prob.ps[sys_g.gauss.T0]
    τ0_val = fe_gauss.prob.ps[sys_g.gauss.τ0]
    t₀_gauss = 5 * T0_val

    # LG(0,0) with Gaussian envelope matched to GaussLaser τ0
    @named lg = LaguerreGaussLaser(; wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val,
        radial_index=0, azimuthal_index=0, ref_frame, temporal_profile=:gaussian,
        temporal_width=τ0_val)
    fe_lg = FieldEvaluator(lg, ref_frame)

    # Evaluate both at the same spacetime points.
    # The envelope centers differ (t₀=5T0 vs t₀=0), so we compare
    # the envelope-free ratio: field / env should match for both.
    rng = Xoshiro(42)
    for _ in 1:50
        x = w₀_val * 0.3 * (2rand(rng) - 1)
        y = w₀_val * 0.3 * (2rand(rng) - 1)
        z = w₀_val * 0.1 * (2rand(rng) - 1)
        t_eval = T₀ * rand(rng)  # arbitrary time

        result_g = fe_gauss([t_eval, x, y, z])
        result_lg = fe_lg([t_eval, x, y, z])

        # Envelope ratio between the two
        env_gauss = exp(-((t_eval - t₀_gauss - z / c) / τ0_val)^2)
        env_lg = exp(-((t_eval - 0 - z / c) / τ0_val)^2)

        for i in 1:3
            # Compare field/envelope (the envelope-free field should match)
            if abs(result_lg.E[i]) > 1e-15 && env_lg > 1e-10 && env_gauss > 1e-10
                @test isapprox(result_g.E[i] / env_gauss, result_lg.E[i] / env_lg, rtol=1e-8)
            end
            if abs(result_lg.B[i]) > 1e-15 && env_lg > 1e-10 && env_gauss > 1e-10
                @test isapprox(result_g.B[i] / env_gauss, result_lg.B[i] / env_lg, rtol=1e-8)
            end
        end
    end
end

@testset "GaussLaser Gaussian radial profile" begin
    # At z=0, the transverse field amplitude should follow exp(-r²/w₀²)
    c = 137.03599908330932
    ω = 0.057
    λ_val = 2π * c / ω
    w₀_val = 75 * λ_val

    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=1.0, beam_waist=w₀_val, ref_frame)

    fe = FieldEvaluator(laser, ref_frame)
    sys = fe.prob.f.sys
    T0_val = fe.prob.ps[sys.laser.T0]
    t₀ = 5 * T0_val  # pulse center

    # At z=0, t=t₀: Ex = E₀ * exp(-r²/w₀²) * cos(ω*t₀)
    # So |Ex(r)| / |Ex(0)| = exp(-r²/w₀²)
    E_at_origin = fe([t₀, 0.0, 0.0, 0.0]).E[1]

    for r_frac in [0.1, 0.3, 0.5, 0.7, 1.0]
        r = r_frac * w₀_val
        E_at_r = fe([t₀, r, 0.0, 0.0]).E[1]
        expected_ratio = exp(-r_frac^2)
        @test E_at_r / E_at_origin ≈ expected_ratio rtol=1e-10
    end
end

@testset "GaussLaser beam width vs z" begin
    # Verify that on-axis (r=0) amplitude scales as w₀/w(z) using LaserTypes
    # as reference (constant profile avoids envelope complications).
    c = 137.03599908330932
    ω = 0.057
    λ_val = 2π * c / ω
    w₀_val = 75 * λ_val

    lt_gauss = LaserTypes.GaussLaser(:atomic; λ=λ_val, a₀=1.0, w₀=w₀_val)
    z_R = π * w₀_val^2 / λ_val

    # At t=0 along z-axis: compare Ex(0,0,z) / Ex(0,0,0)
    Ex_origin = LaserTypes.E([0.0, 0.0, 0.0], 0.0, lt_gauss)[1]

    for z_frac in [0.01, 0.1, 0.5, 1.0, 2.0]
        z = z_frac * z_R
        Ex_at_z = LaserTypes.E([0.0, 0.0, z], 0.0, lt_gauss)[1]
        wz = w₀_val * √(1 + z_frac^2)

        # The amplitude scales as w₀/w(z), but the phase also changes (Gouy + propagation).
        # Compare |Ex|² over a full cycle to isolate the amplitude factor.
        T_period = 2π / ω
        sum_sq_origin = sum(0:499) do i
            t = i * T_period / 500
            LaserTypes.E([0.0, 0.0, 0.0], t, lt_gauss)[1]^2
        end
        sum_sq_z = sum(0:499) do i
            t = i * T_period / 500
            LaserTypes.E([0.0, 0.0, z], t, lt_gauss)[1]^2
        end

        # RMS ratio should equal amplitude ratio = w₀/w(z)
        expected_ratio = w₀_val / wz
        @test √(sum_sq_z / sum_sq_origin) ≈ expected_ratio rtol=1e-4
    end

    # Now verify our FieldEvaluator matches LaserTypes at these same points
    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=1.0, beam_waist=w₀_val, ref_frame)
    fe = FieldEvaluator(laser, ref_frame)
    sys = fe.prob.f.sys
    T0_val = fe.prob.ps[sys.laser.T0]
    τ0 = fe.prob.ps[sys.laser.τ0]
    t₀ = 5 * T0_val

    for z_frac in [0.01, 0.5, 2.0]
        z = z_frac * z_R
        # At envelope peak t = t₀ + z/c, compare with LaserTypes at same time
        t_eval = t₀ + z / c
        result = fe([t_eval, 0.0, 0.0, z])
        lt_E = LaserTypes.E([0.0, 0.0, z], t_eval, lt_gauss)
        # env = 1 at this point (envelope peak)
        @test result.E[1] ≈ lt_E[1] rtol=1e-7
    end
end

@testset "GaussLaser paraxial relations" begin
    # For a paraxial beam propagating in +z:
    # Bx ≈ -Ey/c, By ≈ Ex/c (to leading order)
    # This is already explicit in the equations, but verify numerically
    # at off-axis points where all components are nonzero.
    c = 137.03599908330932
    ω = 0.057
    λ_val = 2π * c / ω
    w₀_val = 75 * λ_val

    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=2.0, beam_waist=w₀_val, ref_frame)

    fe = FieldEvaluator(laser, ref_frame)
    sys = fe.prob.f.sys
    T0_val = fe.prob.ps[sys.laser.T0]
    t₀ = 5 * T0_val

    rng = Xoshiro(99)
    for _ in 1:20
        x = w₀_val * 0.3 * (2rand(rng) - 1)
        y = w₀_val * 0.3 * (2rand(rng) - 1)
        z = w₀_val * 0.1 * (2rand(rng) - 1)
        t = t₀ + 2π / ω * rand(rng)

        result = fe([t, x, y, z])

        # Paraxial relations (exact in our model)
        @test result.B[1] ≈ -result.E[2] / c rtol=1e-12
        @test result.B[2] ≈ result.E[1] / c rtol=1e-12
    end
end

@testset "GaussLaser circular polarization" begin
    # For circular polarization, ξx = 1/√2, ξy = i/√2
    # |Ex|² + |Ey|² should be independent of azimuthal angle at fixed r, z, t
    c = 137.03599908330932
    ω = 0.057
    λ_val = 2π * c / ω
    w₀_val = 75 * λ_val

    @named ref_frame = ProperFrame(:atomic)
    @named laser_circ = GaussLaser(; wavelength=λ_val, a0=1.0, beam_waist=w₀_val,
        polarization=:circular, ref_frame)

    fe = FieldEvaluator(laser_circ, ref_frame)
    sys = fe.prob.f.sys
    T0_val = fe.prob.ps[sys.laser_circ.T0]
    t₀ = 5 * T0_val

    r = 0.1 * w₀_val
    # Evaluate at several azimuthal angles; transverse intensity should be constant
    intensities = Float64[]
    for θ in range(0, 2π, length=17)[1:end-1]  # 16 angles
        x = r * cos(θ)
        y = r * sin(θ)
        result = fe([t₀, x, y, 0.0])
        push!(intensities, result.E[1]^2 + result.E[2]^2)
    end

    # All transverse intensities should be equal
    @test all(I -> isapprox(I, intensities[1], rtol=1e-10), intensities)
end

@testset "GaussLaser derived parameters" begin
    # Verify that all derived parameters are computed correctly from the inputs
    c = 137.03599908330932
    m_e = 1.0
    q_e = -1.0

    λ_val = 2.0
    a₀_val = 3.0
    w₀_val = 100.0

    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val, ref_frame)

    fe = FieldEvaluator(laser, ref_frame)
    sys = fe.prob.f.sys

    ω_expected = 2π * c / λ_val
    k_expected = 2π / λ_val
    z_R_expected = w₀_val^2 * k_expected / 2
    E₀_expected = a₀_val * m_e * c * ω_expected / abs(q_e)
    τ0_expected = 10 / ω_expected

    @test fe.prob.ps[sys.laser.ω] ≈ ω_expected rtol=1e-10
    @test fe.prob.ps[sys.laser.k] ≈ k_expected rtol=1e-10
    @test fe.prob.ps[sys.laser.z_R] ≈ z_R_expected rtol=1e-10
    @test fe.prob.ps[sys.laser.E₀] ≈ E₀_expected rtol=1e-10
    @test fe.prob.ps[sys.laser.τ0] ≈ τ0_expected rtol=1e-10
    @test fe.prob.ps[sys.laser.w₀] ≈ w₀_val rtol=1e-10
    @test fe.prob.ps[sys.laser.λ] ≈ λ_val rtol=1e-10

    # Also check z_R = π w₀² / λ (equivalent formula)
    @test z_R_expected ≈ π * w₀_val^2 / λ_val rtol=1e-10
end

@testset "GaussLaser temporal envelope" begin
    # Verify the Gaussian temporal envelope by comparing our pulsed field
    # against LaserTypes (CW, no envelope) at the same spacetime point.
    # The ratio should be exactly env(t) = exp(-((t-t₀)/τ0)²) at z=0.
    c = 137.03599908330932
    ω = 0.057
    λ_val = 2π * c / ω
    w₀_val = 75 * λ_val

    @named ref_frame = ProperFrame(:atomic)
    @named laser = GaussLaser(; wavelength=λ_val, a0=1.0, beam_waist=w₀_val, ref_frame)

    fe = FieldEvaluator(laser, ref_frame)
    sys = fe.prob.f.sys
    T0_val = fe.prob.ps[sys.laser.T0]
    τ0 = fe.prob.ps[sys.laser.τ0]
    t₀ = 5 * T0_val

    lt_gauss = LaserTypes.GaussLaser(:atomic; λ=λ_val, a₀=1.0, w₀=w₀_val)

    # At z=0, r=0.1w₀: our_Ex(t) = LaserTypes_Ex(t) * exp(-((t-t₀)/τ0)²)
    x_test = 0.1 * w₀_val
    for Δt in [0.0, 0.5τ0, τ0, 1.5τ0, 2τ0]
        t_eval = t₀ + Δt
        result = fe([t_eval, x_test, 0.0, 0.0])
        lt_E = LaserTypes.E([x_test, 0.0, 0.0], t_eval, lt_gauss)
        env_expected = exp(-(Δt / τ0)^2)

        @test result.E[1] ≈ lt_E[1] * env_expected rtol=1e-7
    end
end
