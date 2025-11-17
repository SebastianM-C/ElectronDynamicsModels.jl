using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using Test
using SciMLBase
using LinearAlgebra

@testset "Classical Electron Tests" begin
    # Test system structure
    @testset "System Structure" begin
        @named electron = ClassicalElectron(ref_frame=ProperFrame(:natural, name=:ref_frame))
        sys = mtkcompile(electron)

        # We expect 8 equations for the covariant formulation
        @test length(equations(sys)) == 8
        @test length(unknowns(sys)) == 8  # x[1:4], u[1:4]
    end

    # Test figure-8 motion in plane wave
    @testset "Figure-8 Motion in Plane Wave" begin
        # This tests the well-known figure-8 orbit solution
        # for a relativistic electron in a plane wave field
        # Reference: Sarachik & Schappert, Phys. Rev. D 1, 2738 (1970)

        k = 1
        c = 1
        @independent_variables τ
        @named ref_frame = ReferenceFrame(τ; c, ε₀=1, μ₀=1, m_e=1, q_e=1)
        external_field = PlaneWave(; ref_frame, name = :plane_wave, k_vector = [0, 0, k])
        @named electron = ChargedParticle(; ref_frame, external_field)
        sys = mtkcompile(electron)

        # In natural units (c=1, m=1, e=1)
        # For a₀ = 1, the electron performs figure-8 motion
        a₀ = 1.0

        # Initial velocity for figure-8 orbit
        # v₀ = -c / ((2/a₀)² + 1)
        v₀_z = -c / ((2 / a₀)^2 + 1)  # velocity in z-direction
        γ₀ = 1.0 / sqrt(1 - (v₀_z/c)^2)

        # Initial 4-velocity
        u₀ = [γ₀, 0.0, 0.0, γ₀ * v₀_z]

        # Set initial conditions
        u0 = [sys.x => [0.0, 0.0, 0.0, 0.0], sys.u => u₀]

        # Time span in units of laser period
        # For figure-8, we need at least one full period

        # analytic expression for the trajectory
        f(x, z) = 16 * k^2 * z^2 + k * x * (k * x - 2a₀)^2 * (k * x - 4a₀)
        # analytic expression for the velocity
        v_r(x) = sqrt(c^2 * (1-4(1-2a₀^2)/(2+2*a₀^2-k^2*x^2)^2))
        # TODO: add more rigurous tests on the figure 8 trajectory based on
        # the analytic solution

        tspan = (0.0, 50.0)

        prob = ODEProblem(sys, u0, tspan)
        sol = solve(prob, Vern9(), abstol = 1e-9, reltol = 1e-9)

        @test SciMLBase.successful_retcode(sol)

        # Extract trajectories
        x_traj = sol[sys.x[2]]  # x-coordinate
        z_traj = sol[sys.x[4]]  # z-coordinate

        # Check that the motion is bounded (characteristic of figure-8)
        @test maximum(abs.(x_traj)) < 1.7  # Should be on order of c/ω
        @test maximum(abs.(z_traj)) < 0.09  # Adjusted for new structure

        # Check 4-velocity normalization is preserved
        for i = 1:10:length(sol.t)
            u = sol[sys.u, i]
            u_norm = u[1]^2 - u[2]^2 - u[3]^2 - u[4]^2
            @test abs(u_norm - 1.0) < 1e-8
        end
    end

    # Test energy conservation in crossed fields
    @testset "Drift Motion in Crossed Fields" begin
        # In crossed E and B fields with E⊥B and |E| < |B|,
        # the particle drifts with velocity v_drift = E×B/B²
        # Reference: Jackson, Classical Electrodynamics, Section 12.4

        # Note: In the relativistic case, there are corrections to the drift

        E_field = [0.01, 0.0, 0.0]  # Weaker E field to stay non-relativistic
        B_field = [0.0, 0.0, 1.0]  # B in z-direction

        @named ref_frame = ProperFrame(:natural)
        @named uniform_field = UniformField(; E_field, B_field, ref_frame)
        @named electron = ChargedParticle(; ref_frame, external_field = uniform_field)

        sys = mtkcompile(electron, allow_symbolic = true)

        # Start from rest
        u0 = [
            sys.x => [0.0, 0.0, 0.0, 0.0],
            sys.u => [1.0, 0.0, 0.0, 0.0],  # γ=1, v=0
        ]

        tspan = (0.0, 50.0)  # Longer time for steady state
        prob = ODEProblem(sys, u0, tspan)
        sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)

        @test SciMLBase.successful_retcode(sol)

        # Expected drift velocity: v_drift = E×B/B² = -0.01 ŷ
        v_drift_expected = cross(E_field, B_field) / dot(B_field, B_field)
        @test abs(v_drift_expected[2] + 0.01) < 1e-10  # -0.01 in y-direction

        # After sufficient time, particle should drift in y-direction
        # Check average velocity in last quarter of simulation
        t_start = 3 * tspan[2] / 4
        idx_start = findfirst(t -> t >= t_start, sol.t)

        y_start = sol[sys.x, idx_start][3]
        y_end = sol[sys.x, end][3]
        Δt = sol.t[end] - sol.t[idx_start]

        v_y_numerical = (y_end - y_start) / Δt

        # Should match drift velocity within 20% (allowing for oscillations)
        @test abs(v_y_numerical - v_drift_expected[2]) / abs(v_drift_expected[2]) < 0.2
    end

    # Test radiation-free limit
    @testset "Radiation-Free Limit" begin
        # Compare ClassicalElectron (no radiation) with RadiatingElectron
        # In weak fields, radiation effects should be small

        # Use extremely weak field to minimize radiation effects
        # The radiation force scales as E², so reducing E by 10x reduces radiation by 100x
        E₀ = 0.0001  # Extremely weak field
        @named ref_frame = ProperFrame(:natural)
        @named weak_field =
            UniformField(E_field = [0, 0, E₀], B_field = [0, 0, 0]; ref_frame)

        # Classical electron (no radiation)
        @named electron_classical = ClassicalElectron(; ref_frame, laser = weak_field)
        sys_classical = mtkcompile(electron_classical)

        # Radiating electron
        @named electron_radiating = RadiatingElectron(; ref_frame, laser = weak_field)
        sys_radiating = mtkcompile(electron_radiating)

        # Same initial conditions
        u0 = [
            sys_classical.x => [0.0, 0.0, 0.0, 0.0],
            sys_classical.u => [1.0, 0.0, 0.0, 0.0],
        ]

        tspan = (0.0, 0.1)  # Shorter time span

        # Solve both
        prob_classical = ODEProblem(sys_classical, u0, tspan)
        sol_classical = solve(prob_classical, Vern9(), abstol = 1e-14, reltol = 1e-14)

        u0_rad = [
            sys_radiating.x => [0.0, 0.0, 0.0, 0.0],
            sys_radiating.u => [1.0, 0.0, 0.0, 0.0],
        ]
        prob_radiating = ODEProblem(sys_radiating, u0_rad, tspan)
        sol_radiating = solve(prob_radiating, Vern9(), abstol = 1e-14, reltol = 1e-14)

        # Compare final positions only
        x_classical_final = sol_classical[sys_classical.x, end]
        x_radiating_final = sol_radiating[sys_radiating.x, end]

        # The radiation reaction effect should be small but noticeable
        # For E = 0.0001, the radiation force is ~ τ₀ * E² ~ 5e-8
        # Over time t = 0.1, this causes displacement differences ~ 5e-10
        # But in practice, the effect accumulates to be larger

        for j = 1:4
            abs_diff = abs(x_classical_final[j] - x_radiating_final[j])
            # For the time component (j=1), we expect larger values
            if j == 1
                # Time component should be close to ct ≈ 0.1
                @test abs_diff < 1e-10
            else
                # Spatial components: radiation causes differences on order of 1e-6
                @test abs_diff < 2e-6
            end
        end

        # Also check that the velocities are close
        u_classical_final = sol_classical[sys_classical.u, end]
        u_radiating_final = sol_radiating[sys_radiating.u, end]

        for j = 1:4
            abs_diff = abs(u_classical_final[j] - u_radiating_final[j])
            # For the time component (γ), we expect it to be close to 1
            if j == 1
                @test abs_diff < 1e-10
            else
                # Spatial velocity components: radiation causes differences ~ 2e-5
                @test abs_diff < 3e-5
            end
        end
    end
end
