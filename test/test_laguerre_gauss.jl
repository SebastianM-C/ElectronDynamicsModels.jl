using Test
using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using LaserTypes: LaserTypes

@testset "LaguerreGaussLaser vs LaserTypes" begin
    # Common parameters (SI units)
    λ_test = 800e-9      # wavelength in meters
    a₀_test = 1.0        # normalized amplitude
    w₀_test = 10e-6      # beam waist
    p_test = 0           # radial index
    m_test = 1           # azimuthal index
    c_test = 299792458.0 # speed of light

    # Test coordinates
    test_points = [
        (r=0.0, θ=0.0, z=0.0),
        (r=1e-6, θ=π/4, z=0.0),
        (r=5e-6, θ=π/2, z=10e-6),
        (r=2e-6, θ=π, z=-5e-6),
    ]

    # Create LaserTypes laser (units: :SI for SI units)
    lt_laser = LaserTypes.LaguerreGaussLaser(:SI;
        λ=λ_test,
        a₀=a₀_test,
        w₀=w₀_test,
        p=p_test,
        m=m_test
    )

    # Setup MTK system with matching parameters
    @named ref_frame = ProperFrame(:SI)
    @named laser = LaguerreGaussLaser(
        wavelength=λ_test,
        amplitude=a₀_test,
        beam_waist=w₀_test,
        radial_index=p_test,
        azimuthal_index=m_test,
        ref_frame=ref_frame,
        temporal_profile=:constant  # Match LaserTypes ConstantProfile
    )

    # Create electron system
    @named lg_elec = ClassicalElectron(; laser, ref_frame)

    # Compile the system
    sys = mtkcompile(lg_elec)

    # Initial conditions: electron at rest at origin
    γ₀ = 1.0
    x₀ = [0.0, 0.0, 0.0, 0.0]
    p₀ = [0.0, 0.0, 0.0, γ₀]

    u0 = [
        sys.x => x₀,
        sys.u => p₀
    ]

    tspan = (0.0, 1e-15)  # 1 femtosecond
    prob = ODEProblem(sys, u0, tspan)

    # Test all points
    for (i, point) in enumerate(test_points)
        r_val, θ_val, z_val = point.r, point.θ, point.z

        # Convert from cylindrical to Cartesian
        x_val = r_val * cos(θ_val)
        y_val = r_val * sin(θ_val)

        # LaserTypes fields
        lt_E = LaserTypes.E([x_val, y_val, z_val], 0.0, lt_laser)
        lt_B = LaserTypes.B([x_val, y_val, z_val], 0.0, lt_laser)

        # ElectronDynamicsModels fields
        local x₀ = [0.0, x_val, y_val, z_val]
        local u0 = [sys.x => x₀, sys.u => p₀]
        prob_i = ODEProblem(sys, u0, tspan)
        sol = solve(prob_i, Vern9())

        edm_Ex = sol[sys.laser.E[1]][1]
        edm_Ey = sol[sys.laser.E[2]][1]
        edm_Ez = sol[sys.laser.E[3]][1]
        edm_Bx = sol[sys.laser.B[1]][1]
        edm_By = sol[sys.laser.B[2]][1]
        edm_Bz = sol[sys.laser.B[3]][1]

        # Test Ex, Ez, Bz (main field components)
        # Allow for numerical precision (relative tolerance ~1e-8)
        if abs(lt_E[1]) > 1e-10  # Skip if value is effectively zero
            @test isapprox(edm_Ex, lt_E[1], rtol=1e-7)
        end

        if abs(lt_E[3]) > 1e-10
            @test isapprox(edm_Ez, lt_E[3], rtol=1e-7)
        end

        if abs(lt_B[3]) > 1e-10
            @test isapprox(edm_Bz, lt_B[3], rtol=1e-7)
        end
    end
end

@testset "LaguerreGaussLaser Parameters" begin
    # Test parameter computation
    λ = 800e-9
    a₀ = 1.0
    w₀ = 10e-6
    c = 299792458.0

    @named ref_frame = ProperFrame(:SI)
    @named laser = LaguerreGaussLaser(
        wavelength=λ,
        amplitude=a₀,
        beam_waist=w₀,
        radial_index=0,
        azimuthal_index=1,
        ref_frame=ref_frame
    )
    @named elec = ClassicalElectron(; laser, ref_frame)
    sys = mtkcompile(elec)

    u0 = [sys.x => [0.0, 0.0, 0.0, 0.0], sys.u => [0.0, 0.0, 0.0, 1.0]]
    prob = ODEProblem(sys, u0, (0.0, 1e-15))

    # Check derived parameters
    ω_expected = 2π * c / λ
    k_expected = 2π / λ
    z_R_expected = π * w₀^2 / λ
    # Use the CODATA 2018 value that matches src/base.jl
    m_e_SI = 9.1093837015e-31  # kg - electron mass (CODATA 2018)
    q_e_SI = 1.602176634e-19   # C - electron charge magnitude
    E₀_expected = a₀ * m_e_SI * c * ω_expected / q_e_SI

    @test isapprox(prob.ps[sys.laser.ω], ω_expected, rtol=1e-10)
    @test isapprox(prob.ps[sys.laser.k], k_expected, rtol=1e-10)
    @test isapprox(prob.ps[sys.laser.z_R], z_R_expected, rtol=1e-10)
    @test isapprox(prob.ps[sys.laser.E₀], E₀_expected, rtol=1e-10)
    @test isapprox(prob.ps[sys.laser.Nₚₘ], 1.0, rtol=1e-10)  # For p=0, m=1
end
