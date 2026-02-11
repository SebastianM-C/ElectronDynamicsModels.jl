using Test
using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using LaserTypes: LaserTypes
using HypergeometricFunctions: pochhammer
using LinearAlgebra
using Random
using CSV

@testset "LaguerreGaussLaser vs LaserTypes" begin
    # Common parameters (SI units)
    λ_test = 800e-9      # wavelength in meters
    a₀_test = 1.0        # normalized amplitude
    w₀_test = 10e-6      # beam waist
    c_SI = 299792458.0

    # Test multiple (p, m) combinations
    pm_combos = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 1)]

    test_points = [
        (r=1e-6, θ=π/4, z=0.0),
        (r=5e-6, θ=π/2, z=10e-6),
        (r=2e-6, θ=π, z=-5e-6),
        (r=3e-6, θ=0.0, z=5e-6),
    ]

    @testset "p=$p_val, m=$m_val" for (p_val, m_val) in pm_combos
        # LaserTypes reference
        lt_laser = LaserTypes.LaguerreGaussLaser(:SI;
            λ=λ_test, a₀=a₀_test, w₀=w₀_test, p=p_val, m=m_val)

        # ElectronDynamicsModels
        @named ref_frame = ProperFrame(:SI)
        @named laser = LaguerreGaussLaser(
            wavelength=λ_test, a0=a₀_test, beam_waist=w₀_test,
            radial_index=p_val, azimuthal_index=m_val,
            ref_frame=ref_frame, temporal_profile=:constant)

        fe = FieldEvaluator(laser, ref_frame)

        for pt in test_points
            x = pt.r * cos(pt.θ); y = pt.r * sin(pt.θ); z = pt.z
            result = fe([0.0, x, y, z])
            lt_E = LaserTypes.E([x, y, z], 0.0, lt_laser)
            lt_B = LaserTypes.B([x, y, z], 0.0, lt_laser)

            for i in 1:3
                if abs(lt_E[i]) > 1e-10
                    @test isapprox(result.E[i], lt_E[i], rtol=1e-7)
                end
                if abs(lt_B[i]) > 1e-10
                    @test isapprox(result.B[i], lt_B[i], rtol=1e-7)
                end
            end
        end
    end
end

@testset "Origin symmetry" begin
    # Port of LaserTypes origin.jl
    # At origin, for (p=1, m=1) with constant profile:
    # All E components = 0
    # Bx = 0, By = 0
    # Bz has an analytical value

    c = 137.03599908330932  # atomic units
    m_e = 1.0
    q_e = -1.0

    @named ref_frame = ProperFrame(:atomic)
    @named laser = LaguerreGaussLaser(
        wavelength=1.0, a0=1.0, beam_waist=nothing,
        radial_index=1, azimuthal_index=1,
        ref_frame=ref_frame, temporal_profile=:constant)

    fe = FieldEvaluator(laser, ref_frame)

    # Extract parameters
    sys = fe.prob.f.sys
    E₀ = fe.prob.ps[sys.laser.E₀]
    Nₚₘ = fe.prob.ps[sys.laser.Nₚₘ]
    w₀ = fe.prob.ps[sys.laser.w₀]
    z_R = fe.prob.ps[sys.laser.z_R]

    result = fe([0.0, 0.0, 0.0, 0.0])

    # All E components are zero at origin
    @test iszero(result.E[1])
    @test iszero(result.E[2])
    @test iszero(result.E[3])

    # Bx = 0, By = 0
    @test iszero(result.B[1])
    @test iszero(result.B[2])

    # Bz analytical value: -factorial(p)/pochhammer(|m|+1, p) * E₀ * Nₚₘ * (√2 * w₀) / (c * z_R)
    p_val = 1; m_val = 1
    Bz_expected = -factorial(p_val) / pochhammer(abs(m_val) + 1, p_val) * E₀ * Nₚₘ * (√2 * w₀) / (c * z_R)
    @test result.B[3] ≈ Bz_expected
end

@testset "Gaussian limit" begin
    # Port of LaserTypes gauss.jl
    # LG(p=0, m=0) with constant profile should match LaserTypes GaussLaser
    # at the same spacetime point. Both use ConstantProfile (no temporal envelope).

    c = 137.03599908330932
    ω = 0.057
    T₀ = 2π / ω
    λ_val = c * T₀
    w₀_val = 75 * λ_val
    a₀_val = 2.0

    @named ref_frame = ProperFrame(:atomic)
    @named lg_laser = LaguerreGaussLaser(
        wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val,
        radial_index=0, azimuthal_index=0,
        ref_frame=ref_frame, temporal_profile=:constant)

    fe_lg = FieldEvaluator(lg_laser, ref_frame)

    # LaserTypes GaussLaser with matching parameters (constant profile)
    lt_gauss = LaserTypes.GaussLaser(:atomic; λ=λ_val, a₀=a₀_val, w₀=w₀_val)

    # 100 random points in the beam volume
    rng = Xoshiro(42)  # reproducible
    for _ in 1:100
        x = w₀_val * (0.5 - rand(rng))
        y = w₀_val * (0.5 - rand(rng))
        z = w₀_val * (0.5 - rand(rng))
        t_val = T₀ * rand(rng)

        result_lg = fe_lg([t_val, x, y, z])

        lt_E = LaserTypes.E([x, y, z], t_val, lt_gauss)
        lt_B = LaserTypes.B([x, y, z], t_val, lt_gauss)

        for i in 1:3
            if abs(lt_E[i]) > 1e-10
                @test isapprox(result_lg.E[i], lt_E[i], rtol=1e-8)
            end
            if abs(lt_B[i]) > 1e-10
                @test isapprox(result_lg.B[i], lt_B[i], rtol=1e-8)
            end
        end
    end
end

@testset "Field stability" begin
    # Port of LaserTypes roots.jl
    # Check that field energy density stays bounded
    c = 137.03599908330932
    ω = 0.057
    T₀ = 2π / ω
    λ_val = c * T₀
    w₀_val = 75 * λ_val
    a₀_val = 2.0

    @named ref_frame = ProperFrame(:atomic)
    @named laser = LaguerreGaussLaser(
        wavelength=λ_val, a0=a₀_val, beam_waist=w₀_val,
        radial_index=1, azimuthal_index=1,
        ref_frame=ref_frame, temporal_profile=:constant)

    fe = FieldEvaluator(laser, ref_frame)

    # Evaluate energy density |E|² + c²|B|² over a grid at z=0, t=0
    # Use coarser grid than LaserTypes (they use step=λ over [-4w₀, 4w₀])
    step = 10λ_val
    domain = -4w₀_val:step:4w₀_val

    wMax = 0.0
    for x in domain, y in domain
        result = fe([0.0, x, y, 0.0])
        w = dot(result.E, result.E) + c^2 * dot(result.B, result.B)
        wMax = max(wMax, w)
    end

    @test wMax < 1e6
end

@testset "Field values vs reference" begin
    # Port of LaserTypes field_values.jl
    # Compare against Mathematica reference data (CSV files)
    c = 137.03599908330932
    ω = 0.057
    λ_val = 2π * c / ω
    w₀_val = 75 * λ_val

    csv_folder = joinpath(Base.pkgdir(LaserTypes), "test", "Laguerre-Gauss", "fields_csv")

    # Only test i=0 files (linear polarization, matching our :linear kwarg)
    i_filter = 0

    @testset "$file" for file in readdir(csv_folder)
        # Parse i, p, m from filename: field_i_p_m_.csv
        parts = split(file, "_")
        length(parts) >= 4 || continue
        i_val = tryparse(Int, parts[2])
        p_val = tryparse(Int, parts[3])
        m_val = tryparse(Int, parts[4])
        (i_val === nothing || p_val === nothing || m_val === nothing) && continue
        i_val != i_filter && continue

        @named ref_frame = ProperFrame(:atomic)
        @named laser = LaguerreGaussLaser(
            wavelength=λ_val, a0=1.0, beam_waist=w₀_val,
            radial_index=p_val, azimuthal_index=m_val,
            ref_frame=ref_frame, temporal_profile=:constant)

        fe = FieldEvaluator(laser, ref_frame)

        # Extract normalization parameters
        sys = fe.prob.f.sys
        E₀ = fe.prob.ps[sys.laser.E₀]
        Nₚₘ = fe.prob.ps[sys.laser.Nₚₘ]
        norm = factorial(p_val) / pochhammer(abs(m_val) + 1, p_val) * E₀ * Nₚₘ

        for row in CSV.File(joinpath(csv_folder, file), header=false)
            x = row[1]
            y = row[2]
            z = row[3]

            # Parse complex values from Mathematica format
            parse_mma(s) = tryparse(ComplexF64, replace(replace(s, "*I" => "im"), "*^" => "e"))

            EX = parse_mma(row[4])
            EY = parse_mma(row[5])
            EZ = parse_mma(row[6])
            BX = parse_mma(row[7])
            BY = parse_mma(row[8])
            BZ = parse_mma(row[9])

            # Evaluate field at t=0 (CSV data is at t=0)
            result = fe([0.0, x, y, z])

            # Our model computes real(complex_expr), so compare against real part.
            # Use rtol=1e-6 to account for numerical differences between
            # Mathematica reference data and our compiled symbolic evaluation.
            @test result.E[1] ≈ real(norm * EX) rtol=1e-6
            @test result.E[2] ≈ real(norm * EY) rtol=1e-6
            @test result.E[3] ≈ real(norm * EZ) rtol=1e-6
            @test result.B[1] ≈ real(norm * BX) rtol=1e-6
            @test result.B[2] ≈ real(norm * BY) rtol=1e-6
            @test result.B[3] ≈ real(norm * BZ) rtol=1e-6
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
        a0=a₀,
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
