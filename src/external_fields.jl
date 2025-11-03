"""
Gaussian laser pulse electromagnetic field.

Represents a focused Gaussian beam with a temporal envelope.
The beam propagates along the z-direction with waist w₀ at z=0.

Parameters:
- λ: wavelength
- a₀: normalized vector potential amplitude
- w₀: beam waist (defaults to 75λ)
- T0: pulse duration parameter
- τ0: temporal width parameter
"""
@component function GaussLaser(; name, wavelength=1.0, amplitude=10.0, beam_waist=nothing, spacetime=Spacetime(c=1, name=:spacetime))
    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; spacetime)
    
    # Get spacetime variables from parent scope
    @unpack c = spacetime
    τ = ParentScope(spacetime.τ)
    
    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)
    
    E, B = field_dynamics.E, field_dynamics.B
    
    @parameters begin
        λ = wavelength
        a₀ = amplitude
        m = 1.0  # electron mass
        q = -1.0  # electron charge
        ω
        k
        E₀
        w₀ = beam_waist === nothing ? 75λ : beam_waist
        z_R
        T0 = 100
        τ0
    end
    
    # Fixed parameters
    ξx = 1.0 + 0im
    ξy = 0
    ϕ₀ = 0
    t₀ = 5T0
    z₀ = 0
    
    @variables wz(τ) z(τ) r(τ)
    
    eqs = [
        # Extract z-coordinate from 4-position
        z ~ x[4]
        # Beam width as function of z
        wz ~ w₀ * √(1 + (z / z_R)^2)
        # Radial distance from beam axis
        r ~ hypot(x[2], x[3])
        
        # Electric field components
        E[1] ~ real(
            E₀ * w₀ / wz *
            exp(
                -(r / wz)^2 + im * (-(r^2 * z) / (z_R * wz^2) + atan(z, z_R) - k * z + ϕ₀),
            ) *
            exp(im * ω * t) *
            exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2),
        )
        E[2] ~ 0
        E[3] ~ real(
            2im / (k * wz^2) *
            (1 + im * (z / z_R)) *
            (x[2] * E[1] + x[3] * E[2]) *
            exp(im * ω * t) *
            exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2),
        )
        
        # Magnetic field components
        B[1] ~ 0
        B[2] ~ 0
        B[3] ~ real(
            2im / (k * c * wz^2) *
            (1 + im * (z / z_R)) *
            (x[3] * E[1] - x[2] * E[2]) *
            exp(im * ω * t) *
            exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2),
        )
        
        # Parameter relations
        ω ~ 2π * c / λ
        E₀ ~ a₀ * m * c * ω / abs(q)
        z_R ~ w₀^2 * k / 2
        k ~ 2π / λ
        τ0 ~ 10 / ω
    ]
    
    System(eqs, τ, [x, t, wz, z, r], [λ, a₀, m, q, ω, k, E₀, w₀, z_R, T0, τ0]; name, systems=[field_dynamics, spacetime])
end


"""
Plane wave electromagnetic field.

For a plane wave with normalized vector potential a₀ = eA/(mc²),
electrons can exhibit figure-8 motion when a₀ ~ 1.

Reference: Sarachik & Schappert, Phys. Rev. D 1, 2738 (1970)
"""
@component function PlaneWave(; name, amplitude=1.0, frequency=1.0, k_vector=[0,0,1], spacetime=Spacetime(c=1, name=:spacetime))
    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; spacetime)

    # Get spacetime variables from parent scope
    @unpack c = spacetime
    τ = ParentScope(spacetime.τ)

    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)

    E, B = field_dynamics.E, field_dynamics.B

    @parameters A=amplitude ω=frequency k[1:3]=k_vector λ
    @variables x⃗(τ)[1:3]

    eqs = [
        # Define spatial position from 4-position
        x⃗[1] ~ x[2]
        x⃗[2] ~ x[3]
        x⃗[3] ~ x[4]
        E[1] ~ A * cos(dot(k, x⃗) - ω * t)
        E[2] ~ 0
        E[3] ~ 0
        B[1] ~ 0
        B[2] ~ A/c * cos(dot(k, x⃗) - ω * t)
        B[3] ~ 0
        # parameters
        λ ~ (2π * c) / ω
    ]

    System(eqs, τ, [x, t, x⃗], [A, ω, k, λ]; name, systems=[field_dynamics, spacetime])
end

"""
Uniform electromagnetic field component.

In crossed E and B fields with E⊥B and |E| < |B|c,
particles drift with velocity v_drift = E×B/B²

Reference: Jackson, "Classical Electrodynamics", Section 12.4
"""
@component function UniformField(; name, E_field=[0,0,1], B_field=[0,0,0], spacetime)
    @named field_dynamics = EMFieldDynamics(; spacetime)

    τ = ParentScope(spacetime.τ)

    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)

    @parameters E₀[1:3]=E_field B₀[1:3]=B_field

    eqs = [
        field_dynamics.E ~ E₀,
        field_dynamics.B ~ B₀
    ]

    System(eqs, τ, [x, t], [E₀, B₀]; name, systems=[field_dynamics, spacetime])
end

"""
Laguerre-Gauss laser beam electromagnetic field.

Represents a focused beam with orbital angular momentum (OAM).
The beam is characterized by radial index p and azimuthal index m.

Parameters:
- λ: wavelength
- a₀: normalized vector potential amplitude
- w₀: beam waist (defaults to 75λ)
- radial_index (p): radial mode number, p ≥ 0
- azimuthal_index (m): azimuthal mode number (orbital angular momentum)
- temporal_profile: :gaussian (pulsed) or :constant (CW)
- temporal_width: pulse width for gaussian profile (defaults to 100.0)
- focus_position: focal position along z-axis (defaults to 0.0)

Reference: Allen et al., Phys. Rev. A 45, 8185 (1992)
"""
@component function LaguerreGaussLaser(;
    name,
    wavelength=1.0,
    amplitude=10.0,
    beam_waist=nothing,
    radial_index=0,      # p
    azimuthal_index=1,   # m
    spacetime=Spacetime(c=1, name=:spacetime),
    temporal_profile=:gaussian,  # :gaussian or :constant
    temporal_width=nothing,      # pulse width (for gaussian profile)
    focus_position=nothing       # focal position along z-axis
)
    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; spacetime)

    # Get spacetime variables from parent scope
    @unpack c = spacetime
    τ = ParentScope(spacetime.τ)

    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)

    E, B = field_dynamics.E, field_dynamics.B

    # Helper: compute |m|
    mₐ = abs(azimuthal_index)

    # Compute normalization factor using Pochhammer symbol (rising factorial)
    # Nₚₘ = √((p+1)_{|m|}) where (x)_n is the Pochhammer symbol
    Npm_val = sqrt(HypergeometricFunctions.pochhammer(radial_index + 1, mₐ))

    params = @parameters begin
        λ = wavelength
        a₀ = amplitude
        m_e = 9.10938356e-31  # electron mass (kg, SI)
        q = -1.602176634e-19  # electron charge (C, SI)
        ω
        k
        E₀
        w₀ = beam_waist === nothing ? 75λ : beam_waist
        z_R
        T0 = 100
        τ0 = temporal_width === nothing ? 100.0 : temporal_width

        # Laguerre-Gauss quantum numbers
        p = radial_index
        m = azimuthal_index

        # Normalization factor (computed from p and m)
        Nₚₘ = Npm_val
    end

    # Fixed parameters (computed values, not symbolic parameters)
    ξx = 1.0 + 0im
    ξy = 0 + 0im
    ϕ₀ = 0
    t₀ = 0  # Center pulse at t=0
    z₀ = focus_position === nothing ? 0.0 : focus_position

    # Derived variables
    @variables begin
        wz(τ)      # Beam width
        z(τ)       # Propagation coordinate
        r(τ)       # Radial distance
        θ(τ)       # Azimuthal angle
        σ(τ)       # Normalized radial coordinate squared
        rwz(τ)     # Scaled radial coordinate
        env(τ)     # Temporal envelope factor
    end

    # Compute temporal envelope expression based on profile type
    env_expr = if temporal_profile == :constant
        1.0  # No temporal envelope
    elseif temporal_profile == :gaussian
        exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2)
    else
        error("Unknown temporal_profile: $temporal_profile. Use :constant or :gaussian")
    end

    # Base Gaussian field (before polarization and LG modification)
    E_g = E₀ * w₀ / wz *
        exp(-(r / wz)^2) *
        exp(im * (-(r^2 * z) / (z_R * wz^2) + atan(z, z_R) - k * z + ϕ₀)) *
        exp(im * ω * t) *
        env

    # Laguerre-Gauss phase factor
    lg_phase = exp(im * ((2radial_index + mₐ) * atan(z, z_R) - azimuthal_index * θ + ϕ₀))

    # NEgexp term used in Ez and Bz
    NEgexp = Npm_val * E_g * lg_phase

    # Base Gaussian field with polarization (for Ex)
    E_gauss = ξx * E_g

    # Complex transverse field components (used in Ez, Bz)
    Ex_complex = E_gauss * Npm_val * rwz^mₐ * HypergeometricFunctions._₁F₁(-radial_index, mₐ + 1, 2σ) * lg_phase
    Ey_complex = ξy * E_g * Npm_val * rwz^mₐ * HypergeometricFunctions._₁F₁(-radial_index, mₐ + 1, 2σ) * lg_phase

    eqs = [
        # Extract coordinates from 4-position
        z ~ x[4]
        r ~ hypot(x[2], x[3])
        θ ~ atan(x[3], x[2])

        # Beam width as function of z
        wz ~ w₀ * √(1 + (z / z_R)^2)

        # Intermediate variables
        σ ~ (r / wz)^2
        rwz ~ r * √2 / wz

        # Temporal envelope
        env ~ env_expr

        # Electric field Ex component (Laguerre-Gauss)
        E[1] ~ real(
            E_gauss * Nₚₘ * rwz^mₐ * HypergeometricFunctions._₁F₁(-p, mₐ + 1, 2σ) * exp(im * ((2p + mₐ) * atan(z, z_R) - m * θ + ϕ₀))
        )

        # Electric field Ey component (reuses Ex)
        E[2] ~ (ξy / ξx) * E[1]

        # Electric field Ez component (longitudinal)
        E[3] ~ real(
            -im / k * (
                # Term 1: m-dependent term
                (azimuthal_index == 0 ? 0.0 :
                    mₐ * (ξx - im * sign(azimuthal_index) * ξy) *
                    (√2 / wz)^mₐ * r^(mₐ - 1) *
                    HypergeometricFunctions._₁F₁(-p, mₐ + 1, 2σ) *
                    NEgexp *
                    exp(im * sign(azimuthal_index) * θ)
                ) -
                # Term 2: transverse field coupling
                2 / (wz^2) * (1 + im * z / z_R) * (x[2] * Ex_complex + x[3] * Ey_complex) -
                # Term 3: p-dependent term
                (radial_index == 0 ? 0.0 :
                    4 * radial_index / ((mₐ + 1) * wz^2) *
                    (x[2] * ξx + x[3] * ξy) *
                    rwz^mₐ *
                    HypergeometricFunctions._₁F₁(-p + 1, mₐ + 2, 2σ) *
                    NEgexp
                )
            )
        )

        # Magnetic field components
        # For paraxial beam: By ≈ Ex/c, Bx ≈ -Ey/c
        B[1] ~ -E[2] / c
        B[2] ~ E[1] / c

        # Bz component (full Laguerre-Gauss formula)
        B[3] ~ real(
            -im / ω * (
                # Term 1: m-dependent term
                -(azimuthal_index == 0 ? 0.0 :
                    mₐ * (ξy + im * sign(azimuthal_index) * ξx) *
                    (√2 / wz)^mₐ * r^(mₐ - 1) *
                    HypergeometricFunctions._₁F₁(-p, mₐ + 1, 2σ) *
                    NEgexp *
                    exp(im * sign(azimuthal_index) * θ)
                ) +
                # Term 2: transverse field coupling
                2 / (wz^2) * (1 + im * z / z_R) * (x[2] * Ey_complex - x[3] * Ex_complex) +
                # Term 3: p-dependent term
                (radial_index == 0 ? 0.0 :
                    4 * radial_index / ((mₐ + 1) * wz^2) *
                    (x[2] * ξy - x[3] * ξx) *
                    rwz^mₐ *
                    HypergeometricFunctions._₁F₁(-p + 1, mₐ + 2, 2σ) *
                    NEgexp
                )
            )
        )

        # Parameter relations
        ω ~ 2π * c / λ
        k ~ 2π / λ
        z_R ~ π * w₀^2 / λ
        E₀ ~ a₀ * m_e * c * ω / abs(q)
    ]

    return System(eqs, τ, [x, t, z, r, θ, wz, σ, rwz, env], params;
                     name, systems=[field_dynamics])
end
