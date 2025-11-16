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
@component function GaussLaser(; name, wavelength=1.0, amplitude=10.0, beam_waist=nothing, ref_frame)
    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; ref_frame)

    # Get spacetime variables and constants from parent scope
    @unpack c, m_e, q_e = ref_frame
    τ = ref_frame.τ

    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)

    @unpack E, B = field_dynamics

    @parameters begin
        λ = wavelength
        a₀ = amplitude
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
        E₀ ~ a₀ * m_e * c * ω / abs(q_e)
        z_R ~ w₀^2 * k / 2
        k ~ 2π / λ
        τ0 ~ 10 / ω
    ]

    sys = System(eqs, τ, [x, t, wz, z, r], [λ, a₀, ω, k, E₀, w₀, z_R, T0, τ0]; name, systems=[ref_frame])

    extend(sys, field_dynamics)
end


"""
Plane wave electromagnetic field.

For a plane wave with normalized vector potential a₀ = eA/(mc²),
electrons can exhibit figure-8 motion when a₀ ~ 1.

Reference: Sarachik & Schappert, Phys. Rev. D 1, 2738 (1970)
"""
@component function PlaneWave(; name, amplitude=1.0, frequency=1.0, k_vector=[0,0,1], ref_frame)
    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; ref_frame)

    # Get spacetime variables and constants from parent scope
    @unpack c, m_e, q_e = ref_frame
    τ = ref_frame.τ

    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)

    @unpack E, B = field_dynamics

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

    sys = System(eqs, τ, [x, t, x⃗], [A, ω, k, λ]; name, systems=[ref_frame])

    extend(sys, field_dynamics)
end

"""
Uniform electromagnetic field component.

In crossed E and B fields with E⊥B and |E| < |B|c,
particles drift with velocity v_drift = E×B/B²

Reference: Jackson, "Classical Electrodynamics", Section 12.4
"""
@component function UniformField(; name, E_field=[0,0,1], B_field=[0,0,0], ref_frame)
    @named field_dynamics = EMFieldDynamics(; ref_frame)
    @unpack E, B = field_dynamics

    # Get spacetime variables and constants from parent scope
    @unpack c, m_e, q_e = ref_frame
    τ = ref_frame.τ

    # Create local position and time variables
    @variables x(τ)[1:4] t(τ)

    @parameters E₀[1:3]=E_field B₀[1:3]=B_field

    eqs = [
        field_dynamics.E ~ E₀,
        field_dynamics.B ~ B₀
    ]

    sys = System(eqs, τ, [x, t], [E₀, B₀]; name, systems=[ref_frame])

    extend(sys, field_dynamics)
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
    ref_frame,
    temporal_profile=:gaussian,  # :gaussian or :constant
    temporal_width=nothing,      # pulse width (for gaussian profile)
    focus_position=nothing,       # focal position along z-axis
    polarization = :linear
)
    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; ref_frame)

    # Get spacetime variables from parent scope
    @unpack c, m_e, q_e = ref_frame
    iv = ModelingToolkit.get_iv(ref_frame)

    # Create local position and time variables
    @variables x(iv)[1:4]
    if nameof(iv) == :τ
        @variables t(iv)
    else
        t = iv
    end

    @unpack E, B = field_dynamics

    # Helper: compute |m|
    mₐ = abs(azimuthal_index)
    sgn = sign(azimuthal_index)

    # Compute normalization factor using Pochhammer symbol (rising factorial)
    # Nₚₘ = √((p+1)_{|m|}) where (x)_n is the Pochhammer symbol
    Npm_val = sqrt(pochhammer(radial_index + 1, mₐ))

    params = @parameters begin
        λ = wavelength
        a₀ = amplitude
        ω
        k
        E₀
        w₀ = beam_waist === nothing ? 75λ : beam_waist
        z_R
        τ0 = temporal_width === nothing ? 100.0 : temporal_width

        # Laguerre-Gauss quantum numbers
        p = radial_index
        m = azimuthal_index

        # Normalization factor (computed from p and m)
        Nₚₘ = Npm_val
    end

    if polarization == :linear
        ξx = 1.0 + 0im
        ξy = 0 + 0im
    elseif polarization == :circular
        ξx, ξy = (1/√2, im/√2) .|> complex
    else
        error("polarization $polarization not supported.")
    end

    # Fixed parameters (computed values, not symbolic parameters)
    ϕ₀ = 0
    t₀ = 0  # Center pulse at t=0
    z₀ = focus_position === nothing ? 0.0 : focus_position

    # Derived variables
    @variables begin
        wz(iv)      # Beam width
        z(iv)       # Propagation coordinate
        r(iv)       # Radial distance
        θ(iv)       # Azimuthal angle
        σ(iv)       # Normalized radial coordinate squared
        rwz(iv)     # Scaled radial coordinate
        env(iv)     # Temporal envelope factor
    end

    # Compute temporal envelope expression based on profile type
    env_expr = if temporal_profile == :constant
        1.0  # No temporal envelope
    elseif temporal_profile == :gaussian
        exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2)
    else
        error("Unknown temporal_profile: $temporal_profile. Use :constant or :gaussian")
    end

    # We need to factor out the complex expressions so that we don't have equations with complex lhs
    # since MTK doesn't handle that well

    # Base Gaussian field (before polarization and LG modification)
    E_g = E₀ * w₀ / wz *
        exp(-(r / wz)^2) *
        exp(im * (-(r^2 * z) / (z_R * wz^2) + atan(z, z_R) - k * z + ϕ₀)) *
        exp(im * ω * t) * env

    # Laguerre-Gauss phase factor
    lg_phase = exp(im * ((2p + mₐ) * atan(z, z_R) - m * θ + ϕ₀))

    # NEgexp term used in Ez and Bz
    NEgexp = Nₚₘ * E_g * lg_phase

    # Complex transverse field components (used in Ez, Bz)
    Ex_complex = ξx * E_g * Nₚₘ * rwz^mₐ * _₁F₁(-p, mₐ + 1, 2σ) * lg_phase
    Ey_complex = ξy * E_g * Nₚₘ * rwz^mₐ * _₁F₁(-p, mₐ + 1, 2σ) * lg_phase

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
        E[1] ~ real(Ex_complex)

        # Electric field Ey component
        E[2] ~ real(Ey_complex)

        # Electric field Ez component (longitudinal)
        E[3] ~ real(
            -im / k * (
                # Term 1: m-dependent term
                (azimuthal_index == 0 ? 0.0 :
                    mₐ * (ξx - im * sgn * ξy) *
                    (√2 / wz)^mₐ * r^(mₐ - 1) * _₁F₁(-p, mₐ + 1, 2σ) *
                    NEgexp * exp(im * sgn * θ)
                ) -
                # Term 2: transverse field coupling
                2 / (wz^2) * (1 + im * z / z_R) * (x[2] * Ex_complex + x[3] * Ey_complex) -
                # Term 3: p-dependent term
                (radial_index == 0 ? 0.0 :
                    4 * p / ((mₐ + 1) * wz^2) * (x[2] * ξx + x[3] * ξy) *
                    rwz^mₐ * _₁F₁(-p + 1, mₐ + 2, 2σ) * NEgexp
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
                    mₐ * (ξy + im * sgn * ξx) *
                    (√2 / wz)^mₐ * r^(mₐ - 1) * _₁F₁(-p, mₐ + 1, 2σ) *
                    NEgexp * exp(im * sgn * θ)
                ) +
                # Term 2: transverse field coupling
                2 / (wz^2) * (1 + im * z / z_R) * (x[2] * Ey_complex - x[3] * Ex_complex) +
                # Term 3: p-dependent term
                (radial_index == 0 ? 0.0 :
                    4 * p / ((mₐ + 1) * wz^2) * (x[2] * ξy - x[3] * ξx) *
                    rwz^mₐ * _₁F₁(-p + 1, mₐ + 2, 2σ) * NEgexp
                )
            )
        )

        # Parameter relations
        ω ~ 2π * c / λ
        k ~ 2π / λ
        z_R ~ π * w₀^2 / λ
        E₀ ~ a₀ * m_e * c * ω / abs(q_e)
    ]

    vars = if nameof(iv) == :τ
        [x, t, z, r, θ, wz, σ, rwz, env]
    else
        @variables τ(iv)
        [x, τ, z, r, θ, wz, σ, rwz, env]
    end

    sys = System(eqs, iv, vars, params; name, systems=[ref_frame])
    extend(sys, field_dynamics)
end
