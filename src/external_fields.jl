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
