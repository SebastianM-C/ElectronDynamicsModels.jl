"""
Landau-Lifshitz Radiation Reaction

Implements the Landau-Lifshitz formulation of radiation reaction for a charged particle.
This avoids the runaway solutions of Abraham-Lorentz by eliminating second derivatives.

References:
- Landau, L.D. & Lifshitz, E.M. "The Classical Theory of Fields" §76
- Jackson, J.D. "Classical Electrodynamics" 3rd Ed., Section 17.8

The Landau-Lifshitz equation in covariant form:
dp^μ/dτ = q F^μν u_ν + (2q³/3m²c³)[∂_ν F^μλ u^ν u_λ + F^μν F_νλ u^λ - F_νλ F^νλ u^μ]

For uniform fields (∂_ν F^μλ = 0), this reduces to:
dp^μ/dτ = q F^μν u_ν + (2q³/3m²c³)[F^μν F_νλ u^λ - (1/4)F_αβ F^αβ u^μ]
"""
@component function LandauLifshitzRadiation(; name, charge=1.0, F_lorentz_ref, spacetime, particle)
    @named field = ElectromagneticSystem()

    @unpack c, gμν = spacetime
    τ = ParentScope(spacetime.τ)
    u = ParentScope(particle.u)

    @parameters m=1.0 ε₀=1.0 q=charge
    @variables begin
        F_rad(τ)[1:4]      # Radiation reaction 4-force
        P_rad(τ)           # Radiated power (invariant)
        τ₀(τ)              # Classical electron radius time scale
        F_sq(τ)            # F_μν F^μν (invariant)
        F_dot_u(τ)[1:4]   # F^μν u_ν (Lorentz force per unit charge)
        FF_dot_u(τ)[1:4]  # F^μν F_νλ u^λ term
    end

    # Reference to the Lorentz force from another component
    F_lorentz = ParentScope(F_lorentz_ref)

    eqs = [
        # Classical electron radius time scale
        # τ₀ = r_e/c = e²/(6πε₀mc³)
        τ₀ ~ q^2 / (6π * ε₀ * m * c^3),

        # F^μν u_ν - this is the Lorentz force per unit charge
        [F_dot_u[μ] ~ sum(field.Fμν[μ,ν] * sum(gμν[ν,λ] * u[λ] for λ in 1:4)
                          for ν in 1:4) for μ in 1:4]...,

        # Calculate F_μν F^μν (Lorentz invariant)
        # For electromagnetic field: F_μν F^μν = 2(B² - E²/c²)
        F_sq ~ sum(sum(field.Fμν[μ,ν] * sum(sum(gμν[μ,α] * gμν[ν,β] * field.Fμν[α,β]
                    for α in 1:4) for β in 1:4) for ν in 1:4) for μ in 1:4),

        # F^μν F_νλ u^λ term
        [FF_dot_u[μ] ~ sum(sum(field.Fμν[μ,ν] *
                              sum(sum(gμν[ν,α] * field.Fμν[α,λ] *
                                     sum(gμν[λ,β] * u[β] for β in 1:4)
                                 for λ in 1:4) for α in 1:4)
                          for ν in 1:4)) for μ in 1:4]...,

        # Landau-Lifshitz radiation reaction force
        # F_rad^μ = (2q³/3m²c³) * [F^μν F_νλ u^λ - (1/4)F_αβ F^αβ u^μ]
        # Factor of q² comes from q³/q (since F_lorentz already has one q)
        F_rad ~ (2/3) * τ₀ * q / (m * c^2) * (FF_dot_u - (1/4) * F_sq * u),

        # Radiated power - Larmor formula
        # P = (2e²/3c³) * (dp^μ/dτ)(dp_μ/dτ) = (2e²/3m²c³) * F^μ F_μ
        # Using absolute value to ensure positive power
        P_rad ~ (2/3) * τ₀ * c * abs(m_dot(F_lorentz, F_lorentz)) / m^2,
    ]

    System(eqs, τ, [F_rad, P_rad, τ₀, F_sq, F_dot_u, FF_dot_u], [m, ε₀, q];
           name, systems=[field, spacetime])
end

"""
Abraham-Lorentz Radiation Reaction

Implements the covariant formulation of radiation reaction for a charged particle.

References:
- Abraham, M. (1905). "Theorie der Elektrizität"
- Lorentz, H.A. (1916). "The Theory of Electrons"
- Landau, L.D. & Lifshitz, E.M. "The Classical Theory of Fields" §76
- Jackson, J.D. "Classical Electrodynamics" 3rd Ed., Chapter 17

The radiation reaction force is given by:
F_rad^μ = (2e²/3mc³) * (d²x^μ/dτ²)

We use the Schott term approximation to avoid runaway solutions.
This is valid when ω << m*c²/ℏ (classical regime).
"""
@component function AbrahamLorentzRadiation(; name, charge=1.0, F_lorentz_ref, spacetime, particle)
    @unpack c, gμν = spacetime
    τ = ParentScope(spacetime.τ)
    u = ParentScope(particle.u)

    @parameters m=1.0 ε₀=1.0 q=charge
    @variables begin
        F_rad(τ)[1:4]     # 4-force from radiation reaction
        P_rad(τ)          # Radiated power (invariant)
        τ₀(τ)             # Classical electron radius time scale
        F_ext_squared(τ)  # External force squared (invariant)
    end

    # Reference to the Lorentz force from another component
    F_lorentz = ParentScope(F_lorentz_ref)

    eqs = [
        # Classical electron radius time scale
        # τ₀ = r_e/c = e²/(6πε₀mc³)
        τ₀ ~ q^2 / (6π * ε₀ * m * c^3),

        # External force squared (Lorentz invariant)
        F_ext_squared ~ m_dot(F_lorentz, F_lorentz) / m^2,

        # Abraham-Lorentz 4-force (leading order)
        # F_rad^μ = (2/3) * τ₀/c² * F_ext²/m * u^μ
        # This avoids the runaway solutions
        F_rad ~ (2/3) * τ₀ * (F_ext_squared / c^2) * u,

        # Radiated power - Covariant Larmor formula
        # P = (2e²/3c³) * (dp^μ/dτ)(dp_μ/dτ)
        P_rad ~ (2/3) * τ₀ * c * abs(F_ext_squared),
    ]

    System(eqs, τ, [F_rad, P_rad, τ₀, F_ext_squared], [m, ε₀, q];
           name, systems=[spacetime])
end
