"""
Landau-Lifshitz Radiation Reaction

Implements the Landau-Lifshitz formulation of radiation reaction for a charged particle.
This avoids the runaway solutions of Abraham-Lorentz by eliminating second derivatives.

References:
- Landau, L.D. & Lifshitz, E.M. "The Classical Theory of Fields" §76
- Niel et. al. 2018, 10.1103/PhysRevE.97.043209, eq. 2 and 3

The Landau-Lifshitz equation in covariant form:
dp^μ/dτ = q F^μν u_ν + (2τₑ/3)[q ∂_ν F^μλ u^ν u_λ + q²/m F^μν F_νλ u^λ + q²/mc² (F^νλ u^λ) (F_νγ u^γ) u^μ]
"""
@component function LandauLifshitzRadiation(; name, charge = 1.0, F_lorentz_ref, world, particle)
    @unpack c, gμν, ε₀ = world
    iv = ModelingToolkit.get_iv(world)
    @named field = ElectromagneticSystem(iv)
    Fμν = field.Fμν

    u = ParentScope(particle.u)
    x = ParentScope(particle.x)

    @parameters m = 1.0 q = charge
    @variables begin
        F_rad(iv)[1:4]        # Radiation reaction 4-force
        P_rad(iv)             # Radiated power (invariant)
        τₑ(iv)                # Classical electron radius time scale
        grad_F_dot_u(iv)[1:4] # ∂_ν F^μλ u^ν u_λ term
        FF_dot_u(iv)[1:4]     # F^μν F_νλ u^λ term
        w(iv)[1:4]            # w_ν = F_νβ u^β
    end

    # Reference to the Lorentz force from another component
    F_lorentz = ParentScope(F_lorentz_ref)
    ∂_nu(ν, f) = expand_derivatives(Differential(x[ν])(f))

    eqs = [
        # Classical electron radius time scale
        # τₑ = r_e/c = e²/(4πε₀mc³)
        τₑ ~ q^2 / (4π * ε₀ * m * c^3),

        # ∂_ν F^μλ u^ν u_λ = ∂_ν F^μλ u^ν g_μβ u^β
        [
            grad_F_dot_u[μ] ~
                sum(sum(sum(∂_nu(ν, Fμν[μ, λ]) * u[ν] * gμν[λ, β] * u[β] for ν in 1:4) for λ in 1:4) for β in 1:4)
                for μ in 1:4
        ]...,

        # F^μν F_νλ u^λ = F^μν g_αν g_βλ F^αβ u^λ
        [
            FF_dot_u[μ] ~ sum(sum(sum(sum(Fμν[μ, ν] * gμν[α, ν] * gμν[β, λ] * Fμν[α, β] * u[λ] for ν in 1:4) for α in 1:4) for β in 1:4) for λ in 1:4)
                for μ in 1:4
        ]...,

        # w_ν = F_νβ u^β = g_αν g_βλ F^αλ u^β
        [
            w[ν] ~ sum(sum(sum(gμν[α, ν] * gμν[β, λ] * Fμν[α, λ] * u[β] for α in 1:4) for β in 1:4) for λ in 1:4)
                for ν in 1:4
        ]...,

        # Landau-Lifshitz radiation reaction force
        # F_rad^μ = (2τₑ/3)[q ∂_ν F^μλ u^ν u_λ + q²/m F^μν F_νλ u^λ + q²/mc² w_ν w_ν u^μ]
        [F_rad[μ] ~ (2 / 3) * τₑ * (q * grad_F_dot_u[μ] + q^2 / m * FF_dot_u[μ] + q^2 / (m * c^2) * m_dot(w, w) * u[μ]) for μ in 1:4]...,

        # Radiated power - Larmor formula
        # P = (2/3m) τₑ * (dp^μ/dτ)(dp_μ/dτ) = (2τₑ/3m) * F^μ F_μ
        # Using absolute value to ensure positive power
        P_rad ~ (2 / 3) * τₑ * abs(m_dot(F_lorentz, F_lorentz)) / m,
    ]

    System(
        eqs, iv, [F_rad, P_rad, τₑ, grad_F_dot_u, FF_dot_u, w], [m, ε₀, q];
        name, systems = [field]
    )
end
