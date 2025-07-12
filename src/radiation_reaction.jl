function LandauLifshitzRadiationReaction(; name)
    @parameters begin
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
        ϵ₀ = austrip(ε_0)
    end

    @independent_variables t
    Dt = Differential(t)
    @variables γ(t) x(t)[1:4] [guess = [c * t, 0, 0, 0]] τ(t)
    @variables u(t)[1:4] [guess = [c, 0, 0, 0]] v(t)[1:3] p(t)[1:4] p⃗(t)[1:3]

    @variables Pcl(t) Ė(t)[1:3] Ḃ(t)[1:3]
    @parameters P0 Ecr rₑ = q^2 / (4π * ϵ₀ * m * c^2) [tunable = false] τₑ = rₑ / c [
        tunable = false,
    ]
    @named external_field = ElectromagneticField(; t)
    E, B = external_field.E, external_field.B

    eqs = [
        x[1] ~ c * t
        Dt(τ) ~ 1 / γ
        Dt(x[2]) ~ (p⃗/(γ*m))[1]
        Dt(x[3]) ~ (p⃗/(γ*m))[2]
        Dt(x[4]) ~ (p⃗/(γ*m))[3]
        v ~ p⃗ / (γ * m)
        Ė ~ Dt(E)
        Ḃ ~ Dt(B)
        Dt(γ) ~
            q / (m * c^2) * v ⋅ E + 2 / 3 * q * τₑ * γ * Ė ⋅ v -
            2 / 3 * q / Ecr * E ⋅ (E + v × B) +
            2 / 3 * q / Ecr * γ^2 * ((E + v × B) ⋅ (E + v × B) - (v ⋅ E)^2)
        Dt(p⃗) ~
            q * (E + v × B) + 2 / (3 * c) * q * τₑ * γ * (Ė + v × Ḃ) -
            2 / (3 * c) * q / Ecr * ((v ⋅ E) * E - B × (E + v × B)) +
            2 / (3 * c) * γ^2 * ((E + v × B) ⋅ (E + v × B) - (v ⋅ E)^2) * v
        # parameters
        P0 ~ 2m * c^2 / (3τₑ)
        Ecr ~ 4π * ϵ₀ * m^2 * c^4 / q^3
    ]

    System(eqs, t; name, systems = [external_field])
end
