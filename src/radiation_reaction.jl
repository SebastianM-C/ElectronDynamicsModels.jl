function LandauLifshitzRadiationReaction(; name)
    @parameters begin
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
        ϵ₀ = austrip(ε_0)
    end

    @independent_variables t
    Dt = Differential(t)
    @variables γ(t) x(t)[1:4] [guess = [c * t, 0, 0, 0]] u(t)[1:4] [
        guess = [c, 0, 0, 0],
    ] v(t)[1:3] p(t)[1:4] p⃗(t)[1:3]

    @variables Pcl(t)
    @parameters P0 Ecr rₑ = q^2 / (4π * ϵ₀ * m * c^2) [tunable = false] τₑ = rₑ / c [
        tunable = false,
    ]
    @named external_field = ElectromagneticField(; t)
    E, B = external_field.E, external_field.B

    eqs = [
        # t ~ x[1] / c
        p ~ m * u
        p[1] ~ γ * m * c
        p[2] ~ p⃗[1]
        p[3] ~ p⃗[2]
        p[4] ~ p⃗[3]
        # u[1] ~ γ * c
        u[2] ~ γ * v[1]
        u[3] ~ γ * v[2]
        u[4] ~ γ * v[3]
        # γ ~ 1 / √(1 - (v ⋅ v) / c^2)
        Pcl ~ P0 * γ^2 / Ecr^2 * abs((E + v × B) ⋅ (E + v × B) - (v ⋅ E)^2)
        Dt(x) ~ u
        m * c^2 * Dt(γ) ~ q * c * v ⋅ E - Pcl
        Dt(p⃗) ~ q * (E + v × B) - Pcl * v / (c * v ⋅ v)
    ]

    ODESystem(
        eqs,
        t;
        name,
        systems = [external_field],
        parameter_dependencies = [
            P0 ~ 2m * c^2 / (3τₑ)
            Ecr ~ 4π * ϵ₀ * m^2 * c^4 / q^3
        ],
    )
end
