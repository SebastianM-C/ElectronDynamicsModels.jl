function ClassicalElectronModel(; name)
    @parameters begin
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
    end

    @variables t(τ) [guess = 0] γ(τ) x(τ)[1:4] [guess = [c * t, 0, 0, 0]]
    @variables u(τ)[1:4] [guess = [c, 0, 0, 0]] p(τ)[1:4]

    @named external_field = ElectromagneticField()
    Fμν = external_field.Fμν

    eqs = [
        t ~ x[1] / c
        p ~ m * u
        u[1] ~ γ * c
        # u[2] ~ γ * v[1]
        # u[3] ~ γ * v[2]
        # u[4] ~ γ * v[3]
        # γ ~ 1 / √(1 - (v ⋅ v) / c^2)
        dτ(x) ~ u
        dτ(u) ~ q / m * (Fμν * gμν * u)
    ]

    ODESystem(eqs, τ; name, systems = [external_field])
end

function ClassicalElectronModel2(; name)
    @parameters begin
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
    end

    @independent_variables t
    Dt = Differential(t)
    @variables γ(t) x(t)[1:4] [guess = [c * t, 0, 0, 0]]
    @variables u(t)[1:4] [guess = [c, 0, 0, 0]] p(t)[1:4] v(t)[1:3] = zeros(3) p⃗(t)[1:3]

    @named external_field = ElectromagneticField(; t)
    (; E, B) = external_field

    eqs = [
        t ~ x[1] / c
        p ~ m * u
        # p[1] ~ γ * m * c
        p[2] ~ p⃗[1]
        p[3] ~ p⃗[2]
        p[4] ~ p⃗[3]
        u[1] ~ γ * c
        u[2] ~ γ * v[1]
        u[3] ~ γ * v[2]
        u[4] ~ γ * v[3]
        # γ ~ 1 / √(1 - (v ⋅ v) / c^2)
        Dt(x[2]) ~ v[1]
        Dt(x[3]) ~ v[2]
        Dt(x[4]) ~ v[3]
        Dt(γ) ~ q / (m * c) * v ⋅ E
        Dt(p⃗) ~ q * (E + v × B)
    ]

    ODESystem(eqs, t; name, systems = [external_field])
end
