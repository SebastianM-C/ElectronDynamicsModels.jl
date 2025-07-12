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

    System(eqs, τ; name, systems = [external_field])
end

function ClassicalElectronModel2(; name)
    @parameters begin
        c = austrip(c_0), [tunable = false]
        m = austrip(m_e)
        q = austrip(e)
    end

    @independent_variables t
    Dt = Differential(t)
    @variables γ(t) x(t)[1:4] [guess = [c * t, 0, 0, 0]] τ(t)
    @variables u(t)[1:4] [guess = [c, 0, 0, 0]] p(t)[1:4] v(t)[1:3] = zeros(3) p⃗(t)[1:3]

    @named external_field = ElectromagneticField(; t)
    (; E, B) = external_field

    eqs = [
        x[1] ~ c * t
        Dt(τ) ~ 1 / γ
        # p ~ m * u
        # p⃗ ~ γ*m*v
        # u[1] ~ γ * c
        # u[2] ~ γ * v[1]
        # u[3] ~ γ * v[2]
        # u[4] ~ γ * v[3]
        # γ ~ 1 / √(1 - (v ⋅ v) / c^2)
        Dt(x[2]) ~ (p⃗/(γ*m))[1]
        Dt(x[3]) ~ (p⃗/(γ*m))[2]
        Dt(x[4]) ~ (p⃗/(γ*m))[3]
        Dt(γ) ~ q / (m * c^2) * (p⃗/(γ*m)) ⋅ E
        Dt(p⃗) ~ q * (E + p⃗/(γ*m) × B)
    ]

    System(eqs, t; name, systems = [external_field])
end
