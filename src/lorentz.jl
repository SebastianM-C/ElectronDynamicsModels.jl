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
