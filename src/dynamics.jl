@component function ParticleDynamics(; name, mass = 1.0, spacetime)
    @unpack c = spacetime

    @variables begin
        t(τ), [description = "Universal time"]
        γ(τ), [description = "Lorentz factor"]
        x(τ)[1:4], [guess = [c * t, 0, 0, 0], description = "4-position (ct, x, y, z)"]
        u(τ)[1:4], [guess = [c, 0, 0, 0], description = "4-velocity (u⁰, u¹, u², u³, u⁴)"]
        p(τ)[1:4]
        F_total(τ)[1:4]
    end
    @parameters m = mass

    eqs = [
        # Proper time and coordinate time relation
        # dτ = dt/γ where γ = 1/√(1-v²/c²)
        t ~ x[1] / c
        p ~ m * u
        u[1] ~ γ * c
        # Covariant equation of motion
        # dx^μ/dτ = u^μ
        dτ(x) ~ u
        # Newton's second law in covariant form
        # m du^μ/dτ = F^μ
        dτ(u) ~ F_total / m
    ]

    System(eqs, τ, [t, γ, x, u, p, F_total], [m]; name, systems = [spacetime])
end
