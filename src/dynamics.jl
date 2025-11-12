@component function ParticleDynamics(; name, mass, ref_frame)
    iv = ModelingToolkit.get_iv(ref_frame)
    D = Differential(iv)

    @variables begin
        γ(iv), [description = "Lorentz factor"]
        x(iv)[1:4], [description = "4-position (ct, x, y, z)"]
        u(iv)[1:4], [description = "4-velocity (u⁰, u¹, u², u³, u⁴)"]
        p(iv)[1:4]
        F_total(iv)[1:4]
    end

    if nameof(iv) == :τ
        τ = iv
        c = ref_frame.c

        @variables begin
            t(iv), [description = "Universal time"]
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
            D(x) ~ u
            # Newton's second law in covariant form
            # m du^μ/dτ = F^μ
            D(u) ~ F_total / m
        ]

        System(eqs, iv, [t, γ, x, u, p, F_total], [m]; name, systems = [ref_frame])
    elseif nameof(iv) == :t
        t = iv
        c = ref_frame.c

        @variables begin
            τ(t), [description = "Proper time"]
        end
        @parameters m = mass

        eqs = [
            # Proper time evolution: dτ/dt = 1/γ
            D(τ) ~ 1/γ
            # Time coordinate
            x[1] ~ c * t
            # 4-momentum
            p ~ m * u
            # Timelike component of 4-velocity
            u[1] ~ γ * c
            # Position evolution: dx^μ/dt = u^μ/γ
            D(x) ~ u / γ
            # Momentum/velocity evolution: du^μ/dt = F^μ/(m*γ)
            D(u) ~ F_total / (m * γ)
        ]

        System(eqs, iv, [τ, γ, x, u, p, F_total], [m]; name, systems = [ref_frame])
    end
end
