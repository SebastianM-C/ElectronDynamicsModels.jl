function faraday(E, B, c)
    return [
        0 -E[1] / c -E[2] / c -E[3] / c
        E[1] / c 0 -B[3] B[2]
        E[2] / c B[3] 0 -B[1]
        E[3] / c -B[2] B[1] 0
    ]
end

@component function EMFieldDynamics(; name, world)
    iv = ModelingToolkit.get_iv(world)

    @unpack gμν, c, μ₀, ε₀ = world

    @variables begin
        E(iv)[1:3]                                  # Electric field (3-vector)
        B(iv)[1:3]                                  # Magnetic field (3-vector)
        Fμν(iv)[1:4, 1:4], [description = "Faraday tensor"]
        T(iv)[1:4, 1:4], [description = "Stress-energy tensor T^μν"]
    end

    eqs = [
        # Faraday tensor from E and B
        Fμν ~ faraday(E, B, c),

        # Electromagnetic stress-energy tensor (kept as a diagnostic observable)
        # T^μν = (1/μ₀)[F^μα F^ν_α - (1/4)η^μν F^αβ F_αβ]
        # References:
        # - Landau & Lifshitz "The Classical Theory of Fields" §33
        # - Jackson "Classical Electrodynamics" 3rd Ed. §12.10
        T ~ (1 / μ₀) * (
            Fμν * transpose(Fμν * gμν) -            # F^μα F^ν_α
                (1 / 4) * gμν * (dot(E, E) / c^2 - dot(B, B))
        ),
    ]

    System(eqs, iv, [E, B, Fμν, T], [μ₀, ε₀]; name, systems = [world])
end
