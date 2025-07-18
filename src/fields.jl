@component function EMFieldDynamics(; name, spacetime)
    @named field = ElectromagneticSystem()

    @unpack c, gμν = spacetime

    @parameters μ₀ = 1.0 ε₀ = 1.0
    @variables begin
        E(τ)[1:3]         # Electric field (3-vector)
        B(τ)[1:3]         # Magnetic field (3-vector)
    end

    eqs = [
        # Faraday tensor from E and B
        field.Fμν ~ [
            0 -E[1]/c -E[2]/c -E[3]/c
            E[1]/c 0 -B[3] B[2]
            E[2]/c B[3] 0 -B[1]
            E[3]/c -B[2] B[1] 0
        ],

        # Electromagnetic stress-energy tensor
        # T^μν = (1/μ₀)[F^μα F^ν_α - (1/4)η^μν F^αβ F_αβ]
        # References:
        # - Landau & Lifshitz "The Classical Theory of Fields" §33
        # - Jackson "Classical Electrodynamics" 3rd Ed. §12.10
        field.T ~ (1 / μ₀) * (
            field.Fμν * transpose(field.Fμν * gμν) -  # F^μα F^ν_α
            (1 / 4) * gμν * (dot(E, E) / c^2 - dot(B, B))
        ),
    ]

    System(eqs, τ, [E, B], [μ₀, ε₀]; name, systems=[field])
end
