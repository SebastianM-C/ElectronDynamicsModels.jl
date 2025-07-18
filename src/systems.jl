@component function ChargedParticle(;
    name,
    mass = 1.0,
    charge = 1.0,
    spacetime = Spacetime(c = 1, name = :spacetime),
    external_field = PlaneWave(; spacetime, name = :laser),
    radiation_model = nothing,
)
    @named particle = ParticleDynamics(; mass, spacetime)

    @unpack gμν = spacetime
    Fμν = ParentScope(external_field.field_dynamics.field.Fμν)
    J = ParentScope(external_field.field_dynamics.field.J)
    τ = ParentScope(spacetime.τ)

    @parameters q = charge
    @variables x(τ)[1:4] u(τ)[1:4] F_lorentz(τ)[1:4]

    systems = [particle, external_field, spacetime]

    eqs = [
        # shortcuts
        x ~ particle.x
        u ~ particle.u
        
        # Connect particle position and time to external field
        external_field.x ~ particle.x
        external_field.t ~ particle.t

        # Lorentz force
        F_lorentz ~ q * (Fμν * gμν * u)
        # Set 4-current density in electromagnetic system
        J ~ q * u
    ]

    # Handle radiation reaction if specified
    if radiation_model == :landau_lifshitz
        @named radiation =
            LandauLifshitzRadiation(; charge, F_lorentz_ref = F_lorentz, spacetime, particle)
        push!(systems, radiation)
        push!(eqs, external_field.field_dynamics.field.Fμν ~ radiation.field.Fμν)
        push!(eqs, particle.F_total ~ F_lorentz + radiation.F_rad)
    elseif radiation_model == :abraham_lorentz
        @named radiation =
            AbrahamLorentzRadiation(; charge, F_lorentz_ref = F_lorentz, spacetime, particle)
        push!(systems, radiation)
        push!(eqs, particle.F_total ~ F_lorentz + radiation.F_rad)
    else
        # No radiation reaction
        push!(eqs, particle.F_total ~ F_lorentz)
    end

    System(eqs, τ; systems, name)
end

# Convenience constructors for backward compatibility
@component function ClassicalElectron(;
    name,
    spacetime = Spacetime(c = 1, name = :spacetime),
    laser = PlaneWave(; spacetime, name = :laser),
)
    ChargedParticle(; name, spacetime, external_field = laser, radiation_model = nothing)
end

@component function RadiatingElectron(;
    name,
    spacetime = Spacetime(c = 1, name = :spacetime),
    laser = PlaneWave(; spacetime, name = :laser),
)
    ChargedParticle(;
        name,
        spacetime,
        external_field = laser,
        radiation_model = :abraham_lorentz,
    )
end

@component function LandauLifshitzElectron(;
    name,
    spacetime = Spacetime(c = 1, name = :spacetime),
    laser = PlaneWave(; spacetime, name = :laser),
)
    ChargedParticle(;
        name,
        spacetime,
        external_field = laser,
        radiation_model = :landau_lifshitz,
    )
end
