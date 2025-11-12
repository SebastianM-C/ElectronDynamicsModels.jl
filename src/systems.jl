@component function ChargedParticle(;
    name,
    mass = 1.0,
    charge = 1.0,
    ref_frame,
    external_field,
    radiation_model = nothing,
)
    iv = ModelingToolkit.get_iv(ref_frame)
    @named particle = ParticleDynamics(; mass, ref_frame)
    @unpack x, u, F_total = particle

    @unpack gμν = ref_frame
    Fμν = ParentScope(external_field.field.Fμν)
    J = ParentScope(external_field.field.J)

    @parameters q = charge
    @variables F_lorentz(iv)[1:4]

    systems = [external_field, ref_frame]

    eqs = [
        # Connect particle position and time to external field
        external_field.x ~ x

        # Lorentz force
        F_lorentz ~ q * (Fμν * gμν * u)
        # Set 4-current density in electromagnetic system
        J ~ q * u
    ]

    if nameof(iv) == :τ
        @unpack t = particle

        push!(eqs, external_field.t ~ t)
    end

    # Handle radiation reaction if specified
    if radiation_model == :landau_lifshitz
        @named radiation =
            LandauLifshitzRadiation(; charge, F_lorentz_ref = F_lorentz, ref_frame, particle)
        push!(systems, radiation)
        push!(eqs, external_field.field_dynamics.Fμν ~ radiation.field.Fμν)
        push!(eqs, F_total ~ F_lorentz + radiation.F_rad)
    elseif radiation_model == :abraham_lorentz
        @named radiation =
            AbrahamLorentzRadiation(; charge, F_lorentz_ref = F_lorentz, ref_frame, particle)
        push!(systems, radiation)
        push!(eqs, F_total ~ F_lorentz + radiation.F_rad)
    else
        # No radiation reaction
        push!(eqs, F_total ~ F_lorentz)
    end

    sys = System(eqs, iv; systems, name)
    extend(sys, particle)
end

# Convenience constructors for backward compatibility
@component function ClassicalElectron(;
    name,
    ref_frame = ReferenceFrame(c = 1, ε₀ = 1, μ₀ = 1, name = :ref_frame),
    laser = PlaneWave(; ref_frame, name = :laser),
)
    ChargedParticle(; name, ref_frame, external_field = laser, radiation_model = nothing)
end

@component function RadiatingElectron(;
    name,
    ref_frame = ReferenceFrame(c = 1, ε₀ = 1, μ₀ = 1, name = :ref_frame),
    laser = PlaneWave(; ref_frame, name = :laser),
)
    ChargedParticle(;
        name,
        ref_frame,
        external_field = laser,
        radiation_model = :abraham_lorentz,
    )
end

@component function LandauLifshitzElectron(;
    name,
    ref_frame = ReferenceFrame(c = 1, ε₀ = 1, μ₀ = 1, name = :ref_frame),
    laser = PlaneWave(; ref_frame, name = :laser),
)
    ChargedParticle(;
        name,
        ref_frame,
        external_field = laser,
        radiation_model = :landau_lifshitz,
    )
end
