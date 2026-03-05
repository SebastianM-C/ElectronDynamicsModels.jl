"""
    _find_ref_frame(external_field)

Find the reference frame subsystem within an external field by checking for
the metric tensor `gμν` and electron mass `m_e` parameters.
"""
function _find_ref_frame(external_field::AbstractSystem)
    for s in get_systems(external_field)
        s isa AbstractSystem || continue
        names = getname.(parameters(s))
        if :gμν ∈ names && :m_e ∈ names
            return s
        end
    end
    error("No reference frame found in $(nameof(external_field))")
end

@component function ChargedParticle(;
        name,
        external_field,
        mass = nothing,
        charge = nothing,
        radiation_model = nothing,
    )
    ref_frame = _find_ref_frame(external_field)
    @unpack c, m_e, q_e, gμν, ε₀ = ref_frame
    iv = ModelingToolkit.get_iv(ref_frame)

    if isnothing(mass)
        mass = m_e
    end
    if isnothing(charge)
        charge = abs(q_e)
    end

    @named particle = ParticleDynamics(; mass, ref_frame)
    @unpack x, u, F_total = particle

    Fμν = ParentScope(external_field.field.Fμν)
    J = ParentScope(external_field.field.J)

    @parameters q = charge
    @variables F_lorentz(iv)[1:4]

    systems = [external_field]

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
        push!(eqs, external_field.field.Fμν ~ radiation.field.Fμν)
        push!(eqs, F_total ~ F_lorentz + radiation.F_rad)
    elseif radiation_model == :abraham_lorentz
        @named radiation =
            AbrahamLorentzRadiation(; charge, F_lorentz_ref = F_lorentz, ref_frame, particle)
        push!(systems, radiation)
        push!(eqs, u ~ radiation.u)
        push!(eqs, F_total ~ F_lorentz + radiation.F_rad)
    else
        # No radiation reaction
        push!(eqs, F_total ~ F_lorentz)
    end

    sys = System(eqs, iv; systems, name)
    extend(sys, particle)
end

# Convenience constructors
@component function ClassicalElectron(;
        name,
        laser,
    )
    ChargedParticle(; name, external_field = laser, radiation_model = nothing)
end

@component function RadiatingElectron(;
        name,
        laser,
    )
    ChargedParticle(;
        name,
        external_field = laser,
        radiation_model = :abraham_lorentz,
    )
end

@component function LandauLifshitzElectron(;
        name,
        laser,
    )
    ChargedParticle(;
        name,
        external_field = laser,
        radiation_model = :landau_lifshitz,
    )
end
