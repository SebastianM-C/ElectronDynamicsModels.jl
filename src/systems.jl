"""
    _find_world(sys)

Find the `Worldline` subsystem within an external field by checking for the
metric tensor `gμν` and electron mass `m_e` parameters.
"""
function _find_world(sys::AbstractSystem)
    if isempty(get_systems(sys)) && ModelingToolkitBase.has_parent(sys)
        return _find_world(ModelingToolkitBase.get_parent(sys))
    end
    for s in get_systems(sys)
        s isa AbstractSystem || continue
        names = getname.(parameters(s))
        if :gμν ∈ names && :m_e ∈ names
            return s
        elseif !isempty(get_systems(s))
            return _find_world(s)
        end
    end
    error("No Worldline subsystem found in $(nameof(sys))")
end

function field_gradient(external_field, x, c)
    # Resolve E, B as explicit functions of x using the field's OWN (un-namespaced)
    # equations so the fixpoint_sub keys match. `external_field` (still namespaced)
    # is kept to restore the parameter namespace afterwards.
    ef = toggle_namespacing(external_field, false)
    eqs = full_equations(ef)
    sub = Dict(eq.lhs => eq.rhs for eq in eqs)
    sub[ef.t] = x[1] / c

    # resolve the full expressions for E and B
    E = SymbolicT[]
    B = SymbolicT[]
    for i in 1:3
        push!(E, Symbolics.fixpoint_sub(ef.E[i], sub))
        push!(B, Symbolics.fixpoint_sub(ef.B[i], sub))
    end
    ∂E = Vector{SymbolicT}[]
    ∂B = Vector{SymbolicT}[]
    for ν in 1:4
        ∂E_i = SymbolicT[]
        ∂B_i = SymbolicT[]
        for i in 1:3
            push!(∂E_i, expand_derivatives(Differential(x[ν])(E[i])))
            push!(∂B_i, expand_derivatives(Differential(x[ν])(B[i])))
        end
        push!(∂E, ∂E_i)
        push!(∂B, ∂B_i)
    end

    ∂Fμν = Matrix{SymbolicT}[]
    for ν in 1:4
        push!(∂Fμν, faraday(∂E[ν], ∂B[ν], c))
    end

    # The toggled symbols above are bare. Restore the field PARAMETERS to their
    # namespaced form (so they keep their values/bindings and don't clash with
    # particle variables such as the momentum p(τ) or mass m). The coordinate x
    # is left bare so it unifies with the particle's 4-position.
    leaves = SymbolicT[]
    for Mtx in ∂Fμν, e in Mtx
        append!(leaves, Symbolics.get_variables(e))
    end
    paramback = Dict(v => ModelingToolkit.renamespace(external_field, v)
                     for v in unique(leaves) if !occursin("x(", string(v)))
    ∂Fμν = [Symbolics.substitute.(Mtx, Ref(paramback)) for Mtx in ∂Fμν]

    return ∂Fμν
end

@component function ChargedParticle(;
        name,
        external_field,
        mass = nothing,
        charge = nothing,
        radiation_model = nothing,
    )
    world = _find_world(external_field)
    @unpack c, m_e, q_e, gμν, ε₀ = world
    iv = ModelingToolkit.get_iv(world)

    if isnothing(mass)
        mass = m_e
    end
    if isnothing(charge)
        charge = abs(q_e)
    end

    @named particle = ParticleDynamics(; mass, world)
    @unpack x, u, F_total = particle

    Fμν = ParentScope(external_field.Fμν)

    # τₑ = r_e/c = q²/(4πε₀mc³): classical-electron-radius timescale. A constant
    # property of the charge (not trajectory-dependent), so it is a bound
    # parameter, not a variable. Defined here rather than in the RR component so
    # ClassicalElectron can also report radiated power.
    @parameters q = charge τₑ = q^2 / (4π * ε₀ * mass * c^3)
    @variables F_lorentz(iv)[1:4] P_rad(iv)

    systems = [external_field]

    eqs = [
        # Connect particle position and time to external field
        external_field.x ~ x

        # Lorentz force
        F_lorentz ~ q * (Fμν * gμν * u)

        # Larmor radiated power. Kinematic — any accelerating charge radiates —
        # so it is defined for ClassicalElectron and LandauLifshitzElectron
        # alike. Evaluated on the FULL 4-force (F_total = m·duᵘ/dτ) so the energy
        # balance closes at O(τₑ²) when radiation reaction is on. The leading
        # minus is the textbook covariant Larmor sign: the 4-acceleration is
        # space-like in (+,−,−,−) (it is ⊥ to the time-like u, since F_total is),
        # so m_dot(F_total, F_total) ≤ 0 and negating yields a non-negative power.
        P_rad ~ -(2 / 3) * τₑ * m_dot(F_total, F_total) / mass
    ]

    if nameof(iv) == :τ
        @unpack t = particle

        push!(eqs, external_field.t ~ t)
    end

    # Handle radiation reaction if specified
    if radiation_model == :landau_lifshitz
        # Field gradient ∂_ν F^μλ, computed symbolically at the field boundary
        # (where Fμν is an explicit function of x) and handed to the RR component.
        ∂Fμν = field_gradient(external_field, toggle_namespacing(external_field, false).x, c)
        @named radiation = LandauLifshitzRadiation(;
            charge, Fμν_ref = Fμν, ∂Fμν_ref = ∂Fμν, u_ref = u, τₑ_ref = τₑ, world
        )
        push!(systems, radiation)
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
