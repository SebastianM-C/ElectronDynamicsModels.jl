function ReferenceFrame(iv; name, c, ε₀, μ₀, m_e, q_e)
    @parameters gμν[1:4, 1:4] = diagm([1, -1, -1, -1])
    constants = @constants c=c ε₀=ε₀ μ₀=μ₀ m_e=m_e q_e=q_e
    System(Equation[], iv, [], GlobalScope.([constants..., gμν]); name)
end

# struct Constants{T <: Real}
#     c::T
#     ε₀::T
#     μ₀::T
#     m_e::T
#     q_e::T
# end

# @symstruct Constants{T}

function ReferenceFrame(iv, units::Symbol; name)

    if units == :SI
        # SI unit constants
        c = 299792458.0           # m/s - speed of light
        ε₀ = 8.854187817e-12      # F/m - vacuum permittivity
        μ₀ = 1.25663706212e-6     # H/m - vacuum permeability (4π × 10⁻⁷)
        m_e = 9.1093837015e-31    # kg - electron mass
        q_e = -1.602176634e-19    # C - electron charge
    elseif units == :atomic
        # Atomic unit constants (Hartree atomic units)
        c = 137.03599908330932    # α⁻¹ - speed of light in a.u.
        m_e = 1.0                 # a.u. - electron mass
        q_e = -1.0                # a.u. - electron charge

        # Derived constants in atomic units
        # ε₀ = 1/(4π) in Gaussian-like a.u., but for consistency with q_e²/(2αℏc):
        # (Note: ℏ = 1 in atomic units, so h = 2π)
        h = 2π                    # Planck constant in a.u. (since ℏ = 1)
        α = 1/c                   # fine structure constant
        ε₀ = q_e^2 / (2α * h * c) # vacuum permittivity in a.u.
        μ₀ = 1 / (ε₀ * c^2)       # vacuum permeability in a.u.
    elseif units == :natural
        # Natural units (c = ℏ = m_e = e = 1)
        c = 1.0                   # speed of light
        m_e = 1.0                 # electron mass
        q_e = -1.0                # electron charge (negative for electron)

        # Derived constants in natural units
        # α = e²/(4πε₀) ≈ 1/137 (with c = ℏ = 1)
        α = 1/137.03599908330932  # fine structure constant
        ε₀ = abs(q_e)^2 / (4π * α)  # vacuum permittivity
        μ₀ = 1 / (ε₀ * c^2)       # vacuum permeability (= 1 with c=1)
    else
        error("$units not supported!")
    end

    # @parameters gμν[1:4, 1:4] = diagm([1, -1, -1, -1])
    # @parameters constants = Constants(c, ε₀, μ₀, m_e, q_e)

    # System(Equation[], iv, [], GlobalScope.([constants, gμν]); name)

    ReferenceFrame(iv; name, c, ε₀, μ₀, m_e, q_e)
end

function ProperFrame(units::Symbol; name)
    @independent_variables τ

    ReferenceFrame(τ, units; name)
end

function LabFrame(units::Symbol; name)
    @independent_variables t
    # t = ModelingToolkit.t_nounits

    ReferenceFrame(t, units; name)
end

function ElectromagneticSystem(iv; name)
    @variables begin
        Fμν(iv)[1:4, 1:4], [description = "Faraday tensor"]
        T(iv)[1:4, 1:4], [description="Stress-energy tensor T^μν"]
        J(iv)[1:4], [description="4-current density"]
    end
    System(Equation[], iv, [Fμν, T, J], []; name)
end
