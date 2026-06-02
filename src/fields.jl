"""
    faraday(E, B, c) -> SMatrix{4,4}

Upper-index Faraday tensor `F^{μν}` from the field 3-vectors, in the package's
(+,−,−,−) convention: `F^{0i} = −Eⁱ/c`, `F^{ij} = −ε^{ijk}Bᵏ`.  Returns a
StaticArrays matrix so the *same* assembly serves both the symbolic models (with
`Num` entries) and the numeric screen reduction.  Its inverse is [`extract_EB`](@ref).
"""
function faraday(E, B, c)
    return @SMatrix [
        0.0      -E[1]/c   -E[2]/c   -E[3]/c
        E[1]/c    0.0      -B[3]      B[2]
        E[2]/c    B[3]      0.0      -B[1]
        E[3]/c   -B[2]      B[1]      0.0
    ]
end

"""
    stress_energy(F, g, μ₀) -> SMatrix{4,4}

Electromagnetic stress-energy tensor `T^{μν}` from the upper-index Faraday tensor
`F`, metric `g`, and vacuum permeability `μ₀`, in the (+,−,−,−) convention
(`F^{0i} = −Eⁱ/c`):

    M = F g Fᵀ                       # M^{μν} = F^{μα} g_{αβ} F^{νβ}
    F_{αβ}F^{αβ} = tr(g M)           # = 2(B² − E²/c²)   (uses Fᵀ = −F)
    T^{μν} = (1/μ₀) [ −M^{μν} + ¼ g^{μν} (F_{αβ}F^{αβ}) ]

The sign is fixed so `T⁰⁰ = ½ε₀(E²+c²B²) > 0` (energy density) and `Sⁱ = c·T⁰ⁱ`
(Poynting).  Written with StaticArrays matrix ops so one function serves both the
symbolic `EMFieldDynamics` (via `@SMatrix` of `Num`) and the numeric reduction in
[`screen_observables`](@ref).

References: Landau & Lifshitz "Classical Theory of Fields" §33; Jackson §12.10.
"""
function stress_energy(F, g, μ₀)
    M = F * g * transpose(F)
    gM = g * M
    invariant = gM[1, 1] + gM[2, 2] + gM[3, 3] + gM[4, 4]   # tr(gM) = F_{αβ}F^{αβ}
    return (1 / μ₀) * (-M + (1 / 4) * g * invariant)
end

@component function EMFieldDynamics(; name, world)
    iv = ModelingToolkit.get_iv(world)

    @unpack c, μ₀, ε₀ = world

    @variables begin
        E(iv)[1:3]                                  # Electric field (3-vector)
        B(iv)[1:3]                                  # Magnetic field (3-vector)
        Fμν(iv)[1:4, 1:4], [description = "Faraday tensor"]
        T(iv)[1:4, 1:4], [description = "Stress-energy tensor T^μν"]
    end

    F = faraday(E, B, c)
    eqs = [
        # Faraday tensor and EM stress-energy tensor, the same `faraday` /
        # `stress_energy` used by the numeric screen reduction.  `η` is the
        # package-wide Minkowski metric.
        Fμν ~ F,
        T ~ stress_energy(F, η, μ₀),
    ]

    System(eqs, iv, [E, B, Fμν, T], [μ₀, ε₀]; name, systems = [world])
end
