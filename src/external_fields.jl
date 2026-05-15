"""
Gaussian laser pulse electromagnetic field.

Represents a focused Gaussian beam with a temporal envelope.
The beam propagates along the z-direction with waist w₀ at focus.

Parameters:
- λ: wavelength
- a₀: normalized vector potential (a0 kwarg)
- w₀: beam waist (defaults to 75λ)
- n_cycles: number of optical cycles to pulse center (determines t₀)
- τ0: temporal envelope half-width
- t₀: pulse center time (= n_cycles × 2π/ω)
- z₀: focus position along z-axis
"""
@component function GaussLaser(;
        name, wavelength = nothing, frequency = nothing,
        a0 = 10.0, beam_waist = nothing, polarization = :linear,
        n_cycles = 5, focus_position = nothing, world
    )
    if wavelength === nothing && frequency === nothing
        wavelength = 1.0
    end
    if wavelength !== nothing && frequency !== nothing
        error("Specify either wavelength or frequency, not both")
    end

    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; world)

    # Get spacetime variables and constants from parent scope
    @unpack c, m_e, q_e = world
    iv = ModelingToolkit.get_iv(world)

    # Create local position and time variables
    @variables x(iv)[1:4] t(iv)

    @unpack E, B = field_dynamics

    @parameters begin
        λ, [guess = 1.0, description = "Wavelength"]
        a₀ = a0, [description = "Normalized vector potential"]
        ω, [guess = 1.0, description = "Angular frequency"]
        k, [description = "Wave number (2π/λ)"]
        E₀, [guess = 1.0, description = "Peak electric field amplitude"]
        w₀, [description = "Beam waist radius at focus"]
        z_R, [description = "Rayleigh length"]
        τ0, [description = "Temporal envelope half-width"]
        n_cycles = n_cycles, [description = "Number of optical cycles to pulse center"]
        t₀, [guess = 1.0, description = "Pulse center time"]
        z₀ = 0, [description = "Focus position along z-axis"]
    end

    # Fixed parameters
    if polarization == :linear
        ξx = 1.0 + 0im
        ξy = 0 + 0im
    elseif polarization == :circular
        ξx, ξy = (1 / √2, im / √2) .|> complex
    else
        error("polarization $polarization not supported.")
    end

    ϕ₀ = 0

    @variables wz(iv) z(iv) r(iv)

    env = exp(-(((t - t₀) - (z - z₀) / c) / τ0)^2)

    E_g = E₀ * w₀ / wz *
        exp(-(r / wz)^2) *
        exp(im * (-(r^2 * z) / (z_R * wz^2) + atan(z, z_R) - k * z + ϕ₀)) *
        exp(im * ω * t) * env

    eqs = [
        # Extract z-coordinate from 4-position
        z ~ x[4]
        # Beam width as function of z
        wz ~ w₀ * √(1 + (z / z_R)^2)
        # Radial distance from beam axis
        r ~ hypot(x[2], x[3])

        # Electric field components
        E[1] ~ real(ξx * E_g)
        E[2] ~ real(ξy * E_g)
        E[3] ~ real(
            2im / (k * wz^2) * (1 + im * (z / z_R)) *
                (x[2] * (ξx * E_g) + x[3] * (ξy * E_g)),
        )

        # Magnetic field components
        B[1] ~ -E[2] / c
        B[2] ~ E[1] / c
        B[3] ~ real(
            2im / (k * c * wz^2) *
                (1 + im * (z / z_R)) *
                (x[3] * (ξx * E_g) - x[2] * (ξy * E_g)),
        )
    ]

    initialization_eqs = [
        ω ~ 2π * c / λ
        E₀ ~ a₀ * m_e * c * ω / abs(q_e)
        z_R ~ w₀^2 * k / 2
        k ~ 2π / λ
        t₀ ~ n_cycles * 2π / ω
    ]
    bindings = [
        ω => missing
        λ => missing
        k => missing
        E₀ => missing
        z_R => missing
        w₀ => missing
        t₀ => missing
    ]

    initial_conditions = Pair{SymbolicT, Any}[τ0 => 10 / ω]
    push!(initial_conditions, w₀ => (beam_waist === nothing ? 75λ : beam_waist))
    if focus_position !== nothing
        push!(initial_conditions, z₀ => focus_position)
    end
    if wavelength !== nothing
        push!(initial_conditions, λ => wavelength)
    end
    if frequency !== nothing
        push!(initial_conditions, ω => frequency)
    end

    sys = System(
        eqs, iv, [x, t, wz, z, r], [λ, a₀, ω, k, E₀, w₀, z_R, τ0, n_cycles, t₀, z₀];
        name,
        systems = [world],
        initial_conditions,
        initialization_eqs,
        bindings
    )

    extend(sys, field_dynamics)
end


"""
Plane wave electromagnetic field.

For a plane wave with normalized vector potential a₀ = eA/(mc²),
electrons can exhibit figure-8 motion when a₀ ~ 1.

Reference: Sarachik & Schappert, Phys. Rev. D 1, 2738 (1970)
"""
@component function PlaneWave(; name, amplitude = 1.0, wavelength = nothing, frequency = nothing, k_direction = [0, 0, 1], polarization = [1, 0, 0], world)
    if wavelength === nothing && frequency === nothing
        frequency = 1.0
    end
    if wavelength !== nothing && frequency !== nothing
        error("Specify either wavelength or frequency, not both")
    end

    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; world)

    # Get spacetime variables and constants from parent scope
    @unpack c, m_e, q_e = world
    iv = ModelingToolkit.get_iv(world)

    # Create local position and time variables
    @variables x(iv)[1:4]
    if nameof(iv) == :τ
        @variables t(iv)
    else
        t = iv
    end

    @unpack E, B = field_dynamics

    @parameters begin
        A = amplitude, [description = "Field amplitude"]
        ω, [guess = 1.0, description = "Angular frequency"]
        k_dir[1:3] = k_direction, [description = "Wave vector direction"]
        pol[1:3] = polarization, [description = "Polarization direction"]
        λ, [guess = 1.0, description = "Wavelength"]
    end

    # Normalize k direction to get unit vector k̂
    k_norm = sqrt(k_dir[1]^2 + k_dir[2]^2 + k_dir[3]^2)
    k̂ = [k_dir[1] / k_norm, k_dir[2] / k_norm, k_dir[3] / k_norm]

    # Normalize polarization vector (user must ensure it's perpendicular to k)
    pol_norm = sqrt(pol[1]^2 + pol[2]^2 + pol[3]^2)
    ê = [pol[1] / pol_norm, pol[2] / pol_norm, pol[3] / pol_norm]

    # B-field direction: k̂ × ê
    b̂ = [
        k̂[2] * ê[3] - k̂[3] * ê[2],
        k̂[3] * ê[1] - k̂[1] * ê[3],
        k̂[1] * ê[2] - k̂[2] * ê[1],
    ]

    # Spatial position from 4-position (x = [ct, x, y, z] or [t, x, y, z])
    x⃗ = [x[2], x[3], x[4]]

    # Phase: k·r - ωt where |k| = ω/c (dispersion relation)
    phase = (ω / c) * (k̂[1] * x⃗[1] + k̂[2] * x⃗[2] + k̂[3] * x⃗[3]) - ω * t

    eqs = [
        E[1] ~ A * ê[1] * cos(phase)
        E[2] ~ A * ê[2] * cos(phase)
        E[3] ~ A * ê[3] * cos(phase)
        B[1] ~ (A / c) * b̂[1] * cos(phase)
        B[2] ~ (A / c) * b̂[2] * cos(phase)
        B[3] ~ (A / c) * b̂[3] * cos(phase)
    ]

    initialization_eqs = [
        λ ~ (2π * c) / ω,
    ]
    bindings = [
        λ => missing
        ω => missing
    ]

    initial_conditions = Pair{SymbolicT, Any}[]
    if wavelength !== nothing
        push!(initial_conditions, λ => wavelength)
    end
    if frequency !== nothing
        push!(initial_conditions, ω => frequency)
    end

    vars = nameof(iv) == :τ ? [x, t] : [x]

    sys = System(eqs, iv; name, systems = [world], initialization_eqs, bindings, initial_conditions)

    extend(sys, field_dynamics)
end

"""
Uniform electromagnetic field component.

In crossed E and B fields with E⊥B and |E| < |B|c,
particles drift with velocity v_drift = E×B/B²

Reference: Jackson, "Classical Electrodynamics", Section 12.4
"""
@component function UniformField(; name, E_field = [0, 0, 1], B_field = [0, 0, 0], world)
    @named field_dynamics = EMFieldDynamics(; world)
    @unpack E, B = field_dynamics

    # Get spacetime variables and constants from parent scope
    @unpack c, m_e, q_e = world
    iv = ModelingToolkit.get_iv(world)

    # Create local position and time variables
    @variables x(iv)[1:4] t(iv)

    @parameters begin
        E₀[1:3] = E_field, [description = "Electric field vector"]
        B₀[1:3] = B_field, [description = "Magnetic field vector"]
    end

    eqs = [
        E ~ E₀,
        B ~ B₀,
    ]

    sys = System(eqs, iv, [x, t], [E₀, B₀]; name, systems = [world])

    extend(sys, field_dynamics)
end

"""
    _kz_sign(k_direction) -> ±1

Validate that `k_direction` is along ±ẑ and return its sign. Currently only ±ẑ
propagation is supported for `LaguerreGaussLaser`; oblique directions require
defining a transverse basis convention that has not yet been implemented.
"""
function _kz_sign(k_direction)
    length(k_direction) == 3 || error("k_direction must be a 3-vector, got length $(length(k_direction))")
    isapprox(norm(k_direction), 1; atol=1e-10) || error("k_direction must be a unit vector, got norm $(norm(k_direction))")
    if k_direction ≈ [0, 0, 1]
        return +1
    elseif k_direction ≈ [0, 0, -1]
        return -1
    else
        error("LaguerreGaussLaser currently supports only k_direction ≈ ±ẑ; got $k_direction. Pass [0,0,1] for +ẑ or [0,0,-1] for -ẑ.")
    end
end

"""
    a0_from_pulse_energy(W, w₀, τ₀, ω; world, mode = (p = 0, m = 2)) -> a₀

Compute the dimensionless vector potential `a₀` for a Laguerre-Gauss pulse of
total energy `W`, host-beam waist `w₀`, field-envelope half-width `τ₀`, and
angular frequency `ω`. All inputs in the unit system of `world` (atomic, SI,
or natural). `mode = (p, m)` selects the LG mode; the formula is
`p`-independent (in EDM's normalization), so only `|m|` matters in practice.

Formula:
    a₀² = 2W·|q_e|² / (ε₀ c³ A(p,m) (m_e ω)² w₀² τ₀ √(π/2))
with A(p,m) = (π/2)·(|m|!)².

Use this script-side when `u0_constructor` (e.g. `SVector{8}`) precludes
`LaguerreGaussLaser`'s built-in `pulse_energy` initialization, since the
init sub-problem inherits the constructor with the wrong dimension. Pass the
result to `LaguerreGaussLaser(; a0 = …)` instead.

See `references/lg_pulse_energy_a0.tex` for the derivation.
"""
function a0_from_pulse_energy(W, w₀, τ₀, ω; world, mode = (p = 0, m = 2))
    m = mode.m
    ε₀      = getdefault(world.ε₀)
    c       = getdefault(world.c)
    m_e     = getdefault(world.m_e)
    q_e_abs = abs(getdefault(world.q_e))
    A_pm    = (π/2) * factorial(abs(m))^2

    # Sanity check on cycle-averaging validity (slow-envelope approximation).
    ωτ = ω * τ₀
    if ωτ < 5
        @warn "a0_from_pulse_energy assumes ωτ₀ ≫ 1 (slow envelope vs fast carrier)" ωτ suppression=exp(-ωτ^2/2)
    end

    a₀² = 2 * W * q_e_abs^2 /
          (ε₀ * c^3 * A_pm * (m_e * ω)^2 * w₀^2 * τ₀ * sqrt(π/2))
    return sqrt(a₀²)
end

"""
Laguerre-Gauss laser beam electromagnetic field.

Represents a focused beam with orbital angular momentum (OAM).
The beam is characterized by radial index p and azimuthal index m.

Parameters:
- λ: wavelength
- a₀: normalized vector potential (a0 kwarg)
- w₀: beam waist (defaults to 75λ)
- radial_index (p): radial mode number, p ≥ 0
- azimuthal_index (m): azimuthal mode number (orbital angular momentum)
- temporal_profile: :gaussian (pulsed) or :constant (CW)
- temporal_width: pulse width for gaussian profile (defaults to 100.0)
- focus_position: focal position along z-axis (defaults to 0.0)
- k_direction: propagation direction unit vector. Currently restricted to
  `[0, 0, 1]` (default, +ẑ) or `[0, 0, -1]` (−ẑ).

Reference: Allen et al., Phys. Rev. A 45, 8185 (1992)
"""
@component function LaguerreGaussLaser(;
        name,
        wavelength = nothing,
        frequency = nothing,
        a0 = nothing,
        pulse_energy = nothing,
        beam_waist = nothing,
        radial_index = 0,      # p
        azimuthal_index = 1,   # m
        world,
        temporal_profile = :gaussian,  # :gaussian or :constant
        temporal_width = nothing,      # pulse width (for gaussian profile)
        n_cycles = 0,                  # number of optical cycles to pulse center
        focus_position = nothing,       # focal position along z-axis
        polarization = :linear,
        k_direction = [0, 0, 1]
    )
    kz_sign = _kz_sign(k_direction)
    if wavelength === nothing && frequency === nothing
        wavelength = 1.0
    end
    if wavelength !== nothing && frequency !== nothing
        error("Specify either wavelength or frequency, not both")
    end

    # Validate field-strength specification: exactly one of a0 / pulse_energy
    a0_given = a0           !== nothing
    we_given = pulse_energy !== nothing
    if temporal_profile == :constant
        we_given && error("pulse_energy is not meaningful for temporal_profile = :constant (CW beams have no finite total energy). Use a0 instead.")
        a0_given || (a0 = 10.0; a0_given = true)  # backward-compat default for CW
    else
        if !(a0_given ⊻ we_given)
            error("Specify exactly one of `a0` or `pulse_energy` (got a0 = $a0, pulse_energy = $pulse_energy)")
        end
    end

    # When pulse_energy is given, the W ↔ a₀ relation assumes the slow-envelope
    # approximation ωτ₀ ≫ 1 (cycle-averaged intensity). For few-cycle pulses
    # this breaks down — see references/lg_pulse_energy_a0.tex §2.4.
    if we_given && temporal_profile == :gaussian && temporal_width !== nothing
        c_val = getdefault(world.c)
        ω_estimate = wavelength !== nothing ? 2π * c_val / wavelength :
                     frequency  !== nothing ? frequency : nothing
        if ω_estimate !== nothing
            ωτ = ω_estimate * temporal_width
            if ωτ < 5
                @warn "pulse_energy formula assumes ωτ₀ ≫ 1; got ωτ₀ = $ωτ. Fast Fourier-suppression term e^(-(ωτ₀)²/2) ≈ $(exp(-ωτ^2/2)) is no longer negligible. For few-cycle pulses, derive a0 from pulse energy by integrating the carrier-explicit field directly."
            end
        end
    end

    # New interface with spacetime
    @named field_dynamics = EMFieldDynamics(; world)

    # Get spacetime variables from parent scope
    @unpack c, m_e, q_e, ε₀ = world
    iv = ModelingToolkit.get_iv(world)

    # Create local position and time variables
    @variables x(iv)[1:4]
    if nameof(iv) == :τ
        @variables t(iv)
    else
        t = iv
    end

    @unpack E, B = field_dynamics

    # Helper: compute |m|
    mₐ = abs(azimuthal_index)
    sgn = sign(azimuthal_index)

    # Compute normalization factor using Pochhammer symbol (rising factorial)
    # Nₚₘ = √((p+1)_{|m|}) where (x)_n is the Pochhammer symbol
    Npm_val = sqrt(pochhammer(radial_index + 1, mₐ))

    # Spatial-integral coefficient for LG_p^m: A(p, m) = (π/2) * (|m|!)²
    # Independent of p; see references/lg_pulse_energy_a0.tex for derivation.
    A_pm = (π/2) * factorial(abs(azimuthal_index))^2

    params = @parameters begin
        λ, [guess = 1.0, description = "Wavelength"]
        a₀, [guess = 1.0, description = "Normalized vector potential"]
        W, [guess = 1.0, description = "Total pulse energy"]
        ω, [guess = 1.0, description = "Angular frequency"]
        k, [description = "Wave number (2π/λ)"]
        E₀, [guess = 1.0, description = "Peak electric field amplitude"]
        w₀, [description = "Beam waist radius at focus"]
        z_R, [description = "Rayleigh length"]
        τ0 = temporal_width === nothing ? 100.0 : temporal_width, [description = "Temporal envelope half-width"]

        # Laguerre-Gauss quantum numbers
        p = radial_index, [description = "Radial mode index"]
        m = azimuthal_index, [description = "Azimuthal mode index (orbital angular momentum)"]

        # Normalization factor (computed from p and m)
        Nₚₘ = Npm_val, [description = "Normalization factor √((p+1)_{|m|})"]

        n_cycles = n_cycles, [description = "Number of optical cycles to pulse center"]
        t₀, [guess = 1.0, description = "Pulse center time"]
        z₀ = 0, [description = "Focus position along z-axis"]
    end

    if polarization == :linear
        ξx = 1.0 + 0im
        ξy = 0 + 0im
    elseif polarization == :circular
        ξx, ξy = (1 / √2, im / √2) .|> complex
    else
        error("polarization $polarization not supported.")
    end

    ϕ₀ = 0

    # Derived variables
    @variables begin
        wz(iv)      # Beam width
        z(iv)       # Propagation coordinate
        r(iv)       # Radial distance
        θ(iv)       # Azimuthal angle
        σ(iv)       # Normalized radial coordinate squared
        rwz(iv)     # Scaled radial coordinate
        env(iv)     # Temporal envelope factor
    end

    # Compute temporal envelope expression based on profile type
    # The traveling wave moves at speed c along k̂, so the envelope follows
    # `(t - t₀) - kz_sign·(z - z₀)/c`.
    env_expr = if temporal_profile == :constant
        1.0  # No temporal envelope
    elseif temporal_profile == :gaussian
        exp(-(((t - t₀) - kz_sign * (z - z₀) / c) / τ0)^2)
    else
        error("Unknown temporal_profile: $temporal_profile. Use :constant or :gaussian")
    end

    # We need to factor out the complex expressions so that we don't have equations with complex lhs
    # since MTK doesn't handle that well

    # Base Gaussian field (before polarization and LG modification).
    # Replacing z → kz_sign·z in the wavefront curvature, Gouy phase, and
    # carrier phase flips the sign of k̂ from +ẑ to -ẑ.
    E_g = E₀ * w₀ / wz *
        exp(-(r / wz)^2) *
        exp(im * (-(r^2 * kz_sign * z) / (z_R * wz^2) + atan(kz_sign * z, z_R) - k * kz_sign * z + ϕ₀)) *
        exp(im * ω * t) * env

    # Laguerre-Gauss phase factor (Gouy term acquires kz_sign for -ẑ propagation)
    lg_phase = exp(im * ((2p + mₐ) * atan(kz_sign * z, z_R) - m * θ + ϕ₀))

    # NEgexp term used in Ez and Bz
    NEgexp = Nₚₘ * E_g * lg_phase

    # Complex transverse field components (used in Ez, Bz)
    Ex_complex = ξx * E_g * Nₚₘ * rwz^mₐ * _₁F₁(-p, mₐ + 1, 2σ) * lg_phase
    Ey_complex = ξy * E_g * Nₚₘ * rwz^mₐ * _₁F₁(-p, mₐ + 1, 2σ) * lg_phase

    eqs = [
        # Extract coordinates from 4-position
        z ~ x[4]
        r ~ hypot(x[2], x[3])
        θ ~ atan(x[3], x[2])

        # Beam width as function of z
        wz ~ w₀ * √(1 + (z / z_R)^2)

        # Intermediate variables
        σ ~ (r / wz)^2
        rwz ~ r * √2 / wz

        # Temporal envelope
        env ~ env_expr

        # Electric field Ex component (Laguerre-Gauss)
        E[1] ~ real(Ex_complex)

        # Electric field Ey component
        E[2] ~ real(Ey_complex)

        # Electric field Ez component (longitudinal).
        # The longitudinal component points along k̂, so the lab z-component
        # picks up an overall kz_sign. The (1 + im·z/z_R) curvature factor
        # gets z → kz_sign·z to match the +ẑ-derived formula's analogue.
        E[3] ~ kz_sign * real(
            -im / k * (
                # Term 1: m-dependent term
                (
                    azimuthal_index == 0 ? 0.0 :
                        mₐ * (ξx - im * sgn * ξy) *
                        (√2 / wz)^mₐ * r^(mₐ - 1) * _₁F₁(-p, mₐ + 1, 2σ) *
                        NEgexp * exp(im * sgn * θ)
                ) -
                    # Term 2: transverse field coupling
                    2 / (wz^2) * (1 + im * kz_sign * z / z_R) * (x[2] * Ex_complex + x[3] * Ey_complex) -
                    # Term 3: p-dependent term
                    (
                    radial_index == 0 ? 0.0 :
                        4 * p / ((mₐ + 1) * wz^2) * (x[2] * ξx + x[3] * ξy) *
                        rwz^mₐ * _₁F₁(-p + 1, mₐ + 2, 2σ) * NEgexp
                )
            )
        )

        # Magnetic field components: B = (1/c) k̂ × E
        # For k̂ = +ẑ: (B_x, B_y) = (-E_y, +E_x)/c; flipping k̂ → -ẑ flips both.
        B[1] ~ -kz_sign * E[2] / c
        B[2] ~  kz_sign * E[1] / c

        # Bz component: longitudinal magnetic field along k̂; lab z-component
        # picks up an overall kz_sign analogously to Ez.
        B[3] ~ kz_sign * real(
            -im / ω * (
                # Term 1: m-dependent term
                -(
                    azimuthal_index == 0 ? 0.0 :
                        mₐ * (ξy + im * sgn * ξx) *
                        (√2 / wz)^mₐ * r^(mₐ - 1) * _₁F₁(-p, mₐ + 1, 2σ) *
                        NEgexp * exp(im * sgn * θ)
                ) +
                    # Term 2: transverse field coupling
                    2 / (wz^2) * (1 + im * kz_sign * z / z_R) * (x[2] * Ey_complex - x[3] * Ex_complex) +
                    # Term 3: p-dependent term
                    (
                    radial_index == 0 ? 0.0 :
                        4 * p / ((mₐ + 1) * wz^2) * (x[2] * ξy - x[3] * ξx) *
                        rwz^mₐ * _₁F₁(-p + 1, mₐ + 2, 2σ) * NEgexp
                )
            )
        )
    ]
    initialization_eqs = [
        ω ~ 2π * c / λ
        k ~ 2π / λ
        z_R ~ π * w₀^2 / λ
        E₀ ~ a₀ * m_e * c * ω / abs(q_e)
        # Pulse-energy ↔ a₀ relation (Gaussian envelope, cycle-averaged):
        #   W = (1/2) ε₀ c · A(p,m) w₀² E₀² · τ₀√(π/2).
        # Written in the `W ~ …` direction so MTK can substitute forward when
        # a₀ is given (no nonlinear init step, so problem-level constructors
        # like `u0_constructor = SVector{N}` don't propagate to a smaller
        # init sub-problem). The reverse direction (W given → solve for a₀)
        # involves the sqrt and would stall Newton's abstol; users wanting
        # `pulse_energy →  a₀` should call `a0_from_pulse_energy` script-side
        # and pass `a0 = …` explicitly. See references/lg_pulse_energy_a0.tex.
        W ~ (1/2) * ε₀ * c * A_pm * E₀^2 * w₀^2 * τ0 * sqrt(π/2)
        t₀ ~ n_cycles * 2π / ω
    ]
    bindings = [
        ω => missing
        λ => missing
        k => missing
        E₀ => missing
        z_R => missing
        w₀ => missing
        t₀ => missing
        z₀ => missing
        a₀ => missing
        W => missing
    ]

    initial_conditions = Pair{SymbolicT, Any}[]
    push!(initial_conditions, w₀ => (beam_waist === nothing ? 75λ : beam_waist))
    push!(initial_conditions, z₀ => (focus_position === nothing ? 0.0 : focus_position))
    if wavelength !== nothing
        push!(initial_conditions, λ => wavelength)
    end
    if frequency !== nothing
        push!(initial_conditions, ω => frequency)
    end
    # Field-strength specification: exactly one of a0 / pulse_energy
    if a0_given
        push!(initial_conditions, a₀ => a0)
    end
    if we_given
        push!(initial_conditions, W => pulse_energy)
    end

    vars = if nameof(iv) == :τ
        [x, t, z, r, θ, wz, σ, rwz, env]
    else
        @variables τ(iv)
        [x, τ, z, r, θ, wz, σ, rwz, env]
    end

    sys = System(
        eqs, iv, vars, params;
        name, systems = [world], initialization_eqs, bindings, initial_conditions
    )
    extend(sys, field_dynamics)
end
