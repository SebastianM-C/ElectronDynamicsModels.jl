# Incoherent far-field angular energy: dW/dΩ summed over electrons in intensity, not in field.
#
# The field kernels sum every electron's Liénard–Wiechert field on one shared observer clock (a
# coherent cube, 48 B per sample per pixel), and a wide angular map at a high Doppler upshift needs
# terabytes of it. For an incoherent observable — the γ-ray angular distribution of an ensemble with
# random phases — each electron's radiated energy per solid angle can be integrated along its own
# worldline and summed; no observer clock, no window, O(directions) memory.

"""
    far_field_directions(θx, θy; zsign = 1, tilt = 0.0) -> Matrix{SVector{3,Float64}}

Unit vectors `n̂ = normalize(tan θx, tan θy, zsign)` on the grid `θx × θy` (radians): the directions
of the pixels of a flat screen normal to ẑ at distance `Z` on the `zsign` side, seen from the
origin (`x = Z tan θx`, `y = Z tan θy`, as `ObserverScreen` pixels map to angles). `zsign` follows
`EDM_SCREEN_ZSIGN`. `tilt` (rad) rotates the whole grid about ŷ, so its centre points along
(zsign·sin tilt, 0, zsign·cos tilt): the grid of an electron beam moving at `tilt` from ±ẑ in the x–z plane.
"""
function far_field_directions(θx, θy; zsign = 1, tilt = 0.0)
    dirs = [normalize(SVector{3, Float64}(tan(a), tan(b), zsign)) for a in θx, b in θy]
    iszero(tilt) && return dirs
    s, c = sincos(tilt)
    return [SVector(c * n[1] + s * n[3], n[2], -s * n[1] + c * n[3]) for n in dirs]
end

"""
    angular_energy(trajs, dirs; c, ε₀, oversample = 1, τspan = nothing) -> Array{Float64}

Radiated energy per unit solid angle along each unit vector in `dirs`, summed INCOHERENTLY over
the electrons `trajs` (Jackson 14.38 integrated over the emission time t′):

    dW/dΩ(n̂) = ε₀ c K² Σₑ ∫ |n̂ × ((n̂ − β) × β̇)|² / (1 − n̂·β)⁵ dt′,     K = q/(4π ε₀ c),

the far-field limit of `R² ε₀ c ∫ |E_far|² dt` over observer time for the `E_far` of
[`accumulate_field`](@ref) (same `K`, same units: energy per steradian). β = u⃗/u⁰ and
β̇ = dβ/dt′ = (𝔞⃗ − β 𝔞⁰)/(u⁰ γ) come from each trajectory's 4-velocity and its `a_itp`, the spline
through the model's own right-hand side at the saved states (exact at the knots, O(h⁴) between
them; for a radiation-reaction model it includes that force) — no derivative of the state spline.

The emission-time integral runs in proper time (dt′ = γ dτ), trapezoidal on the trajectory knots
subdivided `oversample` times (each electron's own grid), over its whole span or `τspan = (τa, τb)`.
Directions are arbitrary unit vectors (no axis is assumed); `dirs` keeps its shape in the result.
Threads over directions, electron by electron.
"""
function angular_energy(trajs::AbstractVector{<:TrajectoryInterpolant}, dirs::AbstractArray{<:SVector{3}};
        c, ε₀, oversample::Integer = 1, τspan = nothing)
    oversample >= 1 || throw(ArgumentError("oversample must be ≥ 1, got $oversample"))
    D = vec(collect(SVector{3, Float64}, dirs))
    W = zeros(length(D))
    βs = SVector{3, Float64}[]
    bs = SVector{3, Float64}[]
    for traj in trajs
        τs = _emission_grid(traj.itp.t, oversample, τspan)
        _emission_samples!(βs, bs, traj, τs)
        Threads.@threads :static for d in eachindex(D)
            @inbounds W[d] += _direction_sum(D[d], βs, bs)
        end
    end
    K = first(trajs).K
    return reshape(ε₀ * c^2 * K^2 .* W, size(dirs))
end

angular_energy(trajs, θx::AbstractVector, θy::AbstractVector; zsign = 1, kw...) =
    angular_energy(trajs, far_field_directions(θx, θy; zsign); kw...)

# Knots subdivided `os` times, clipped to `span`.
function _emission_grid(knots, os, span)
    τs = Float64[]
    for j in 1:(length(knots) - 1), i in 0:(os - 1)
        push!(τs, knots[j] + (knots[j + 1] - knots[j]) * i / os)
    end
    push!(τs, last(knots))
    span === nothing && return τs
    return filter(τ -> span[1] <= τ <= span[2], τs)
end

# Per sample: β and β′√(w/u⁰), β′ = dβ/dτ, w the trapezoid weight in τ — so that
# Σ |(n̂−β)(n̂·b) − b(1−n̂·β)|²/(1−n̂·β)⁵ is ∫ |n̂×((n̂−β)×β′)|²/((1−n̂·β)⁵ u⁰) dτ; ε₀c²K² does the rest.
function _emission_samples!(βs, bs, traj, τs)
    n = length(τs)
    resize!(βs, n)
    resize!(bs, n)
    Threads.@threads :static for k in 1:n
        _, uμ, 𝔞μ = state_with_acceleration(traj, τs[k])
        u⁰ = uμ[1]
        β = SVector(uμ[2], uμ[3], uμ[4]) / u⁰
        β′ = (SVector(𝔞μ[2], 𝔞μ[3], 𝔞μ[4]) - β * 𝔞μ[1]) / u⁰
        w = ((k > 1 ? τs[k] - τs[k - 1] : 0.0) + (k < n ? τs[k + 1] - τs[k] : 0.0)) / 2
        @inbounds βs[k] = β
        @inbounds bs[k] = β′ * sqrt(w / u⁰)
    end
    return nothing
end

function _direction_sum(n̂, βs, bs)
    acc = 0.0
    @inbounds for k in eachindex(βs)
        β, b = βs[k], bs[k]
        s = 1 - n̂ ⋅ β
        v = (n̂ - β) * (n̂ ⋅ b) - b * s
        acc += (v ⋅ v) / (s * s * s * s * s)
    end
    return acc
end

"""
    one_over_e_halfwidths(W, θx, θy) -> (; x, y, r_eq, peak)

Angular half-widths of the map `W[θx, θy]` at 1/e of its peak: `x`/`y` are half the full width
of the level crossing along the cuts through the peak (linear interpolation between samples;
`NaN` if the level is not crossed inside the grid), `r_eq = √(A/π)` with `A` the solid-angle area
where `W ≥ peak/e` (pixel count × pixel area; a round spot's radius). Units of `θx`, `θy`.
"""
function one_over_e_halfwidths(W, θx, θy)
    peak, I = findmax(W)
    i0, j0 = Tuple(I)
    lvl = peak / ℯ
    edge(v, θ, i0, step) = begin
        i = i0
        while 1 <= i + step <= length(v) && v[i + step] >= lvl
            i += step
        end
        1 <= i + step <= length(v) || return NaN
        θ[i] + (θ[i + step] - θ[i]) * (v[i] - lvl) / (v[i] - v[i + step])
    end
    wx = (edge(view(W, :, j0), θx, i0, 1) - edge(view(W, :, j0), θx, i0, -1)) / 2
    wy = (edge(view(W, i0, :), θy, j0, 1) - edge(view(W, i0, :), θy, j0, -1)) / 2
    A = count(>=(lvl), W) * (θx[end] - θx[1]) / (length(θx) - 1) * (θy[end] - θy[1]) / (length(θy) - 1)
    return (; x = wx, y = wy, r_eq = sqrt(A / π), peak)
end
