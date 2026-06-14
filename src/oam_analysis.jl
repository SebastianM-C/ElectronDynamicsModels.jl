# Azimuthal-phase diagnostics on a frequency-domain field slice: read a complex field on a
# test circle (`ring_pixels`) and fit its azimuthal winding (`phase_winding_fit`). The fit's
# slope is a continuous estimate of the topological charge ℓ (= d∠F/dφ), its intercept the
# absolute phase offset b. Pure numerics (LinearAlgebra) — the CairoMakie rendering that uses
# these lives in the plotting extension; `scripts/compare_lpwa_vs_thomson.jl` uses them too.

"""
    ring_pixels(x_grid, y_grid, R; tol) -> (idxs, az)

Grid pixels whose distance from the centre lies within `tol` of radius `R` (a thin annulus
about the grid origin), as a `Vector{CartesianIndex{2}}` **sorted by azimuth**, together with
those azimuths `az = atan(y, x) ∈ (-π, π]`. `x_grid`/`y_grid` must be centred on the beam axis.
Reads a field/phase on a test circle for [`phase_winding_fit`](@ref); shared by the phase-ring
plots and the LPWA-vs-numeric comparison so both sample identical pixels.
"""
function ring_pixels(x_grid, y_grid, R; tol)
    CI = CartesianIndices((length(x_grid), length(y_grid)))
    idxs = filter(ci -> abs(hypot(x_grid[ci[1]], y_grid[ci[2]]) - R) < tol, CI)
    az = [atan(y_grid[ci[2]], x_grid[ci[1]]) for ci in idxs]
    o = sortperm(az)
    return idxs[o], az[o]
end

"""
    phase_winding_fit(az, phase; weights = nothing) -> (; slope, intercept, unwrapped)

Least-squares fit of the azimuthal phase winding: model `phase ≈ slope·az + intercept`, with
`az` a vector of azimuths sorted ascending (as from [`ring_pixels`](@ref)) and `phase` the
wrapped phase `angle(F)` at those points.

Because `angle` wraps to `(-π, π]`, a winding field reads as a **sawtooth**, so a naive fit on
the raw series returns slope ≈ 0 regardless of the true winding. The phase is therefore first
**unwrapped** along `az` (adding ±2π at jumps larger than π) before fitting — then the `slope`
recovers the winding `ℓ = d∠F/dφ` and `intercept` the absolute phase offset `b`. `weights`
(e.g. `abs.(F)` on the ring) down-weight phase nodes where the angle is ill-defined; `nothing`
⇒ uniform. Returns the fitted `slope`, `intercept`, and the `unwrapped` series (for plotting).
"""
function phase_winding_fit(az::AbstractVector, phase::AbstractVector; weights = nothing)
    n = length(az)
    n == length(phase) || throw(ArgumentError("az ($n) and phase ($(length(phase))) length mismatch"))
    w = weights === nothing ? ones(n) : collect(float.(weights))

    unwrapped = zeros(n)
    unwrapped[1] = phase[1]

    m = 0
    for k in 2:n
        d = phase[k] - phase[k - 1]
        if d < -π
            m += 1
        elseif d > π
            m -= 1
        end
        unwrapped[k] = phase[k] + 2π * m
    end

    A = [az ones(n)]
    coeffs = (sqrt.(w) .* A) \ (sqrt.(w) .* unwrapped)
    slope, intercept = coeffs

    return (; slope, intercept, unwrapped)
end
