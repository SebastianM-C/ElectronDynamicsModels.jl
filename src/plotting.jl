# Plotting API surface. The heavy CairoMakie implementations live in the package extension
# `ext/EDMPlotsExt.jl`, loaded automatically once CairoMakie is. These stubs make the names
# exist + exported so the extension can add methods and callers can reference them; the
# colorrange helpers are pure (no CairoMakie) so they're usable as plain functions too.

"""
    plot_harmonic_grid(maps, x_grid, y_grid; w₀=1, labels, title="", colormap=:jet,
                       colorrange=harmonic_colorrange, ncols=3, panelsize=300, outfile=nothing)

Draw a grid of per-component heat-maps for one harmonic — the real part of each component of
`maps`, which is `(n_components, Nx, Ny)` (e.g. a slice `fields_h[k, :, :, :]`). Axes are shown
in units of `w₀` (set it to the beam waist; left as `1` ⇒ raw coordinates, label "x"/"y").
`colormap` and `colorrange` are configurable: `colorrange` is any `data -> (lo, hi)` applied
per panel (default [`harmonic_colorrange`] = guarded extrema; pass [`symmetric_colorrange`] for a
diverging map). `ncols` sets panels-per-row (3 ⇒ 2×3 for E/B, 2 ⇒ 2×2 for the 4-potential).
Saves to `outfile` if given; returns the `Figure`.

**Requires CairoMakie** — `using CairoMakie` activates the implementation (package extension);
calling this without it raises a `MethodError`.
"""
function plot_harmonic_grid end

"""
    plot_power_spectrum(freqs, power_spec; ω, labels, colors=…, linestyles=nothing, title="", outfile=nothing)

Log-y plot of per-component power spectra `power_spec` (`(Nf, n)`, from [`power_spectrum`]) vs
`freqs`, x-axis in units of the fundamental ω₁ (= `ω/2π`), with dashed integer-harmonic markers
and a legend. Saves to `outfile` if given; returns the `Figure`.

**Requires CairoMakie** — `using CairoMakie` activates the implementation (package extension).
"""
function plot_power_spectrum end

"""
    plot_phase_grid(maps, x_grid, y_grid; w₀=1, labels, title="", ncols=3, panelsize=300, outfile=nothing)

Like [`plot_harmonic_grid`] but plots the **phase** `angle.(maps[c])` of each component on a
cyclic `:phase` colormap over `(-π, π)` — the `(x/w₀, y/w₀, ∠F)` view. The panel title still
reports each component's peak complex amplitude. **Requires CairoMakie** (package extension).
"""
function plot_phase_grid end

"""
    plot_phase_rings_grid(maps, x_grid, y_grid; w₀=1, labels, radii, tol, title="", ncols=3,
                          panelsize=300, outfile=nothing)

Per-component grid of the **azimuthal phase winding**: for each component of `maps`
(`(n_components, Nx, Ny)`), scatter `angle(F)` of the pixels on each test circle of radius
`R ∈ radii` (a thin annulus, half-width `tol`, about the grid centre) against their azimuth
`atan(y, x)`. A vortex of topological charge ℓ shows ℓ phase windings as φ runs once around —
the ring counterpart of [`plot_phase_grid`]'s heatmap. **Requires CairoMakie** (package extension).
"""
function plot_phase_rings_grid end

"""
    harmonic_colorrange(data) -> (lo, hi)

Default per-panel color range for harmonic maps: the data extrema, guarded against a
degenerate/underflowing panel (falls back to `(-1, 1)`). See also [`symmetric_colorrange`].
"""
function harmonic_colorrange(data)
    lo, hi = extrema(data)
    return (isfinite(hi) && Float32(hi - lo) > 0.0f0) ? (Float64(lo), Float64(hi)) : (-1.0, 1.0)
end

"""
    symmetric_colorrange(data) -> (-m, m)

A color range symmetric about zero (`m = maximum(abs, data)`), for diverging colormaps like
`:seismic`; guarded against a degenerate panel. Pass as `plot_harmonic_grid`'s `colorrange`.
"""
function symmetric_colorrange(data)
    m = maximum(abs, data)
    return (isfinite(m) && Float32(m) > 0.0f0) ? (-Float64(m), Float64(m)) : (-1.0, 1.0)
end
