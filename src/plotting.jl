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
    plot_phase_with_rings(maps, x_grid, y_grid; w₀=1, labels, radii, tol, title="",
                          panelsize=300, outfile=nothing)

Combined per-field phase view for *one* field type (`maps` is `(3, Nx, Ny)` — e.g. the E or B
slice `fields_h[k, 1:3, :, :]` / `[k, 4:6, :, :]`). One row per component, two columns:

  * **left** — the phase heatmap `angle.(maps[c])` on the cyclic `:phase` colormap over `(-π, π)`,
    with the test annuli drawn as dashed circles at `R ± tol` (in `w₀` units, to match the axes),
    one colour per radius;
  * **right** — the **azimuthal phase winding**: `angle(F)` of the pixels on each test circle of
    radius `R ∈ radii` (a thin annulus, half-width `tol`, about the grid centre) scattered against
    azimuth `atan(y, x)`, coloured to match the circles on the left. A vortex of topological
    charge ℓ shows ℓ phase windings as φ runs once around. The **transverse** components (1,2 —
    Eˣ/Eʸ or Bˣ/Bʸ) additionally get the [`phase_winding_fit`](@ref) line overlaid (re-wrapped to
    ±π), with `slope ≈ ℓ` and intercept `b` the phase offset.

Returns `(; fig, fits)` where `fits[c]` (for the transverse components `c ∈ 1:2`) is
`(; slope, b)` — vectors over `radii` (NaN where a ring has no pixels), so callers can record the
winding/offset in `[plot_params]`. **Requires CairoMakie** (package extension).
"""
function plot_phase_with_rings end

"""
    plot_phase_polar(maps, x_grid, y_grid; w₀=1, labels, radii, tol, title="", panelsize=300, outfile=nothing)

Polar companion to [`plot_phase_with_rings`]: for each component of `maps` (`(3, Nx, Ny)`), a
`PolarAxis` with **angular = azimuth φ** and **radial = ∠F shifted to `[0, 2π)`** (`angle(F)+π`),
one colour-matched series per test ring `R ∈ radii` (same pixels as the cartesian view, via
[`ring_pixels`](@ref)). A winding of charge ℓ traces ℓ radial oscillations as φ runs once around.
A separate diagnostic from the main phase figure. **Requires CairoMakie** (package extension).
"""
function plot_phase_polar end

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
