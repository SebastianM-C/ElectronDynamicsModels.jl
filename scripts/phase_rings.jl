# Azimuthal-phase diagnostic for a Thomson-scattering run, computed offline from
# its .jls (same pattern as plot_harmonics.jl). Two plots:
#   (left)  phase map  angle.(Â[bin, comp, :, :])  over the screen, and
#   (right) the phase of the pixels on each test circle of radius R, scattered
#           against azimuth φ — a vortex of charge ℓ shows ℓ phase windings as φ
#           goes once around.
#
# The radiated quantity is the 4-potential Aᵘ (thomson_scattering.jl computes
# accumulate_potential and serializes it); we rfft along observer time for the
# harmonic Â(ω). The azimuthal phase winding lives in the potential as in the
# field. Component index into Aᵘ: 1=A⁰, 2=Aˣ, 3=Aʸ, 4=Aᶻ (default Aˣ).
#
# Emits a derived_*.toml sidecar (via manifest.jl) binding the plot to its parent
# run and recording the diagnostic parameters (harmonic, component, tolerance) so
# the results dashboard shows what was used.
#
#   julia --project=scripts scripts/phase_rings.jl [file.jls] [harmonic]
# Env: EDM_COMP (default 2), EDM_R (ring radius; default 0.4·half-width),
#      EDM_RADII (comma list, overrides EDM_R), EDM_TOL (annulus half-width).

using Serialization
using FFTW
using LinearAlgebra
using Printf
using CairoMakie

# ── Thomson screen constants (identical to scripts/thomson_scattering.jl) ──
const c = 137.03599908330932
const ω = 0.057
const λ = 2π * c / ω
const w₀ = 75λ
const complabels = ("A⁰", "Aˣ", "Aʸ", "Aᶻ")
const asciilabels = ("A0", "Ax", "Ay", "Az")   # filename/sidecar-safe

const datafile = length(ARGS) ≥ 1 ? ARGS[1] : "A_rk4_400_N10000_Ns8000_spp16.jls"
const harmonic = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : parse(Int, get(ENV, "EDM_HARMONIC", "1"))
const comp = parse(Int, get(ENV, "EDM_COMP", "2"))

# Resolve the parent run manifest once: it is the source of truth for
# `samples_per_period` (not the data filename) and for the sidecar's run_id.
include(joinpath(@__DIR__, "manifest.jl"))
const dir = dirname(abspath(datafile))
const parent = find_parent_manifest(dir, basename(datafile))
parent === nothing && error("no run_*.toml in $dir binds $(basename(datafile)) — " *
    "needed for samples_per_period (thomson_scattering.jl emits the run manifest)")
const samples_per_period = spp_from_manifest(parent[2])
const δt = 2π / ω / samples_per_period
const stem = replace(datafile, r"\.jls$" => "")

# ── Plotting primitives ──

"""
    ring_pixels(x_grid, y_grid, R; tol) -> (idxs, azimuths)

CartesianIndices of pixels whose radius is within `tol` of `R` (a thin annulus
about the grid centre), sorted by azimuth, with those azimuths `atan(y, x)`.
"""
function ring_pixels(x_grid, y_grid, R; tol)
    CI = CartesianIndices((length(x_grid), length(y_grid)))
    idxs = filter(ci -> abs(hypot(x_grid[ci[1]], y_grid[ci[2]]) - R) < tol, CI)
    az = [atan(y_grid[ci[2]], x_grid[ci[1]]) for ci in idxs]
    order = sortperm(az)
    return idxs[order], az[order]
end

"""
    plot_phase_rings(phase, x_grid, y_grid; radii, tol, outfile, comp_label)

`phase :: (Nx, Ny)` real phase map. Saves the two-panel figure: phase heatmap with
the test circles overlaid, and the phase-vs-azimuth scatter on each ring.
"""
function plot_phase_rings(phase, x_grid, y_grid; radii, tol, outfile, comp_label)
    fig = Figure(size = (1100, 470))
    ax1 = Axis(fig[1, 1]; aspect = 1, xlabel = "x", ylabel = "y",
        title = "phase  angle($comp_label)")
    hm = heatmap!(ax1, x_grid, y_grid, phase; colormap = :phase, colorrange = (-π, π))
    Colorbar(fig[1, 2], hm; label = "phase [rad]")

    ax2 = Axis(fig[1, 3]; xlabel = "azimuth  φ  [rad]", ylabel = "phase on ring [rad]",
        title = "phase vs azimuth on rings", limits = (-π, π, -π, π))
    θ = range(-π, π, 200)
    for R in radii
        idxs, az = ring_pixels(x_grid, y_grid, R; tol)
        if isempty(idxs)
            @warn "no pixels within tol of R — widen tol" R tol
            continue
        end
        scatter!(ax2, az, phase[idxs]; markersize = 6,
            label = "R = $(round(R; sigdigits = 3))  ($(length(idxs)) px)")
        lines!(ax1, R .* cos.(θ), R .* sin.(θ); color = :white, linestyle = :dash, linewidth = 0.7)
    end
    axislegend(ax2; labelsize = 9, position = :lt)
    save(outfile, fig)
    return fig
end

# ── Load the run, take the harmonic phase of one component ──
isfile(datafile) || error("data file not found: $datafile (pass the run .jls as ARG 1)")
println("loading $datafile …")
A_s = deserialize(datafile)
N_samples, _, Nx, Ny = size(A_s)
x_grid = LinRange(-25w₀, 25w₀, Nx)
y_grid = LinRange(-25w₀, 25w₀, Ny)

freqs = rfftfreq(N_samples, 1 / δt)
bin = findmin(f -> abs(f - harmonic * ω / 2π), freqs)[2]
@printf("harmonic %d → bin %d  (%.4f× fundamental), comp = %d (%s)\n",
    harmonic, bin, freqs[bin] / (ω / 2π), comp, complabels[comp])

Â = rfft(@view(A_s[:, comp, :, :]), 1)
phase = angle.(Â[bin, :, :])
A_s = nothing
GC.gc()

# ── Ring radii (R_test) and annulus tolerance — the diagnostic parameters ──
const halfwidth = 25w₀
radii = if haskey(ENV, "EDM_RADII")
    parse.(Float64, split(ENV["EDM_RADII"], ","))
elseif haskey(ENV, "EDM_R")
    [parse(Float64, ENV["EDM_R"])]
else
    [0.4 * halfwidth]
end
tol = haskey(ENV, "EDM_TOL") ? parse(Float64, ENV["EDM_TOL"]) : 1.5 * step(x_grid)

out = stem * @sprintf("_phaserings_h%d_%s.png", harmonic, asciilabels[comp])
plot_phase_rings(phase, x_grid, y_grid;
    radii, tol, outfile = out,
    comp_label = @sprintf("%s, %dω₁", complabels[comp], harmonic))
println("saved → $out")

# ── Derived-artifact sidecar for the results dashboard (research.314159265.dev) ──
# Bind the plot to its parent run (already resolved for samples_per_period above);
# `setup` keys are the dashboard picker axis and record the diagnostic parameters
# used (tolerance, harmonic, component).
let pid = parent[1]
    if pid === nothing
        @warn "parent run manifest for $(basename(datafile)) has no run_id; skipping sidecar"
    else
        write_derived(dir;
            kind = "phaserings",
            label = @sprintf("phase rings %dω₁ %s (R/hw = %s)", harmonic, complabels[comp],
                join(string.(round.(radii ./ halfwidth; sigdigits = 2)), ",")),
            run_id = pid, plot = basename(out), source = basename(datafile),
            setup = Dict(
                "harmonic" => harmonic,
                "component" => asciilabels[comp],
                "tol" => round(tol; sigdigits = 4),
            ))
        println("derived sidecar → phaserings (parent run $pid)")
    end
end
