# Shared harmonic-map products for the radiated-field runs: reduce a field cube
# `fld = (; E, B)` to the per-harmonic complex maps `fields_h`, serialize the reduced
# `hmaps_<run_id>.jls`, and emit the 2×3 E/B grids + ∠F phase grids + power spectrum.
# The live solvers (thomson_scattering.jl, lpwa.jl) `include` this and call
# `write_harmonic_products` so the plotting lives in ONE place. Run standalone on a
# run_*.toml to rebuild the reduced maps + plots from a serialized cube — e.g. recovering
# a Thomson run whose script only saved the 46 GB cube:
#
#   julia +release --project=scripts scripts/harmonic_products.jl runs/.../run_<UUID>.toml [...]
#
# Reading run params goes through the manifest's CURRENT sections ([config]/[laser]) via
# RunManifests, so this can't silently drift on a schema change the way hand-parsing the
# raw TOML would.

using TOML, Serialization, FFTW, Printf
using ElectronDynamicsModels    # harmonic_bins, harmonic_maps, power_spectrum, plot_*, symmetric_colorrange
using RunManifests              # check_schema_version, spp_from_manifest
using CairoMakie                # activates EDMPlotsExt (the plot_* methods)

const COMPLABELS = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")
const C_LIGHT = 137.03599908330932

"""
    write_harmonic_products(fld, x_grid, y_grid, ω, δt; w₀, run_tag, outdir, source_datafile,
        harmonics = (1, 2), title_prefix, fileprefix, colormap = :jet, colorrange = nothing)
        -> (; hmapsfile, plots, fields_h)

Reduce `fld = (; E, B)` to `fields_h` at the `harmonics` bins, serialize the reduced
`hmaps_<run_tag>.jls`, and write the per-run plots into `outdir`:

  * **field E/B grids + power spectrum** → the run's *intrinsic* plots, returned in `plots`
    for the solver's `[outputs].plots` (kinds `h<n>` + `powspec`).
  * **∠F phase grids** → a single parametrized `phase` derived chip — one
    `derived_phase_<n>_*.toml` per harmonic via `write_derived` (`setup.harmonic = n`,
    `source = source_datafile`), so the builder shows ONE phase chip with a harmonic
    selector and the phase maps never collide with the field maps on `h<n>`.

`title_prefix`/`fileprefix` name the run family; `colormap`/`colorrange` carry its style.
"""
function write_harmonic_products(
        fld, x_grid, y_grid, ω, δt; w₀, run_tag, outdir, source_datafile,
        harmonics = (1, 2), title_prefix, fileprefix, colormap = :jet, colorrange = nothing
    )
    N_samples = size(fld.E, 1)
    freqs = rfftfreq(N_samples, 1 / δt)
    hbins = harmonic_bins(N_samples, δt, ω, harmonics)
    fields_h = harmonic_maps(fld, hbins)        # (length(harmonics), 6, Nx, Ny): E in 1:3, B in 4:6

    hmapsfile = joinpath(outdir, "hmaps_$(run_tag).jls")
    serialize(
        hmapsfile,
        (;
            fields_h, harmonics, ffund = [freqs[b] / (ω / 2π) for b in hbins],
            x_grid = collect(x_grid), y_grid = collect(y_grid), w₀,
        ),
    )
    println("serialized harmonic maps → $hmapsfile")

    # Intrinsic plots ([outputs].plots): the E/B field maps (kinds h1/h2) + power spectrum.
    plots = String[]
    gridkw = colorrange === nothing ? (; colormap) : (; colormap, colorrange)
    for (k, n) in enumerate(harmonics)
        out = joinpath(outdir, @sprintf("%s_field_h%d_%s.png", fileprefix, n, run_tag))
        plot_harmonic_grid(
            fields_h[k, :, :, :], x_grid, y_grid;
            w₀, labels = COMPLABELS, gridkw...,
            title = @sprintf("%s (field) — %dω₁ (%.3f× fundamental)", title_prefix, n, freqs[hbins[k]] / (ω / 2π)),
            outfile = out,
        )
        println("saved → $out")
        push!(plots, out)
    end

    psfile = joinpath(outdir, "powspec_$(run_tag).png")
    plot_power_spectrum(
        freqs, power_spectrum(fld);
        ω, labels = COMPLABELS, title = "$title_prefix — field power spectra", outfile = psfile,
    )
    println("saved → $psfile")
    push!(plots, psfile)

    # Phase view (phaseE/phaseB chips). Factored out so the cached-hmaps recovery path can
    # rebuild it without the cube.
    write_phase_products(
        fields_h, x_grid, y_grid;
        w₀, harmonics, run_tag, outdir, source_datafile, title_prefix, fileprefix,
    )

    return (; hmapsfile, plots, fields_h)
end

"""
    write_phase_products(fields_h, x_grid, y_grid; w₀, harmonics, run_tag, outdir,
        source_datafile, title_prefix, fileprefix) -> Vector{String}

The per-field-type ∠F view from reduced harmonic maps `fields_h` (`(length(harmonics), 6, Nx,
Ny)`): for E (comps `1:3`) and B (`4:6`), one parametrized derived chip (`phaseE`/`phaseB`,
harmonic selector) — each component's ∠F heatmap with the test annuli drawn as dashed R±ringtol
circles, beside the ∠F-vs-azimuth winding (charge ℓ → ℓ windings). Ring radii + tolerance are
recorded in `[plot_params]` (w₀ units, as on the plot). Takes `fields_h` (not the cube), so the
live solve and the cached-hmaps recovery share one implementation; `x_grid`/`y_grid` may be
ranges or plain vectors (hmaps stores them collected, hence `x_grid[2] - x_grid[1]` not `step`).
"""
function write_phase_products(
        fields_h, x_grid, y_grid; w₀, harmonics, run_tag, outdir, source_datafile,
        title_prefix, fileprefix,
    )
    ringradii = maximum(abs, x_grid) .* (0.2, 0.4, 0.6)
    ringtol = 0.5 * (x_grid[2] - x_grid[1])   # ½ pixel pitch; annulus ≈1.3% of innermost R at Nx=400
    pparams = Dict(
        "ringtol/w₀" => round(ringtol / w₀; sigdigits = 3),
        "radii/w₀" => round.(collect(ringradii) ./ w₀; sigdigits = 3),
    )
    plots = String[]
    for (fld_tag, comps) in (("E", 1:3), ("B", 4:6)), (k, n) in enumerate(harmonics)
        out = joinpath(outdir, @sprintf("%s_phase%s_h%d_%s.png", fileprefix, fld_tag, n, run_tag))
        plot_phase_with_rings(
            fields_h[k, comps, :, :], x_grid, y_grid;
            w₀, labels = COMPLABELS[comps], radii = ringradii, tol = ringtol,
            title = @sprintf("%s — ∠F %s-field at %dω₁", title_prefix, fld_tag, n), outfile = out,
        )
        println("saved → $out")
        write_derived(
            outdir; kind = "phase$fld_tag", label = "$title_prefix ∠F $fld_tag", run_id = run_tag,
            plot = basename(out), source = source_datafile,
            setup = Dict("harmonic" => n), plot_params = pparams,
        )
        push!(plots, out)
    end
    return plots
end

# Delete the superseded single-grid phase artifacts (kinds `phase`/`phaserings`) for `run_tag` so
# a re-plotted run doesn't show both old and new (phaseE/phaseB) chips. Matches the OLD names only
# (NOT `phaseE`/`phaseB`): `derived_phase_<n>_<id8>` / `derived_phaserings_…` sidecars carrying this
# run's id8, and `*_phase_h<n>_<tag>.png` / `*_phaserings_h<n>_<tag>.png`.
function _retire_stale_phase(dir, run_tag)
    id8 = first(run_tag, 8)
    for f in readdir(dir)
        stale = ((occursin(r"^derived_phase_\d", f) || startswith(f, "derived_phaserings_")) && occursin(id8, f)) ||
                occursin(Regex("_phase_h\\d+_" * run_tag * "\\.png\$"), f) ||
                occursin(Regex("_phaserings_h\\d+_" * run_tag * "\\.png\$"), f)
        stale || continue
        rm(joinpath(dir, f))
        println("retired stale → $f")
    end
end

# ── Standalone recovery: rebuild reduced maps + plots from a serialized cube via its manifest ──
function recover_from_manifest(toml)
    m = TOML.parsefile(toml)
    check_schema_version(m; source = basename(toml))
    dir = dirname(abspath(toml))
    cfg, las = m["config"], m["laser"]
    # Only title/filename differ by family — both use the shared :jet/per-panel default colormap.
    lpwa = get(cfg, "trajectory_source", "") == "lpwa_analytic"
    title_prefix, fileprefix = lpwa ? ("LPWA", "lpwa") : ("Thomson scattering", "thomson")
    run_tag = m["provenance"]["run_id"]
    cube = joinpath(dir, m["outputs"]["datafile"])

    if isfile(cube)
        λ = las["wavelength"]
        ω = C_LIGHT * 2π / λ
        δt = 2π / ω / spp_from_manifest(m)
        x_grid = LinRange(-25las["w0"], 25las["w0"], cfg["Nx"])
        y_grid = LinRange(-25las["w0"], 25las["w0"], cfg["Ny"])
        println("loading $(m["outputs"]["datafile"]) …")
        raw = deserialize(cube)
        fld = (; E = raw.E, B = raw.B)   # total-field maps only; drop any split E_rad/B_rad to halve resident RAM
        raw = nothing
        GC.gc()
        return write_harmonic_products(
            fld, x_grid, y_grid, ω, δt;
            w₀ = las["w0"], run_tag, outdir = dir,
            source_datafile = m["outputs"]["datafile"], title_prefix, fileprefix,
        )
    end

    # Cube absent (e.g. a published run that shipped only the reduced hmaps): rebuild just the
    # phase view from the cached harmonic maps. Field grids + powspec need the cube — left as-is.
    hmapsfile = joinpath(dir, "hmaps_$(run_tag).jls")
    isfile(hmapsfile) ||
        error("neither cube ($(basename(cube))) nor hmaps ($(basename(hmapsfile))) found in $dir")
    println("cube absent — replotting phase from $(basename(hmapsfile)) …")
    h = deserialize(hmapsfile)
    _retire_stale_phase(dir, run_tag)
    return write_phase_products(
        h.fields_h, h.x_grid, h.y_grid;
        w₀ = h.w₀, harmonics = h.harmonics, run_tag, outdir = dir,
        source_datafile = m["outputs"]["datafile"], title_prefix, fileprefix,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: harmonic_products.jl run_<UUID>.toml [run_*.toml ...]")
    for toml in ARGS
        recover_from_manifest(toml)
    end
end
