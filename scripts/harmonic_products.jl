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
        harmonics = (1, 2, 3, 4), title_prefix, fileprefix, colormap = :jet, colorrange = nothing)
        -> (; hmapsfile, plots, fields_h, fields_far_h)

Reduce `fld = (; E, B)` to `fields_h` at the `harmonics` bins, serialize the reduced
`hmaps_<run_tag>.jls`, and write the per-run plots into `outdir`. For a split cube (`fld`
also carries `E_far`/`B_far`) the far field is reduced to `fields_far_h` the same way and
saved alongside the total maps (else `nothing`); the comparison script differences it for the
far-field-only diagnostic:

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
        harmonics = (1, 2, 3, 4), title_prefix, fileprefix, colormap = :jet, colorrange = nothing
    )
    N_samples = size(fld.E, 1)
    freqs = rfftfreq(N_samples, 1 / δt)
    hbins = harmonic_bins(N_samples, δt, ω, harmonics)
    fields_h = harmonic_maps(fld, hbins)        # (length(harmonics), 6, Nx, Ny): E in 1:3, B in 4:6

    # Far-field-only harmonic maps, saved next to the total `fields_h` so the comparison script
    # can difference the radiation field on its own (LPWA is a far-field formula → comparing it
    # against the numeric far field is the rigorous apples-to-apples). A split cube
    # (accumulate_field mode=Val(:split)) carries `fld.E_far`/`fld.B_far`; a total cube does not.
    # Reduce the far field the SAME way as the total when present, else leave it `nothing` so the
    # serialized layout stays backward-compatible with the already-published total-only hmaps.
    if hasproperty(fld, :E_far)
        fields_far_h = harmonic_maps((; E = fld.E_far, B = fld.B_far), hbins)
    else
        fields_far_h = nothing
    end

    hmapsfile = joinpath(outdir, "hmaps_$(run_tag).jls")
    serialize(
        hmapsfile,
        (;
            fields_h, fields_far_h, harmonics, ffund = [freqs[b] / (ω / 2π) for b in hbins],
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

    # Far-field maps → a `fieldFar` derived chip (harmonic selector), parallel to the total-field
    # h<n> grids but for the radiation (1/R) field alone — the quantity the LPWA analytic formula
    # is compared against. Only for a split cube (fields_far_h present); the builder renders new
    # derived kinds generically.
    if fields_far_h !== nothing
        for (k, n) in enumerate(harmonics)
            out = joinpath(outdir, @sprintf("%s_fieldfar_h%d_%s.png", fileprefix, n, run_tag))
            plot_harmonic_grid(
                fields_far_h[k, :, :, :], x_grid, y_grid;
                w₀, labels = COMPLABELS, gridkw...,
                title = @sprintf("%s (far field) — %dω₁ (%.3f× fundamental)", title_prefix, n, freqs[hbins[k]] / (ω / 2π)),
                outfile = out,
            )
            println("saved → $out")
            write_derived(
                outdir; kind = "fieldFar", label = "$title_prefix far-field maps", run_id = run_tag,
                plot = basename(out), source = source_datafile, setup = Dict("harmonic" => n),
                description = "Far (radiation) field harmonic maps — the 1/R term alone, without " *
                    "the near-field 1/R². Each panel is a component (Eˣ…Bᶻ). This is the field the " *
                    "LPWA analytic formula is compared against in the lpwa-vs-numeric view.",
            )
        end
    end

    # Phase view (phaseE/phaseB chips). Factored out so the cached-hmaps recovery path can
    # rebuild it without the cube.
    write_phase_products(
        fields_h, x_grid, y_grid;
        w₀, harmonics, run_tag, outdir, source_datafile, title_prefix, fileprefix,
    )

    return (; hmapsfile, plots, fields_h, fields_far_h)
end

"""
    write_phase_products(fields_h, x_grid, y_grid; w₀, harmonics, run_tag, outdir,
        source_datafile, title_prefix, fileprefix) -> Vector{String}

The per-field-type ∠F view from reduced harmonic maps `fields_h` (`(length(harmonics), 6, Nx,
Ny)`): for E (comps `1:3`) and B (`4:6`), per harmonic, two derived chips — `phaseE`/`phaseB`
(heatmap with dashed R±ringtol annuli beside the ∠F-vs-azimuth winding + per-radius linear fit on
the transverse components) and a separate polar companion `phasePolarE`/`phasePolarB`. Test radii
are `4,8,12 w₀`. `[plot_params]` records the ring geometry (`ringtol/w₀`, `radii/w₀`) plus, per
transverse component, the winding `slope` (≈ ℓ) and offset `b/π` over the radii. Takes `fields_h`
(not the cube), so the live solve and the cached-hmaps recovery share one implementation;
`x_grid`/`y_grid` may be ranges or vectors (hmaps stores them collected, hence `x_grid[2]-x_grid[1]`).
"""
function write_phase_products(
        fields_h, x_grid, y_grid; w₀, harmonics, run_tag, outdir, source_datafile,
        title_prefix, fileprefix,
    )
    ringradii = w₀ .* (4, 8, 12)              # test radii in beam waists
    ringtol = 0.5 * (x_grid[2] - x_grid[1])   # ½ pixel pitch; thin annulus (vector x_grid ⇒ not `step`)
    base_pp = Dict{String, Any}(
        "ringtol/w₀" => round(ringtol / w₀; sigdigits = 3),
        "radii/w₀" => round.(collect(ringradii) ./ w₀; sigdigits = 3),
    )
    plots = String[]
    for (fld_tag, comps) in (("E", 1:3), ("B", 4:6)), (k, n) in enumerate(harmonics)
        lbls = COMPLABELS[comps]
        out = joinpath(outdir, @sprintf("%s_phase%s_h%d_%s.png", fileprefix, fld_tag, n, run_tag))
        res = plot_phase_with_rings(
            fields_h[k, comps, :, :], x_grid, y_grid;
            w₀, labels = lbls, radii = ringradii, tol = ringtol,
            title = @sprintf("%s — ∠F %s-field at %dω₁", title_prefix, fld_tag, n), outfile = out,
        )
        println("saved → $out")
        # Per-(field, harmonic) plot_params: the shared ring geometry + the per-component winding
        # fits (slope ≈ ℓ, intercept b/π) per radius — surfaced in the modal "Plot parameters" panel.
        pp = copy(base_pp)
        for (c, fit) in res.fits          # one entry per component (Eˣ/Eʸ/Eᶻ or Bˣ/Bʸ/Bᶻ)
            pp["slope $(lbls[c])"] = round.(fit.slope; sigdigits = 3)
            pp["b/π $(lbls[c])"] = round.(fit.b ./ π; sigdigits = 3)
        end
        write_derived(
            outdir; kind = "phase$fld_tag", label = "$title_prefix ∠F $fld_tag", run_id = run_tag,
            plot = basename(out), source = source_datafile,
            setup = Dict("harmonic" => n), plot_params = pp,
            description = "Each row is a component. Left: ∠F heatmap with white dashed circles at " *
                "R ± ringtol marking the test rings sampled on the right. Right: ∠F vs azimuth on " *
                "each ring (colour-matched to its circle), with the unwrapped linear fit " *
                "∠F ≈ slope·φ + b overlaid — slope ≈ winding ℓ; fitted slope and b/π per radius are " *
                "in the Plot parameters.",
        )
        push!(plots, out)

        # Polar companion — separate chip (azimuth → angular axis, ∠F → radial).
        pout = joinpath(outdir, @sprintf("%s_phasePolar%s_h%d_%s.png", fileprefix, fld_tag, n, run_tag))
        plot_phase_polar(
            fields_h[k, comps, :, :], x_grid, y_grid;
            w₀, labels = lbls, radii = ringradii, tol = ringtol,
            title = @sprintf("%s — ∠F %s-field (polar) at %dω₁", title_prefix, fld_tag, n), outfile = pout,
        )
        println("saved → $pout")
        write_derived(
            outdir; kind = "phasePolar$fld_tag", label = "$title_prefix ∠F $fld_tag (polar)",
            run_id = run_tag, plot = basename(pout), source = source_datafile,
            setup = Dict("harmonic" => n), plot_params = base_pp,
        )
        push!(plots, pout)
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
    return
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
        # A split cube carries E_far/B_far — keep them so write_harmonic_products also emits the
        # far-field maps the φ0/LPWA comparison needs; a total cube has only E/B.
        fld = hasproperty(raw, :E_far) ? raw : (; E = raw.E, B = raw.B)
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
