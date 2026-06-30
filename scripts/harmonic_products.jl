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
    harmonic_field_style() -> (; colormap, colorrange)

Panel style for the field-harmonic heat-maps, splatted into [`plot_harmonic_grid`]. Those panels
show the **real part** of each component (signed, oscillatory data), so the choice here sets how
the sign structure reads. Returns a `NamedTuple` with:
  * `colormap`  — a colorscheme name/object;
  * `colorrange` — a `data -> (lo, hi)` function applied **per panel** (`symmetric_colorrange` and
    `harmonic_colorrange` are both in scope from ElectronDynamicsModels).
"""
function harmonic_field_style()
    return (; colormap = :jet, colorrange = symmetric_colorrange)
end

"""
    write_field_products(fields_h, fields_far_h, harmonics, ffund, x_grid, y_grid; w₀, run_tag,
        outdir, source_datafile, title_prefix, fileprefix, style = harmonic_field_style())

Per-harmonic field-map chips (`h<n>`, total + far `setup.field` toggle) drawn from already-reduced
harmonic maps via [`plot_harmonic_grid`]. Factored out of [`write_harmonic_products`] (like
[`write_phase_products`]) so the cached-hmaps recolor path can rebuild the field grids WITHOUT the
raw cube — `fields_h`/`fields_far_h`/`harmonics`/`ffund`/grids/`w₀` are all in the serialized hmaps.
A split cube (`fields_far_h !== nothing`) adds the far (radiation-only) variant the LPWA analytic
formula is compared against. `style` (colormap + per-panel colorrange) comes from
[`harmonic_field_style`]. Writes nothing to `[outputs].plots` — the field maps are derived chips.
"""
function write_field_products(
        fields_h, fields_far_h, harmonics, ffund, x_grid, y_grid;
        w₀, run_tag, outdir, source_datafile, title_prefix, fileprefix,
        style = harmonic_field_style(), window = "hann",
    )
    fieldsets = fields_far_h === nothing ? (("total", fields_h),) :
        (("total", fields_h), ("far", fields_far_h))
    for (k, n) in enumerate(harmonics), (ftype, fh) in fieldsets
        tag = ftype == "far" ? "fieldfar" : "field"
        out = joinpath(outdir, @sprintf("%s_%s_h%d_%s.png", fileprefix, tag, n, run_tag))
        plot_harmonic_grid(
            fh[k, :, :, :], x_grid, y_grid;
            w₀, labels = COMPLABELS, style...,
            title = @sprintf(
                "%s (%s field) — %dω₁ (%.3f× fundamental)",
                title_prefix, ftype, n, ffund[k]
            ),
            outfile = out,
        )
        println("saved → $out")
        write_derived(
            outdir; kind = "h$n", label = @sprintf("%dω₁ field maps", n), run_id = run_tag,
            plot = basename(out), source = source_datafile,
            setup = Dict("field" => ftype), plot_params = Dict("apodization" => window),
            description = "Field harmonic maps at $(n)ω₁ — each panel a component (Eˣ…Bᶻ). " *
                "Toggle total (far 1/R + near 1/R²) vs far (radiation 1/R only); the far field is " *
                "what the LPWA analytic formula is compared against in the lpwa-vs-numeric view. " *
                "Time→frequency reduction apodized with the '$window' window (leakage suppression).",
        )
    end
    return
end

"""
    write_harmonic_products(fld, x_grid, y_grid, ω, δt; w₀, run_tag, outdir, source_datafile,
        harmonics = (1, 2, 3, 4), title_prefix, fileprefix)
        -> (; hmapsfile, plots, fields_h, fields_far_h)

Reduce `fld = (; E, B)` to `fields_h` at the `harmonics` bins, serialize the reduced
`hmaps_<run_tag>.jls`, and write the per-run plots into `outdir`. For a split cube (`fld`
also carries `E_far`/`B_far`) the far field is reduced to `fields_far_h` the same way and
saved alongside the total maps (else `nothing`); the comparison script differences it for the
far-field-only diagnostic:

  * **field E/B grids** → per-harmonic `h<n>` derived chips with a total/far `setup.field` toggle
    (a split cube adds the far variant), drawn by [`write_field_products`]. **power spectrum** →
    the one intrinsic plot, returned in `plots` for the solver's `[outputs].plots`.
  * **∠F phase grids** → a single parametrized `phase` derived chip — one
    `derived_phase_<n>_*.toml` per harmonic via `write_derived` (`setup.harmonic = n`,
    `source = source_datafile`), so the builder shows ONE phase chip with a harmonic
    selector and the phase maps never collide with the field maps on `h<n>`.

`title_prefix`/`fileprefix` name the run family; the field-grid panel style (diverging colormap +
symmetric per-panel range) is owned by [`harmonic_field_style`], applied in [`write_field_products`].
"""
function write_harmonic_products(
        fld, x_grid, y_grid, ω, δt; w₀, run_tag, outdir, source_datafile,
        harmonics = (1, 2, 3, 4), title_prefix, fileprefix, window = hann
    )
    N_samples = size(fld.E, 1)
    freqs = rfftfreq(N_samples, 1 / δt)
    hbins = harmonic_bins(N_samples, δt, ω, harmonics)
    # `window` apodizes the time→frequency reduction (default Hann): the bare rfft leaks the strong
    # fundamental's skirt into the weak-harmonic bins, sinking small-a0 harmonics into a leakage floor.
    # See precision_floor_diagnostic. The window NAME is recorded below for provenance/reproducibility.
    win_name = window === nothing ? "none" : string(nameof(window))
    fields_h = harmonic_maps(fld, hbins; window)        # (length(harmonics), 6, Nx, Ny): E in 1:3, B in 4:6

    # Far-field-only harmonic maps, saved next to the total `fields_h` so the comparison script
    # can difference the radiation field on its own (LPWA is a far-field formula → comparing it
    # against the numeric far field is the rigorous apples-to-apples). A split cube
    # (accumulate_field mode=Val(:split)) carries `fld.E_far`/`fld.B_far`; a total cube does not.
    # Reduce the far field the SAME way (same window) as the total when present, else leave it `nothing`.
    if hasproperty(fld, :E_far)
        fields_far_h = harmonic_maps((; E = fld.E_far, B = fld.B_far), hbins; window)
    else
        fields_far_h = nothing
    end

    ffund = [freqs[b] / (ω / 2π) for b in hbins]
    hmapsfile = joinpath(outdir, "hmaps_$(run_tag).jls")
    serialize(
        hmapsfile,
        (;
            fields_h, fields_far_h, harmonics, ffund,
            x_grid = collect(x_grid), y_grid = collect(y_grid), w₀, window = win_name,
        ),
    )
    println("serialized harmonic maps → $hmapsfile  (window = $win_name)")

    # Field maps → per-harmonic `h<n>` chips (total + far `setup.field` toggle), drawn straight from
    # the reduced maps by write_field_products so the cached-hmaps recolor path shares the code.
    write_field_products(
        fields_h, fields_far_h, harmonics, ffund, x_grid, y_grid;
        w₀, run_tag, outdir, source_datafile, title_prefix, fileprefix, window = win_name,
    )

    plots = String[]
    psfile = joinpath(outdir, "powspec_$(run_tag).png")
    # Power spectrum stays UN-windowed (window = nothing): it is the diagnostic that must still SHOW the
    # raw leakage floor (apodizing it would hide the very thing it diagnoses).
    plot_power_spectrum(
        freqs, power_spectrum(fld; window = nothing);
        ω, labels = COMPLABELS, title = "$title_prefix — field power spectra (un-windowed)", outfile = psfile,
    )
    println("saved → $psfile")
    push!(plots, psfile)

    # Phase view (phaseE/phaseB chips). Factored out so the cached-hmaps recovery path can
    # rebuild it without the cube.
    write_phase_products(
        fields_h, x_grid, y_grid;
        w₀, harmonics, run_tag, outdir, source_datafile, title_prefix, fileprefix,
    )

    return (; hmapsfile, plots, fields_h, fields_far_h, window = win_name)
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
    # Only title/filename differ by family — both use the shared harmonic_field_style() colormap.
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
        hprod = write_harmonic_products(
            fld, x_grid, y_grid, ω, δt;
            w₀ = las["w0"], run_tag, outdir = dir,
            source_datafile = m["outputs"]["datafile"], title_prefix, fileprefix,
        )
        # Close the loop the inline (non-SKIP_POST) path already does: a deferred/async reduction
        # must ALSO declare what it produced, so [outputs] is complete for resolve_hmaps + the
        # dashboard hmaps download. `sorted` keeps [timing] last (the ops timing-append relies on it).
        outs = m["outputs"]
        if get(outs, "harmonic_maps", nothing) === nothing
            outs["harmonic_maps"] = basename(hprod.hmapsfile)
            outs["plots"] = basename.(hprod.plots)
            open(io -> TOML.print(io, m; sorted = true), toml, "w")
            println("declared harmonic_maps + plots in $(basename(toml))")
        end
        return hprod
    end

    # Cube absent (e.g. a published run that shipped only the reduced hmaps, or an ephemeral-VM
    # cell whose cube is gone): rebuild the field maps AND the phase view from the cached harmonic
    # maps — both only need `fields_h`/`fields_far_h`, not the raw cube. Only `powspec` (full
    # time-series) can't be regenerated here, so it's left as-is. This is the recolor path.
    hmapsfile = joinpath(dir, "hmaps_$(run_tag).jls")
    isfile(hmapsfile) ||
        error("neither cube ($(basename(cube))) nor hmaps ($(basename(hmapsfile))) found in $dir")
    println("cube absent — replotting field maps + phase from $(basename(hmapsfile)) …")
    h = deserialize(hmapsfile)
    _retire_stale_phase(dir, run_tag)
    # Guard older hmaps that predate fields_far_h/ffund/window: no far variant, integer-harmonic title,
    # unknown apodization.
    far = hasproperty(h, :fields_far_h) ? h.fields_far_h : nothing
    ffund = hasproperty(h, :ffund) ? h.ffund : Float64.(collect(h.harmonics))
    win = hasproperty(h, :window) ? h.window : "unknown"
    write_field_products(
        h.fields_h, far, h.harmonics, ffund, h.x_grid, h.y_grid;
        w₀ = h.w₀, run_tag, outdir = dir,
        source_datafile = m["outputs"]["datafile"], title_prefix, fileprefix, window = win,
    )
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
