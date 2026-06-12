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
    write_harmonic_products(fld, x_grid, y_grid, ω, δt; w₀, run_tag, outdir,
        harmonics = (1, 2), title_prefix, fileprefix, colormap = :jet, colorrange = nothing)
        -> (; hmapsfile, plotfiles, fields_h)

Reduce `fld = (; E, B)` to `fields_h` at the `harmonics` bins, serialize the reduced
`hmaps_<run_tag>.jls`, and write the E/B harmonic grids, ∠F phase grids, and field power
spectrum into `outdir`. `title_prefix`/`fileprefix` name the run family ("Thomson
scattering"/"thomson", "LPWA"/"lpwa"); `colormap`/`colorrange` carry its house style.
"""
function write_harmonic_products(
        fld, x_grid, y_grid, ω, δt; w₀, run_tag, outdir,
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

    plotfiles = String[]
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
        push!(plotfiles, out)
    end

    psfile = joinpath(outdir, "powspec_$(run_tag).png")
    plot_power_spectrum(
        freqs, power_spectrum(fld);
        ω, labels = COMPLABELS, title = "$title_prefix — field power spectra", outfile = psfile,
    )
    println("saved → $psfile")
    push!(plotfiles, psfile)

    for (k, n) in enumerate(harmonics)
        out = joinpath(outdir, @sprintf("%s_phase_h%d_%s.png", fileprefix, n, run_tag))
        plot_phase_grid(
            fields_h[k, :, :, :], x_grid, y_grid;
            w₀, labels = COMPLABELS, title = @sprintf("%s (field) — ∠F at %dω₁", title_prefix, n), outfile = out,
        )
        println("saved → $out")
        push!(plotfiles, out)
    end

    return (; hmapsfile, plotfiles, fields_h)
end

# ── Standalone recovery: rebuild reduced maps + plots from a serialized cube via its manifest ──
function recover_from_manifest(toml)
    m = TOML.parsefile(toml)
    check_schema_version(m; source = basename(toml))
    dir = dirname(abspath(toml))
    cfg, las = m["config"], m["laser"]
    λ = las["wavelength"]
    ω = C_LIGHT * 2π / λ
    δt = 2π / ω / spp_from_manifest(m)
    x_grid = LinRange(-25las["w0"], 25las["w0"], cfg["Nx"])
    y_grid = LinRange(-25las["w0"], 25las["w0"], cfg["Ny"])
    # lpwa runs already serialize hmaps; recovery is mainly the Thomson cube. Only title/
    # filename differ by family — both use the shared :jet/per-panel default colormap.
    lpwa = get(cfg, "trajectory_source", "") == "lpwa_analytic"
    title_prefix, fileprefix = lpwa ? ("LPWA", "lpwa") : ("Thomson scattering", "thomson")

    println("loading $(m["outputs"]["datafile"]) …")
    raw = deserialize(joinpath(dir, m["outputs"]["datafile"]))
    fld = (; E = raw.E, B = raw.B)   # total-field maps only; drop any split E_rad/B_rad to halve resident RAM
    raw = nothing
    GC.gc()
    return write_harmonic_products(
        fld, x_grid, y_grid, ω, δt;
        w₀ = las["w0"], run_tag = m["provenance"]["run_id"], outdir = dir, title_prefix, fileprefix,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: harmonic_products.jl run_<UUID>.toml [run_*.toml ...]")
    for toml in ARGS
        recover_from_manifest(toml)
    end
end
