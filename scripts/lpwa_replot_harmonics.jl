# Recolor the LPWA harmonic maps (1ω₁, 2ω₁) from already-serialized field data,
# WITHOUT re-running the (~3.7 h/run) GPU accumulation. Mirrors the harmonic-plot
# block of scripts/lpwa.jl, but loads `fld` from a .jls and reconstructs the screen
# grid + frequency bins from that run's `run_<UUID>.toml` manifest (so nothing is
# hard-coded — each plot is reproduced from exactly the params it was generated with).
#
# Usage:
#   julia +release --project=scripts scripts/lpwa_replot_harmonics.jl [field_*.jls ...]
# With no args it globs runs/lpwa_local/field_*.jls. Output PNGs reuse the source
# UUID filenames (lpwa_field_h{n}_<UUID>.png), overwriting the previous colormap.

using Serialization
using FFTW
using CairoMakie
using TOML
using Printf

const C_LIGHT = 137.03599908330932          # matches `c` in scripts/lpwa.jl
const HARMONICS = (1, 2)
const COMPLABELS = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")

# Per-panel colorrange for the :jet heatmap: the data's own (min, max) extrema, so the
# full jet rainbow spans each panel's range — matching the reference article's
# imagesc-style (auto-caxis) scaling rather than a zero-centered diverging view.
function harmonic_colorrange(field)
    return extrema(field)
end

"Locate the run_<UUID>.toml sitting next to a field_*.jls and return its parsed params."
function run_manifest(jls)
    stem = splitext(basename(jls))[1]            # field_…_<UUID>
    run_tag = last(split(stem, "_"))             # <UUID> (hyphens, no underscores)
    manifest = joinpath(dirname(jls), "run_$(run_tag).toml")
    isfile(manifest) || error("no manifest beside $jls (expected $manifest)")
    return run_tag, TOML.parsefile(manifest)
end

"Reconstruct (x_grid, y_grid, w₀, ω, freqs, harmonic_bins) from a parsed manifest."
function reconstruct_grid(meta)
    Nx = meta["setup"]["Nx"]
    Ny = meta["setup"]["Ny"]
    w₀ = meta["model"]["w0"]
    N_samples = meta["setup"]["N_samples"]
    spp = meta["config"]["samples_per_period"]
    λ = meta["model"]["wavelength"]
    ω = C_LIGHT * 2π / λ
    δt = 2π / ω / spp

    x_grid = LinRange(-25w₀, 25w₀, Nx)
    y_grid = LinRange(-25w₀, 25w₀, Ny)
    freqs = rfftfreq(N_samples, 1 / δt)
    harmonic_bins = [findmin(x -> abs(x - n * ω / 2π), freqs)[2] for n in HARMONICS]
    return x_grid, y_grid, w₀, ω, freqs, harmonic_bins
end

"rfft each of the 6 field components along time, keep only the harmonic bins."
function harmonic_maps(fld, harmonic_bins, Nx, Ny)
    comps = ((fld.E, 1), (fld.E, 2), (fld.E, 3), (fld.B, 1), (fld.B, 2), (fld.B, 3))
    fields_h = Array{ComplexF64, 4}(undef, length(HARMONICS), 6, Nx, Ny)
    for (cc, (arr, j)) in enumerate(comps)
        Fω = rfft(arr[:, j, :, :], 1)
        for (k, idx) in enumerate(harmonic_bins)
            fields_h[k, cc, :, :] = Fω[idx, :, :]
        end
        Fω = nothing
        GC.gc()
    end
    return fields_h
end

function plot_harmonic(k, n, fields_h, harmonic_bins, freqs, ω, w₀, x_grid, y_grid, out)
    idx = harmonic_bins[k]
    fig = Figure()
    Label(
        fig[0, :], @sprintf(
            "LPWA (field) — %dω₁ (%.3f× fundamental)",
            n, freqs[idx] / (ω / 2π)
        ), fontsize = 16, font = :bold
    )
    for cc in 1:6
        field = real.(fields_h[k, cc, :, :])
        cr = maximum(abs, field)
        row = cc ≤ 3 ? 1 : 2          # E components on the top row, B on the bottom
        col = (cc - 1) % 3 + 1
        gl = fig[row, col] = GridLayout()
        ax = Axis(
            gl[1, 1], width = 300, height = 300, xlabel = "x / w₀", ylabel = "y / w₀",
            title = @sprintf("%s  (peak %.2e)", COMPLABELS[cc], cr)
        )
        hm = heatmap!(
            ax, collect(x_grid) ./ w₀, collect(y_grid) ./ w₀, field,
            colorrange = harmonic_colorrange(field), colormap = :jet
        )
        Colorbar(gl[1, 2], hm, width = 10, height = 300)
    end
    resize_to_layout!(fig)
    save(out, fig)
    println("saved → $out")
    return out
end

function recolor(jls)
    run_tag, meta = run_manifest(jls)
    x_grid, y_grid, w₀, ω, freqs, harmonic_bins = reconstruct_grid(meta)
    Nx, Ny = length(x_grid), length(y_grid)
    println("loading $jls …")
    fld = deserialize(jls)
    fields_h = harmonic_maps(fld, harmonic_bins, Nx, Ny)
    fld = nothing
    GC.gc()
    outdir = dirname(jls)
    for (k, n) in enumerate(HARMONICS)
        out = joinpath(outdir, @sprintf("lpwa_field_h%d_%s.png", n, run_tag))
        plot_harmonic(k, n, fields_h, harmonic_bins, freqs, ω, w₀, x_grid, y_grid, out)
    end
    return
end

const JLS = isempty(ARGS) ?
    sort(
        filter(
            f -> endswith(f, ".jls") && !occursin("obscache", f),
            readdir("runs/lpwa_local", join = true)
        )
    ) : ARGS

isempty(JLS) && error("no .jls files to recolor")
for jls in JLS
    recolor(jls)
end
