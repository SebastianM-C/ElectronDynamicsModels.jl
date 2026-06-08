# Frequency power spectra of all four 4-potential components, summed over the
# screen:  power_spec[ν, μ] = Σ_{ix,iy} |A_ω[ν, μ, ix, iy]|²,  A_ω = rfft(A, 1).
# Overlays A⁰ (ct), Aˣ, Aʸ, Aᶻ on a log-y axis with harmonic markers, so you can
# see which components (if any) carry harmonic structure.  Caches the
# (N÷2+1)×4 spectra so re-plots are instant.
#
#   julia --project=scripts scripts/plot_power_spectrum.jl [file.jls]

using Serialization
using FFTW
using CairoMakie
using Printf
using ElectronDynamicsModels   # power_spectrum + plot_power_spectrum (EDMPlotsExt)

const c = 137.03599908330932
const ω = 0.057

const datafile = length(ARGS) ≥ 1 ? ARGS[1] : "A_rk4_400_N10000_Ns8000_spp16.jls"

# The run TOML (not the data filename) is the source of truth for
# samples_per_period; resolve the parent manifest once and reuse it for the
# derived sidecar below.
include(joinpath(@__DIR__, "manifest.jl"))
const dir = dirname(abspath(datafile))
const parent = find_parent_manifest(dir, basename(datafile))
parent === nothing && error("no run_*.toml in $dir binds $(basename(datafile)) — " *
    "needed for samples_per_period (thomson_scattering.jl emits the run manifest)")
const samples_per_period = spp_from_manifest(parent[2])
const δt = 2π / ω / samples_per_period
const stem = replace(datafile, r"\.jls$" => "")
const cachefile = stem * "_powspec_all.jls"
const labels = ["A⁰ (ct)", "Aˣ", "Aʸ", "Aᶻ"]
const colors = [:black, :dodgerblue, :seagreen, :crimson]
# Aˣ and Aʸ are near-degenerate and overlap; dash the one drawn on top (Aʸ) so the
# one underneath (Aˣ) stays visible, and dot Aᶻ to set it apart too.
const linestyles = [:solid, :solid, :dash, :dot]

if isfile(cachefile)
    cache = deserialize(cachefile)
    println("loaded $cachefile")
else
    println("loading $datafile (slow path)…")
    A = deserialize(datafile)
    N_samples = size(A, 1)
    power_spec = power_spectrum(A)
    A = nothing
    GC.gc()
    freqs = rfftfreq(N_samples, 1 / δt)
    cache = (freqs = collect(freqs), power_spec = power_spec, samples_per_period = samples_per_period)
    serialize(cachefile, cache)
    println("cached → $cachefile")
end

xh = cache.freqs ./ (ω / 2π)               # frequency in units of the fundamental (for the table below)

plot_power_spectrum(
    cache.freqs, cache.power_spec; ω, labels, colors, linestyles,
    title = "4-potential power spectra — $(basename(datafile))  ($(cache.samples_per_period) samples/period)",
    outfile = stem * "_powspec_all.png",
)

# Per-component power at each integer harmonic.
@printf("\n%-4s" , "n")
for L in labels; print(rpad(L, 14)); end
println()
for n in 1:floor(Int, last(xh))
    idx = argmin(abs.(cache.freqs .- n * ω / 2π))
    @printf("%-4d", n)
    for μ in 1:4
        print(rpad(string(round(cache.power_spec[idx, μ], sigdigits = 4)), 14))
    end
    println()
end
println("saved → $(stem)_powspec_all.png")

# ── derived-artifact metadata for the results dashboard (research.314159265.dev) ──
# manifest.jl is already included; `dir`/`parent` resolved at the top.
let pid = parent[1]
    if pid === nothing
        @warn "parent run manifest for $(basename(datafile)) has no run_id; skipping derived sidecar"
    else
        write_derived(dir; kind = "powspec", label = "power spectrum", run_id = pid,
            plot = basename(stem * "_powspec_all.png"), source = basename(datafile),
            datafile = basename(cachefile))   # publish the small reduced-spectrum cache
        println("derived sidecar → powspec (parent run $pid)")
    end
end
