# Screen maps of the radiated 4-potential at chosen harmonics of ω₁, in the same
# 2×2-component layout (A⁰, Aˣ, Aʸ, Aᶻ) as thomson_scattering.jl's built-in h1/h2
# figures — but computed offline from a run's .jls, so harmonics can be added after
# the fact (e.g. 3ω₁, 5ω₁). Each harmonic → one PNG + an `h<n>` derived sidecar for
# the dashboard. Caches the per-harmonic complex slices for all four components to
# "<stem>_hmaps.jls" (~80 MB) so later harmonic picks / re-plots are instant.
#
#   julia --project=scripts scripts/plot_harmonics.jl [file.jls] [n1 n2 …]   # default: 3 5

using Serialization
using FFTW
using CairoMakie
using Printf

const c = 137.03599908330932
const ω = 0.057
const λ = 2π * c / ω
const w₀ = 75λ

const datafile = length(ARGS) ≥ 1 ? ARGS[1] : "A_rk4_400_N10000_Ns8000_spp16.jls"
const harmonics = length(ARGS) ≥ 2 ? parse.(Int, ARGS[2:end]) : [3, 5]

# The run TOML (not the data filename) is the source of truth for
# samples_per_period; resolve the parent manifest once and reuse it for the
# derived sidecars below.
include(joinpath(@__DIR__, "manifest.jl"))
const dir = dirname(abspath(datafile))
const parent = find_parent_manifest(dir, basename(datafile))
parent === nothing && error("no run_*.toml in $dir binds $(basename(datafile)) — " *
    "needed for samples_per_period (thomson_scattering.jl emits the run manifest)")
const samples_per_period = spp_from_manifest(parent[2])
const δt = 2π / ω / samples_per_period
const stem = replace(datafile, r"\.jls$" => "")
const cachefile = stem * "_hmaps.jls"
const complabels = ("A⁰", "Aˣ", "Aʸ", "Aᶻ")
const max_harmonic = 8          # cache through Nyquist (SPP=16 → 8×) so any pick is instant

if isfile(cachefile)
    cache = deserialize(cachefile)
    println("loaded $cachefile")
else
    println("loading $datafile (slow path)…")
    A_s = deserialize(datafile)
    N_samples, _, Nx, Ny = size(A_s)
    freqs = rfftfreq(N_samples, 1 / δt)
    fbins = [findmin(x -> abs(x - n * ω / 2π), freqs)[2] for n in 1:max_harmonic]
    fields = Array{ComplexF64, 4}(undef, max_harmonic, 4, Nx, Ny)
    for μ in 1:4
        A_ω_c = rfft(A_s[:, μ, :, :], 1)      # one component at a time (memory-conscious)
        for (n, idx) in enumerate(fbins)
            fields[n, μ, :, :] = A_ω_c[idx, :, :]
        end
        A_ω_c = nothing
        GC.gc()
        println("  component μ=$μ ($(complabels[μ])) done")
    end
    A_s = nothing
    GC.gc()
    cache = (
        samples_per_period = samples_per_period, fields = fields, fbins = fbins,
        ffund = [freqs[idx] / (ω / 2π) for idx in fbins],
        x_grid = collect(LinRange(-25w₀, 25w₀, Nx)),
        y_grid = collect(LinRange(-25w₀, 25w₀, Ny)),
    )
    serialize(cachefile, cache)
    println("cached → $cachefile")
end

# One figure per harmonic: 2×2 over the four components, each panel scaled to its
# own peak (component amplitudes differ by orders). Mirrors thomson_scattering.jl.
function plot_harmonic(n)
    n ≤ length(cache.fields[:, 1, 1, 1]) || error("harmonic $n exceeds cached max $(max_harmonic)")
    fig = Figure()
    Label(fig[0, :], @sprintf("Thomson scattering — %dω₁ (%.3f× fundamental)", n, cache.ffund[n]),
        fontsize = 16, font = :bold)
    for μ in 1:4
        field = real.(cache.fields[n, μ, :, :])
        cr = maximum(abs, field)
        gl = fig[cld(μ, 2), (μ - 1) % 2 + 1] = GridLayout()
        ax = Axis(gl[1, 1], width = 340, height = 340, xlabel = "x", ylabel = "y",
            title = @sprintf("%s  (peak %.2e)", complabels[μ], cr))
        hm = heatmap!(ax, cache.x_grid, cache.y_grid, field, colorrange = (-cr, cr), colormap = :seismic)
        Colorbar(gl[1, 2], hm, width = 12, height = 340)
    end
    resize_to_layout!(fig)
    out = stem * @sprintf("_h%d.png", n)
    save(out, fig)
    println("saved → $out")
    return basename(out)
end

# Labels match thomson's h1/h2 dashboard style (h1 = "ω₁ (fundamental)", h2 = "2ω₁ (2nd harmonic)").
const ORDINALS = Dict(2 => "2nd", 3 => "3rd")
hlabel(n) = n == 1 ? "ω₁ (fundamental)" :
            @sprintf("%dω₁ (%s harmonic)", n, get(ORDINALS, n, "$(n)th"))

# ── derived-artifact metadata for the results dashboard (research.314159265.dev) ──
# manifest.jl is already included; `dir`/`parent` resolved at the top.
let pid = parent[1]
    for n in harmonics
        plotname = plot_harmonic(n)
        if pid === nothing
            @warn "parent run manifest for $(basename(datafile)) has no run_id; skipping h$n sidecar"
        else
            write_derived(dir; kind = "h$n", label = hlabel(n), run_id = pid,
                plot = plotname, source = basename(datafile),
                datafile = basename(cachefile))   # shared harmonic-slice cache (all h<n>)
            println("derived sidecar → h$n (parent run $pid)")
        end
    end
end
