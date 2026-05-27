# Plot a chosen 4-potential component at harmonics of ω₁, in spatial-map form.
# Component μ is ARG2 (1=A⁰, 2=Aˣ, 3=Aʸ, 4=Aᶻ), default 3 (y).  Reads the
# sampling rate from the filename's "_spp<N>" tag (default 5).  Produces two
# figures: harmonics 1–2 and 3–6 (kept separate; amplitude scales differ by
# orders).  Caches the per-harmonic slices per component to
# "<stem>_hslices_mu<μ>.jls" (~20 MB) so later runs / re-plots are fast.
#
#   julia --project=scripts scripts/plot_harmonic_ladder.jl [file.jls] [μ]

using Serialization
using FFTW
using CairoMakie
using Printf

const c = 137.03599908330932
const ω = 0.057
const λ = 2π * c / ω
const w₀ = 75λ

const datafile = length(ARGS) ≥ 1 ? ARGS[1] : "A_rk4_400_N10000_Ns8000_spp16.jls"
const component = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 3
const complabel = get(Dict(1 => "A⁰", 2 => "Aˣ", 3 => "Aʸ", 4 => "Aᶻ"), component, "A$component")
const m = match(r"_spp(\d+)", datafile)
const samples_per_period = m === nothing ? 5 : parse(Int, m.captures[1])
const δt = 2π / ω / samples_per_period
const stem = replace(datafile, r"\.jls$" => "")
const cachefile = stem * "_hslices_mu$(component).jls"
const max_harmonic = 8

if isfile(cachefile)
    cache = deserialize(cachefile)
    println("Loaded slice cache $cachefile")
else
    println("Loading $datafile, component μ=$component ($complabel) — slow path…")
    A_s = deserialize(datafile)
    N_samples, _, Nx, Ny = size(A_s)
    A_c = A_s[:, component, :, :]
    A_s = nothing
    GC.gc()
    A_ω_c = rfft(A_c, 1)
    A_c = nothing
    GC.gc()
    freqs = rfftfreq(N_samples, 1 / δt)
    fields = Array{ComplexF64, 3}(undef, max_harmonic, Nx, Ny)
    fbins = Int[]
    ffund = Float64[]
    for n in 1:max_harmonic
        idx = findmin(x -> abs(x - n * ω / 2π), freqs)[2]
        fields[n, :, :] = A_ω_c[idx, :, :]
        push!(fbins, idx)
        push!(ffund, freqs[idx] / (ω / 2π))
    end
    cache = (
        component = component, complabel = complabel, samples_per_period = samples_per_period,
        fields = fields, fbins = fbins, ffund = ffund,
        nyq_mult = last(freqs) / (ω / 2π),
        x_grid = collect(LinRange(-25w₀, 25w₀, Nx)),
        y_grid = collect(LinRange(-25w₀, 25w₀, Ny)),
    )
    serialize(cachefile, cache)
    println("Cached harmonic slices → $cachefile")
end

function make_fig(orders, ncols, outsuffix)
    fig = Figure()
    Label(fig[0, :], "$datafile — $(cache.complabel) harmonics  ($(cache.samples_per_period) samples/period)",
        fontsize = 16, font = :bold)
    for (i, n) in enumerate(orders)
        r = cld(i, ncols)
        col = (i - 1) % ncols + 1
        field = real.(cache.fields[n, :, :])
        cr = maximum(abs, field)
        gl = fig[r, col] = GridLayout()
        ax = Axis(gl[1, 1], width = 340, height = 340, xlabel = "x", ylabel = "y",
            title = @sprintf("%dω₁  (%.2f× Nyq,  peak %.2e)", n, cache.ffund[n] / cache.nyq_mult, cr))
        hm = heatmap!(ax, cache.x_grid, cache.y_grid, field, colorrange = (-cr, cr), colormap = :seismic)
        Colorbar(gl[1, 2], hm, width = 12, height = 340)
        @printf("harmonic %d: bin %d  f/fund=%.3f  peak|real(A_ω)|=%.3e\n", n, cache.fbins[n], cache.ffund[n], cr)
    end
    resize_to_layout!(fig)
    out = stem * "_mu$(cache.component)" * outsuffix
    save(out, fig)
    println("saved → $out")
    return nothing
end

make_fig([1, 2], 2, "_h1-2.png")
make_fig([3, 4, 5, 6], 2, "_h3-6.png")

# ── derived-artifact metadata for the results dashboard (research.314159265.dev) ──
include(joinpath(@__DIR__, "manifest.jl"))
let dir = dirname(abspath(datafile))
    pid = find_parent_run(dir, basename(datafile))
    if pid === nothing
        @warn "no parent run manifest for $(basename(datafile)); skipping derived sidecars"
    else
        write_derived(dir; kind = "ladder_lo", label = "harmonic ladder · ω₁–2ω₁", run_id = pid,
            plot = basename(stem * "_mu$(component)_h1-2.png"), source = basename(datafile),
            setup = Dict("component" => component))
        write_derived(dir; kind = "ladder_hi", label = "harmonic ladder · 3ω₁–6ω₁", run_id = pid,
            plot = basename(stem * "_mu$(component)_h3-6.png"), source = basename(datafile),
            setup = Dict("component" => component))
        println("derived sidecars → harmonic ladder (component $component, parent run $pid)")
    end
end
