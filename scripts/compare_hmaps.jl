# scripts/compare_hmaps.jl — agreement between runs that must be the SAME physics: the electron
# sharding is exact by linearity, and the kernels are vendor-independent, so strong-scaling cells
# (D = 1/2/4/8) and the same cell on H200 / H100 / MI300X must agree to roundoff. This turns
# that claim into a number on each compared run.
#
# Usage:
#   julia --project=scripts scripts/compare_hmaps.jl <ref_run.toml> <run.toml>...
#   julia --project=scripts scripts/compare_hmaps.jl --sweep <sweep_id> <campaign_dir>
#       (all runs whose provenance.sweep_id matches, reference = fewest devices / first by time)
#
# For every run vs the reference: relative Frobenius distance ‖M − R‖/‖R‖ of the reduced
# harmonic maps (hmaps_<id>.jls; all harmonics and components, and per harmonic), max
# pointwise |M − R| / max|R|, and a figure — |E⊥| of the reference, of the run, and
# log₁₀ of the pointwise relative difference at the line harmonic — written next to the run
# as hmaps_diff_<run8>_vs_<ref8>.png with a derived_hmaps_agreement_<run8>.toml sidecar
# (depends_on both runs). Grids and harmonic lists must match; a mismatch is reported, not
# compared. Roundoff-level agreement is ~1e-12 for sharding and same-vendor reruns; expect
# ~1e-10…1e-8 across vendors (different FMA contraction / transcendental libraries).
using Serialization
using LinearAlgebra
using Statistics
using Printf
using TOML
using CairoMakie
using ElectronDynamicsModels: ElectronDynamicsModels

function load_run(mfile)
    m = TOML.parsefile(mfile)
    id = m["provenance"]["run_id"]; dir = dirname(abspath(mfile))
    hfs = filter(x -> startswith(x, "hmaps_") && occursin(id, x), readdir(dir))
    isempty(hfs) && error("no hmaps_*.jls for $id in $dir")
    h = deserialize(joinpath(dir, hfs[1]))
    D = Int(get(get(m, "sharding", Dict()), "electrons", get(get(m, "timing", Dict()), "n_devices", 1)))
    dev = String(something(get(get(m, "gpu", Dict()), "device", nothing), "?"))
    γ = Float64(get(m["config"], "gamma", 1.0)); β = γ > 1 ? sqrt(1 - 1 / γ^2) : 0.0
    n0 = (1 + β) / (1 - β)
    ts = String(get(m["provenance"], "timestamp_utc", get(m["provenance"], "timestamp", "")))
    return (; m, mfile, dir, id, h, D, dev, n0, ts)
end

args = copy(ARGS)
runs = if !isempty(args) && args[1] == "--sweep"
    length(args) == 3 || error("usage: --sweep <sweep_id> <campaign_dir>")
    sweep, dir = args[2], args[3]
    ms = [joinpath(dir, f) for f in sort(readdir(dir)) if startswith(f, "run_") && endswith(f, ".toml")]
    sel = filter(mf -> get(TOML.parsefile(mf)["provenance"], "sweep_id", "") == sweep, ms)
    isempty(sel) && error("no runs with sweep_id = $sweep in $dir")
    rs = load_run.(sel)
    sort(rs; by = r -> (r.D, r.ts))
else
    length(args) >= 2 || error("usage: compare_hmaps.jl <ref_run.toml> <run.toml>... | --sweep <id> <dir>")
    load_run.(args)
end
ref = runs[1]
@printf("reference: %s  D=%d  %s\n", first(ref.id, 8), ref.D, ref.dev)
R = ref.h.fields_h
kline = argmin(abs.(collect(Float64.(ref.h.harmonics)) .- ref.n0))
x = collect(ref.h.x_grid) ./ ref.h.w₀; y = collect(ref.h.y_grid) ./ ref.h.w₀
repo_commit = try
    readchomp(`git -C $(pkgdir(ElectronDynamicsModels)) rev-parse HEAD`)
catch
    "unknown"
end
println("run       D  device                     rel ‖Δ‖/‖R‖   max|Δ|/max|R|   per-harmonic")
for r in runs[2:end]
    M = r.h.fields_h
    if size(M) != size(R) || collect(r.h.harmonics) != collect(ref.h.harmonics) ||
            length(r.h.x_grid) != length(ref.h.x_grid)
        @printf("%s  %d  %-25s  NOT COMPARED — grid %s vs %s, harmonics %s vs %s\n", first(r.id, 8), r.D,
            first(r.dev, 25), string(size(M)), string(size(R)), string(collect(r.h.harmonics)), string(collect(ref.h.harmonics)))
        continue
    end
    rel = norm(M .- R) / norm(R)
    maxrel = maximum(abs, M .- R) / maximum(abs, R)
    per_h = [norm(M[k, :, :, :] .- R[k, :, :, :]) / norm(R[k, :, :, :]) for k in axes(R, 1)]
    @printf("%s  %d  %-25s  %.3e      %.3e     %s\n", first(r.id, 8), r.D, first(r.dev, 25), rel, maxrel,
        join([@sprintf("%.1e", v) for v in per_h], " "))

    Aref = sqrt.(abs2.(R[kline, 1, :, :]) .+ abs2.(R[kline, 2, :, :]))
    Arun = sqrt.(abs2.(M[kline, 1, :, :]) .+ abs2.(M[kline, 2, :, :]))
    dif = sqrt.(abs2.(M[kline, 1, :, :] .- R[kline, 1, :, :]) .+ abs2.(M[kline, 2, :, :] .- R[kline, 2, :, :]))
    ldiff = log10.(max.(dif ./ maximum(Aref), 1e-17))
    fig = Figure(size = (1500, 480))
    cmax = maximum(Aref)
    ax1 = Axis(fig[1, 1]; title = @sprintf("|E⊥| reference %s (D=%d, %s)", first(ref.id, 8), ref.D, first(ref.dev, 18)),
        xlabel = "x / w₀", ylabel = "y / w₀", aspect = DataAspect())
    heatmap!(ax1, x, y, Aref; colormap = :viridis, colorrange = (0, cmax))
    ax2 = Axis(fig[1, 2]; title = @sprintf("|E⊥| run %s (D=%d, %s)", first(r.id, 8), r.D, first(r.dev, 18)),
        xlabel = "x / w₀", aspect = DataAspect())
    heatmap!(ax2, x, y, Arun; colormap = :viridis, colorrange = (0, cmax))
    ax3 = Axis(fig[1, 3]; title = @sprintf("log₁₀ |Δ|/max|R| at the line  (rel ‖Δ‖/‖R‖ = %.2e)", rel),
        xlabel = "x / w₀", aspect = DataAspect())
    hm = heatmap!(ax3, x, y, ldiff; colormap = :magma, colorrange = (-16, 0))
    Colorbar(fig[1, 4], hm)
    png = "hmaps_diff_$(first(r.id, 8))_vs_$(first(ref.id, 8)).png"
    outdir = get(ENV, "EDM_OUTDIR", r.dir)
    mkpath(outdir)
    save(joinpath(outdir, png), fig)
    sidecar = Dict(
        "schema_version" => 1,
        "derived" => Dict(
            "depends_on" => [r.id, ref.id], "kind" => "hmaps_agreement",
            "label" => "agreement with $(first(ref.id, 8))", "plot" => png, "source" => basename(r.mfile),
            "description" => @sprintf("Reduced harmonic maps of this run (D=%d, %s) vs reference run %s (D=%d, %s): relative Frobenius distance %.3e over all harmonics/components, max pointwise |Δ|/max|R| = %.3e. Same physics ⇒ roundoff (~1e-12 sharding / same vendor, ~1e-10…1e-8 across vendors).",
                r.D, r.dev, first(ref.id, 8), ref.D, ref.dev, rel, maxrel),
        ),
        "plot_params" => Dict{String, Any}(
            "reference_run" => ref.id, "rel_frobenius" => rel, "max_pointwise_rel" => maxrel,
            "per_harmonic_rel" => per_h, "harmonics" => collect(Float64.(ref.h.harmonics)),
            "devices_run" => r.D, "devices_ref" => ref.D, "device_run" => r.dev, "device_ref" => ref.dev,
        ),
        "provenance" => Dict(
            "host" => readchomp(`hostname`), "repo_commit" => repo_commit, "script" => "compare_hmaps.jl",
            "timestamp" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())),
        ),
    )
    scfile = joinpath(outdir, "derived_hmaps_agreement_$(first(r.id, 8)).toml")
    open(io -> TOML.print(io, sidecar), scfile, "w")
    println("  → $(joinpath(outdir, png))  +  $(basename(scfile))")
end
