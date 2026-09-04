# scripts/alias_metrics.jl — lattice-alias metrics of a run's line map, drawn over the map.
#
# Usage: julia --project=scripts scripts/alias_metrics.jl <run_manifest.toml>...
#
# The sunflower disk of N electrons samples the line-frequency fringes only out to
#   r_alias ≈ λZ / (2 n₀ d),  d = Rmax·√(π/N)   (= 3.09·√N/n₀ w₀ at the campaign geometry)
# beyond which the coherent map gives way to a √N lattice-alias pedestal (the "halo"; see the
# screen-optics-alias report). Per run this script reads hmaps_<id>.jls, takes the harmonic
# nearest the backscatter line n₀ = (1+β)/(1−β), and measures
#   • the radial rms |E⊥| profile and r₅₀ (radius enclosing half the energy inside r_alias),
#   • the halo ONSET: first radius beyond which the azimuthal power at |m| > EDM_ALIAS_MMIN
#     (default 20) stays above half the total (smoothed over ±2 bins, 6 bins in a row),
#   • halo/core rms = rms|E⊥| beyond the onset over rms|E⊥| inside 0.8·min(r_alias, hw),
# and writes alias_metrics_<id>.png (the Eˣ line map with the predicted r_alias — dashed —
# and the measured onset — solid — drawn on it, next to the two profiles with the same
# radii marked) plus a derived_alias_metrics_<id8>.toml sidecar carrying the numbers, so the
# dashboard shows the check on the run's card and the N-ladder prediction (onset ∝ √N/n₀,
# halo/core ∝ 1/√N) is scored automatically. Runs whose predicted r_alias lies beyond the
# screen (low n₀) report no onset — the expected result there.
#
# ENV: EDM_ALIAS_MMIN (20), EDM_ALIAS_NTHETA (1024), EDM_ALIAS_DR (0.1 w₀), EDM_OUTDIR.
using Serialization
using Statistics
using FFTW
using Printf
using TOML
using CairoMakie
using ElectronDynamicsModels: ElectronDynamicsModels
using RunManifests: screen_halfwidth

const MMIN = parse(Int, get(ENV, "EDM_ALIAS_MMIN", "20"))
const NTHETA = parse(Int, get(ENV, "EDM_ALIAS_NTHETA", "1024"))
const DR = parse(Float64, get(ENV, "EDM_ALIAS_DR", "0.1"))
# Campaign geometry constants (inverse_thomson_scattering.jl): λZ = 2e5 λ · λ, Rmax = 3.25 w₀,
# w₀ = 75 λ ⇒ λZ = 35.56 w₀². Read from the manifest where recorded, else these.
isempty(ARGS) && error("usage: alias_metrics.jl <run_manifest.toml>...")

function line_index(harmonics, n0)
    ks = collect(Float64.(harmonics))
    return argmin(abs.(ks .- n0))
end

function analyze(mfile)
    m = TOML.parsefile(mfile)
    cfg, setup = m["config"], m["setup"]
    id = m["provenance"]["run_id"]; dir = dirname(abspath(mfile))
    hfs = filter(x -> startswith(x, "hmaps_") && occursin(id, x), readdir(dir))
    isempty(hfs) && (@warn "no hmaps for $id — skipped"; return nothing)
    h = deserialize(joinpath(dir, hfs[1]))
    w₀ = h.w₀
    γ = Float64(get(cfg, "gamma", 1.0)); β = γ > 1 ? sqrt(1 - 1 / γ^2) : 0.0
    n0 = (1 + β) / (1 - β)
    N = Int(cfg["N"])
    λ = Float64(m["laser"]["wavelength"]); Z = Float64(setup["Z"]); Rmax = Float64(setup["Rmax"])
    d = Rmax * sqrt(π / N)                       # mean sunflower spacing
    r_alias = λ * Z / (2 * n0 * d) / w₀          # Nyquist radius, w₀
    k = line_index(h.harmonics, n0)
    nline = Float64(h.harmonics[k])
    x = collect(h.x_grid) ./ w₀; y = collect(h.y_grid) ./ w₀
    hw = maximum(x); px = x[2] - x[1]
    M = h.fields_h[k, :, :, :]
    Ex = M[1, :, :]
    A = sqrt.(abs2.(M[1, :, :]) .+ abs2.(M[2, :, :]))
    ρ = [hypot(xx, yy) for xx in x, yy in y]

    # radial rms profile + azimuthal high-m fraction along circles
    ix(v) = clamp(round(Int, (v - x[1]) / px) + 1, 1, length(x))
    θ = range(0, 2π; length = NTHETA + 1)[1:(end - 1)]
    rs = collect(DR:DR:(hw * 0.98))
    prof = similar(rs); frac = similar(rs)
    for (j, r) in enumerate(rs)
        line = [Ex[ix(r * cos(t)), ix(r * sin(t))] for t in θ]
        P = abs2.(fft(line)); tot = sum(P)
        frac[j] = tot > 0 ? sum(P[(MMIN + 2):(NTHETA - MMIN)]) / tot : NaN
        prof[j] = sqrt(mean(abs2, line))
    end
    sm = [mean(frac[max(1, i - 2):min(end, i + 2)]) for i in eachindex(frac)]
    jo = findfirst(i -> all(sm[i:min(end, i + 5)] .> 0.5), eachindex(sm))
    onset = jo === nothing ? NaN : rs[jo]
    lim = min(r_alias, hw)
    wgt = prof .^ 2 .* rs; cum = cumsum(wgt .* (rs .< lim))
    r50 = cum[end] > 0 ? rs[findfirst(cum .>= 0.5 * cum[end])] : NaN
    core = sqrt(mean(abs2, prof[rs .< 0.8 * lim]))
    halo = isnan(onset) ? NaN : sqrt(mean(abs2, prof[rs .> onset]))
    halo_core = halo / core
    # ring peaks inside the clean radius (local maxima of the profile above 30 % of its max)
    peaks = [rs[i] for i in 2:(length(rs) - 1) if prof[i] > prof[i - 1] && prof[i] > prof[i + 1] &&
             rs[i] < lim && prof[i] > 0.3 * maximum(prof)]

    # ── figure: map with the two circles + the two profiles ──
    fig = Figure(size = (1500, 500))
    cap = quantile(vec(abs.(real.(Ex))), 0.995)
    ax1 = Axis(fig[1, 1]; title = @sprintf("Eˣ at %.4gω₁ (line n₀ = %.4g), N = %d", nline, n0, N),
        xlabel = "x / w₀", ylabel = "y / w₀", aspect = DataAspect())
    heatmap!(ax1, x, y, real.(Ex); colormap = :jet, colorrange = (-cap, cap),
        highclip = :magenta, lowclip = :cyan)
    circ(r) = (r .* cos.(θ), r .* sin.(θ))
    r_alias <= hw * √2 && lines!(ax1, circ(r_alias)...; color = :white, linestyle = :dash, linewidth = 2,
        label = @sprintf("predicted r_alias = %.2f w₀", r_alias))
    isnan(onset) || lines!(ax1, circ(onset)...; color = :black, linewidth = 2,
        label = @sprintf("measured onset = %.2f w₀", onset))
    lines!(ax1, circ(Rmax / w₀)...; color = :gray30, linestyle = :dot, linewidth = 1, label = "Rmax")
    axislegend(ax1; position = :lt, framevisible = true, backgroundcolor = (:white, 0.7))
    limits!(ax1, -hw, hw, -hw, hw)

    ax2 = Axis(fig[1, 2]; title = "radial rms |E⊥| of the line map", xlabel = "r / w₀", ylabel = "rms |E⊥|")
    lines!(ax2, rs, prof; color = :black)
    ax3 = Axis(fig[1, 3]; title = @sprintf("azimuthal power at |m| > %d", MMIN), xlabel = "r / w₀",
        ylabel = "fraction of power", limits = (nothing, (0, 1)))
    lines!(ax3, rs, frac; color = :gray60)
    lines!(ax3, rs, sm; color = :black, label = "smoothed")
    hlines!(ax3, [0.5]; color = :gray40, linestyle = :dot)
    for ax in (ax2, ax3)
        r_alias <= hw && vlines!(ax, [r_alias]; color = :steelblue, linestyle = :dash, linewidth = 2, label = "r_alias (pred)")
        isnan(onset) || vlines!(ax, [onset]; color = :firebrick, linewidth = 2, label = "onset (meas)")
        isnan(r50) || vlines!(ax, [r50]; color = :seagreen, linestyle = :dot, label = "r₅₀")
    end
    axislegend(ax2; position = :rt); axislegend(ax3; position = :lt)
    Label(fig[0, :], @sprintf("Lattice-alias check — %s: r_alias = 3.09·√N/n₀ = %.2f w₀ (screen ±%.1f), onset %s, halo/core rms %s, r₅₀ %.2f w₀",
        first(id, 8), r_alias, hw, isnan(onset) ? "none on screen" : @sprintf("%.2f w₀", onset),
        isnan(halo_core) ? "—" : @sprintf("%.2f", halo_core), r50); fontsize = 15, font = :bold)

    outdir = get(ENV, "EDM_OUTDIR", dir)
    mkpath(outdir)
    png = "alias_metrics_$(id).png"
    save(joinpath(outdir, png), fig)
    println("saved → $(joinpath(outdir, png))")

    repo_commit = try
        readchomp(`git -C $(pkgdir(ElectronDynamicsModels)) rev-parse HEAD`)
    catch
        "unknown"
    end
    sidecar = Dict(
        "schema_version" => 1,
        "derived" => Dict(
            "depends_on" => [id], "kind" => "alias_metrics", "label" => "lattice-alias check",
            "plot" => png, "source" => basename(mfile),
            "description" => "Line map ($(nline)ω₁ ≈ n₀ = $(round(n0; digits = 3))) with the predicted " *
                "lattice-alias radius r_alias = λZ/(2n₀d) (dashed) and the measured halo onset (solid; " *
                "azimuthal power at |m| > $MMIN exceeding half). " *
                @sprintf("N = %d: r_alias = %.2f w₀, onset %s, halo/core rms %s, r₅₀ = %.2f w₀. ",
                    N, r_alias, isnan(onset) ? "none on screen" : @sprintf("%.2f w₀", onset),
                    isnan(halo_core) ? "n/a" : @sprintf("%.2f", halo_core), r50) *
                "Prediction under the alias reading: onset ∝ √N/n₀ and halo/core ∝ 1/√N; a genuine " *
                "cold-disk decoherence would leave both N-independent.",
        ),
        "plot_params" => Dict{String, Any}(
            "n0" => n0, "line_harmonic" => nline, "N" => N, "screen_hw_w0" => hw,
            "lattice_spacing_w0" => d / w₀, "r_alias_pred_w0" => r_alias,
            "onset_meas_w0" => onset, "halo_core_rms" => halo_core, "r50_w0" => r50,
            "ring_peaks_w0" => peaks, "mmin" => MMIN, "dr_w0" => DR,
            "clean_fraction_of_screen" => min(1.0, r_alias / hw),
        ),
        "provenance" => Dict(
            "host" => readchomp(`hostname`), "repo_commit" => repo_commit,
            "script" => "alias_metrics.jl",
            "timestamp" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())),
        ),
    )
    scfile = joinpath(outdir, "derived_alias_metrics_$(first(id, 8)).toml")
    open(io -> TOML.print(io, sidecar), scfile, "w")
    println("sidecar → $scfile")
    @printf("%s  γ=%.3g n₀=%.4g N=%d hw=%.1f  r_alias=%.2f  onset=%s  halo/core=%s  r50=%.2f\n",
        first(id, 8), γ, n0, N, hw, r_alias, isnan(onset) ? "none" : @sprintf("%.2f", onset),
        isnan(halo_core) ? "n/a" : @sprintf("%.2f", halo_core), r50)
    return nothing
end

foreach(analyze, ARGS)
