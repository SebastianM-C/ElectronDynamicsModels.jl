# scripts/coherence_curve.jl — decoherence vs boost across a γ→1⁺ ladder: how much of the
# rest-case coherent structure survives at each rung's backscatter line.
#
# Usage: julia --project=scripts scripts/coherence_curve.jl <ref_run.toml> <run.toml>...
#
# The physics: a cold disk shares one γ, so the decoherence channel is the FIXED transverse
# geometry measured against the SHRINKING line wavelength λ/n0 — phase spread σ_φ ∝ n0. The
# observable is the line-frequency map's coherent structure (the LG vortex) dissolving into
# 1/√N speckle as n0 grows. Maps MUST be at the line: off-line bins carry wing-leakage
# speckle that mimics decoherence (the reason rest_departure_linemaps exists).
#
# For every run (hmaps_<uuid>.jls next to its manifest): pick the harmonic entry nearest the
# exact line n = (1+β)/(1−β), then
#   C_E⊥, C_B⊥   complex normalized correlation with the REFERENCE run's line map — computed
#                only when the two screens share the grid (bridge runs have γ-scaled screens:
#                their C is skipped, NaN in the cache/curve)
#   contrast     speckle contrast std/mean of |E⊥| in the ring annulus r/halfw ∈ [0.25, 0.75]
#                — grid-free, valid across ALL screens
# Emits coherence_curve.png (C and contrast vs n0, log-x) + a `coherence_curve` chip on the
# reference run + coherence_curve_<reftag>.jls cache.

using Printf
using Serialization
using TOML
using CairoMakie

length(ARGS) >= 2 || error("usage: coherence_curve.jl <ref_run.toml> <run.toml>...")

function load_line_map(mfile)
    m = TOML.parsefile(mfile)
    cfg = m["config"]
    γ = Float64(get(cfg, "gamma", 1.0))
    β = γ > 1 ? sqrt(1 - 1 / γ^2) : 0.0
    nline = (1 + β) / (1 - β)
    dir = dirname(mfile)
    id = m["provenance"]["run_id"]
    hf = joinpath(dir, first(filter(x -> startswith(x, "hmaps_") && occursin(id, x), readdir(dir))))
    h = deserialize(hf)
    ks = collect(h.harmonics)
    k = argmin(abs.(Float64.(ks) .- nline))
    (; id, γ, nline, n_used = ks[k], ffund = h.ffund[k],
        M = h.fields_h[k, :, :, :], x = collect(h.x_grid), y = collect(h.y_grid), w₀ = h.w₀,
        eps = Float64(get(cfg, "gamma_eps", γ - 1)))
end

corr(M, R, comps) = begin
    num = sum(abs, sum(M[c, :, :] .* conj.(R[c, :, :])) for c in comps)
    den = sqrt(sum(abs2, @view M[comps, :, :]) * sum(abs2, @view R[comps, :, :]))
    den > 0 ? num / den : NaN
end

function ring_contrast(r)
    A = sqrt.(abs2.(r.M[1, :, :]) .+ abs2.(r.M[2, :, :]))
    hw = maximum(abs, r.x)
    ρ = [hypot(xx, yy) / hw for xx in r.x, yy in r.y]
    v = A[0.25 .<= ρ .<= 0.75]
    isempty(v) ? NaN : std(v) / mean(v)
end
using Statistics

ref = load_line_map(ARGS[1])
rows = NamedTuple[]
for a in ARGS[2:end]
    r = load_line_map(a)
    same_grid = length(r.x) == length(ref.x) && isapprox(maximum(r.x), maximum(ref.x); rtol = 1e-6)
    cE = same_grid ? corr(r.M, ref.M, 1:2) : NaN
    cB = same_grid ? corr(r.M, ref.M, 4:5) : NaN
    push!(rows, (; r.id, r.eps, r.nline, r.n_used, r.ffund, cE, cB, contrast = ring_contrast(r)))
    @printf("%s  ε=%-8.3g n=%-8.4g used=%-8.4g  C_E⊥=%-8.4g C_B⊥=%-8.4g contrast=%.4g\n",
        first(r.id, 8), r.eps, r.nline, Float64(r.n_used), cE, cB, last(rows).contrast)
end
ref_contrast = ring_contrast(ref)

fig = Figure(size = (1150, 460))
ax1 = Axis(fig[1, 1]; xscale = log10, title = "coherent-structure survival vs line order",
    xlabel = "n₀ = ω_bs/ω₁", ylabel = "|corr| with rest line map")
xs = [r.nline for r in rows]
scatterlines!(ax1, xs, [r.cE for r in rows]; label = "E⊥", color = :crimson)
scatterlines!(ax1, xs, [r.cB for r in rows]; label = "B⊥", color = :steelblue)
axislegend(ax1; position = :lb)
ax2 = Axis(fig[1, 2]; xscale = log10, title = "ring speckle contrast (σ/μ of |E⊥|)",
    xlabel = "n₀ = ω_bs/ω₁", ylabel = "contrast")
scatterlines!(ax2, xs, [r.contrast for r in rows]; color = :darkorange)
hlines!(ax2, [ref_contrast]; color = :gray, linestyle = :dash)
Label(fig[0, :], @sprintf("decoherence across the γ ladder — ref %s (rest), N runs = %d",
    first(ref.id, 8), length(rows)); fontsize = 17, font = :bold)
out = joinpath(dirname(abspath(ARGS[1])), "coherence_curve_$(ref.id).png")
save(out, fig)
println("saved → $out")

serialize(joinpath(dirname(abspath(ARGS[1])), "coherence_curve_$(first(ref.id, 8)).jls"),
    (; ref = (; ref.id, ref.nline, contrast = ref_contrast), rows))

repo_commit = try readchomp(`git rev-parse HEAD`) catch; "unknown" end
sidecar = Dict(
    "schema_version" => 1,
    "derived" => Dict(
        "depends_on" => [ref.id], "kind" => "coherence_curve",
        "label" => "decoherence vs line order",
        "plot" => basename(out), "source" => basename(ARGS[1]),
        "description" => "Coherent-structure survival across the γ→1⁺ ladder: complex " *
            "correlation of each rung's line-frequency map with the rest reference (grid-" *
            "matched runs) and ring speckle contrast (all runs), vs n₀ = (1+β)/(1−β). A cold " *
            "disk shares one γ, so decoherence is the fixed transverse geometry against the " *
            "shrinking line wavelength (σ_φ ∝ n₀); wing-leakage noise is excluded by mapping " *
            "AT the line (linemaps/bridge anchors).",
    ),
    "plot_params" => Dict("n_runs" => length(rows), "ref_contrast" => ref_contrast),
    "provenance" => Dict("host" => readchomp(`hostname`), "repo_commit" => repo_commit,
        "script" => "coherence_curve.jl",
        "timestamp" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time()))),
    "setup" => Dict("field" => "total"),
)
open(joinpath(dirname(abspath(ARGS[1])), "derived_coherence_curve_$(first(ref.id, 8)).toml"), "w") do io
    TOML.print(io, sidecar)
end
println("sidecar → derived_coherence_curve_$(first(ref.id, 8)).toml")
