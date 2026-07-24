# Sweep-level boundary summaries for the emission-R3 figure: |ΔF̃| (LPWA − numeric) vs a₀ per
# harmonic — where does the analytic description bend — and h2/h1 vs a₀ for each side separately.
# Reads the a₀-matched hmaps of BOTH campaigns (numeric run_*.toml + hmaps_*.jls staged in one
# dir) and emits [summary] sidecars onto the LPWA campaign dir (axis = a0), where the
# per-pair diff sidecars from compare_lpwa_vs_thomson.jl already live.
#
#   julia +release --project=scripts scripts/boundary_summary.jl <lpwa_campaign_dir> <numeric_dir>
using TOML, Serialization, Printf
using RunManifests: write_summary
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

lpwa_dir, num_dir = ARGS[1], ARGS[2]

function cells(dir)
    out = []
    for f in sort(readdir(dir; join = true))
        (startswith(basename(f), "run_") && endswith(f, ".toml")) || continue
        m = TOML.parsefile(f)
        id = m["provenance"]["run_id"]
        hm = joinpath(dir, "hmaps_$id.jls")
        isfile(hm) || continue
        push!(out, (; id, a0 = Float64(m["config"]["a0"]), h = deserialize(hm)))
    end
    return sort(out; by = c -> c.a0)
end

L, T = cells(lpwa_dir), cells(num_dir)
[c.a0 for c in L] == [c.a0 for c in T] ||
    error("a0 ladders differ: $([c.a0 for c in L]) vs $([c.a0 for c in T])")

l2(x) = sqrt(sum(abs2, x))
rows = map(zip(L, T)) do (l, t)
    dF = Float64[]; aL = Float64[]; aT = Float64[]
    for n in (1, 2)
        kl = findfirst(==(n), l.h.harmonics)
        kt = findfirst(==(n), t.h.harmonics)
        (kl === nothing || kt === nothing) && error("harmonic $n absent at a0=$(l.a0)")
        push!(dF, l2(l.h.fields_h[kl, 1:3, :, :] .- t.h.fields_h[kt, 1:3, :, :]))
        push!(aL, l2(l.h.fields_h[kl, 1:3, :, :]))
        push!(aT, l2(t.h.fields_h[kt, 1:3, :, :]))
    end
    (; l.a0, dF, aL, aT, lid = l.id, tid = t.id)
end
a0s = [r.a0 for r in rows]
ids = [id for r in rows for id in (r.lid, r.tid)]

# ── 1. |ΔF̃| vs a₀ per harmonic — THE boundary curve (emission R3). ──────────────────────────
fig = Figure(size = (760, 520))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
    xlabel = "a₀", ylabel = "|ΔF̃|  (complex L2 over E, screen)",
    title = "LPWA − numeric: harmonic field difference vs a₀")
for (i, n) in enumerate((1, 2))
    scatterlines!(ax, a0s, [r.dF[i] for r in rows]; markersize = 10, label = "h$n")
end
lines!(ax, a0s, [rows[1].dF[1] * a0 / a0s[1] for a0 in a0s];
    color = :gray40, linestyle = :dash, label = "∝ a₀")
axislegend(ax; position = :lt)
out = joinpath(lpwa_dir, "boundary_dF_$(first(rows[1].lid, 8)).png")
save(out, fig)
write_summary(
    lpwa_dir; kind = "boundary_dF", label = "|ΔF̃| (LPWA − numeric) vs a₀",
    run_ids = ids, axis = "a0", plot = basename(out),
    plot_params = Dict(
        "a₀ values" => a0s,
        "|ΔF̃| h1" => [round(r.dF[1]; sigdigits = 3) for r in rows],
        "|ΔF̃| h2" => [round(r.dF[2]; sigdigits = 3) for r in rows],
        "rel-L2 h1" => [round(r.dF[1] / max(r.aT[1], eps()); sigdigits = 3) for r in rows],
        "rel-L2 h2" => [round(r.dF[2] / max(r.aT[2], eps()); sigdigits = 3) for r in rows]),
    description = "Complex L2 of the screen-field difference LPWA − numeric per harmonic, " *
        "against a₀ (log–log; dashed = ∝a₀ guide anchored at the smallest a₀). Where the " *
        "curve leaves the linear guide, the analytic LPWA description has bent — the " *
        "operating boundary the emission chapter's R3 figure reads off. rel-L2 per point " *
        "(vs the numeric amplitude) is in the plot parameters.",
)
println("summary → boundary_dF ($(length(rows)) pairs)")

# ── 2. h2/h1 vs a₀, each side separately — where the harmonic ratios diverge. ───────────────
fig = Figure(size = (760, 520))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
    xlabel = "a₀", ylabel = "|F̃(2ω)| / |F̃(ω)|",
    title = "second-to-first harmonic ratio — LPWA vs numeric")
scatterlines!(ax, a0s, [r.aL[2] / r.aL[1] for r in rows]; markersize = 10, label = "LPWA (analytic)")
scatterlines!(ax, a0s, [r.aT[2] / r.aT[1] for r in rows]; markersize = 10, label = "numeric")
axislegend(ax; position = :lt)
out = joinpath(lpwa_dir, "boundary_h2h1_$(first(rows[1].lid, 8)).png")
save(out, fig)
write_summary(
    lpwa_dir; kind = "boundary_h2h1", label = "h2/h1 vs a₀ — LPWA vs numeric",
    run_ids = ids, axis = "a0", plot = basename(out),
    plot_params = Dict(
        "a₀ values" => a0s,
        "h2/h1 LPWA" => [round(r.aL[2] / r.aL[1]; sigdigits = 3) for r in rows],
        "h2/h1 numeric" => [round(r.aT[2] / r.aT[1]; sigdigits = 3) for r in rows]),
    description = "Second-to-first harmonic amplitude ratio (screen L2 over E) per side. The " *
        "two curves separating marks which observable bends first as a₀ grows — the " *
        "empirical answer to Q-EM1 in the results interview.",
)
println("summary → boundary_h2h1 ($(length(rows)) pairs)")
