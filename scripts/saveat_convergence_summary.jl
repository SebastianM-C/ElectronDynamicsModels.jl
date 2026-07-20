# Spline-convergence summary for ll_saveat_check: per regime (γ, a₀), the harmonic-map rel-L2
# between EDM_INTERP_SAVEAT levels — the reduced field maps are the science product, so their
# agreement across saveat IS the convergence metric (no cube needed). Verdict gate: production
# used saveat 64 at γ=100 (S cells) and 32 at γ=1000 (probes); a flat rel-L2 at the production
# level vs the finer level certifies those runs. Emits one summary chip (rel-L2 vs saveat,
# per-regime series) via RunManifests.write_summary on the campaign card.
#   julia +release --project=scripts scripts/saveat_convergence_summary.jl <ll_saveat_check_dir>
using TOML, Serialization, Printf, LinearAlgebra
using CairoMakie
using RunManifests: write_summary

function main(dir)
    cells = []
    for m in sort(readdir(dir))
        (startswith(m, "run_") && endswith(m, ".toml")) || continue
        t = TOML.parsefile(joinpath(dir, m)); c = t["config"]
        id = t["provenance"]["run_id"]; hm = joinpath(dir, "hmaps_$id.jls")
        isfile(hm) || continue
        # interp_saveat is recorded as a STRING (env-value verbatim) in these manifests.
        sv = get(c, "interp_saveat", 0)
        saveat = sv isa AbstractString ? parse(Int, sv) : Int(sv)
        push!(cells, (; id, γ = get(c, "gamma", 0), a0 = c["a0"], saveat, h = deserialize(hm)))
    end
    regimes = Dict()
    for c in cells
        push!(get!(regimes, (c.γ, c.a0), []), c)
    end
    # Collect measurements first: each is one (regime, coarse-saveat) rel-L2 vs that regime's
    # finest level. Grouped bar chart — x = regime, dodged by saveat level, LINEAR axes (the
    # 2-7% spread reads better linearly than on the old log2/log10; saveat is a small category
    # set, not a continuum).
    reglist = sort(collect(keys(regimes)); by = first)
    saveats = sort(unique(c.saveat for g in values(regimes) for c in g))   # dodge categories
    xs = Int[]; heights = Float64[]; dodge = Int[]; blabels = String[]
    rows = String[]
    for (ri, (γ, a0)) in enumerate(reglist)
        gs = sort(regimes[(γ, a0)]; by = c -> c.saveat)
        fine = last(gs)
        for c in gs[1:(end - 1)]
            rel = norm(vec(c.h.fields_h) .- vec(fine.h.fields_h)) /
                  max(norm(vec(fine.h.fields_h)), eps())
            push!(xs, ri); push!(heights, rel)
            push!(dodge, findfirst(==(c.saveat), saveats))
            push!(blabels, @sprintf("%.1f%%", 100rel))
            push!(rows, @sprintf("γ=%g a₀=%g: saveat %d vs %d → rel-L2 %.2e",
                γ, a0, c.saveat, fine.saveat, rel))
        end
    end
    fig = Figure(size = (820, 520))
    ax = Axis(fig[1, 1];
        xlabel = "regime", ylabel = "harmonic-map rel-L2 vs finest saveat (%)",
        title = "Trajectory-spline convergence — boosted regimes",
        xticks = (1:length(reglist), [@sprintf("γ=%g\na₀=%g", γ, a0) for (γ, a0) in reglist]))
    ylims!(ax, 0, 1.2 * 100 * maximum(heights))   # heights plotted in %, so scale the limit too
    cmap = (:dodgerblue, :crimson, :seagreen)
    barplot!(ax, xs, 100 .* heights; dodge = dodge, color = [cmap[d] for d in dodge],
        bar_labels = blabels, label_size = 12,
        strokewidth = 0.5, gap = 0.15, dodge_gap = 0.05)
    # Legend only for saveat levels that actually appear as a COARSE bar (the finest level is
    # the reference, never plotted — its dodge index never occurs, so skip it).
    used = sort(unique(dodge))
    elems = [PolyElement(color = cmap[i]) for i in used]
    Legend(fig[1, 2], elems, ["saveat $(saveats[i]) vs finest" for i in used], "coarse level",
        framevisible = false)
    out = joinpath(dir, "saveat_convergence.png")
    save(out, fig)
    println("saved → $(basename(out))")
    for r in rows; println("  ", r); end

    write_summary(dir; kind = "saveat-convergence",
        label = "spline convergence (rel-L2 vs saveat)", plot = basename(out),
        run_ids = [c.id for c in cells], axis = "interp_saveat",
        plot_params = Dict(rows[i] => "" for i in eachindex(rows)),
        description = "Harmonic-map rel-L2 between EDM_INTERP_SAVEAT levels per (γ, a₀) regime " *
            "— the reduced field maps are the science product, so their agreement across the " *
            "trajectory-spline resolution certifies convergence. Production used saveat 64 at " *
            "γ=100 (S cells) and 32 at γ=1000 (probes); a small rel-L2 to the finer level " *
            "validates those runs.")
    return
end

isempty(ARGS) && error("usage: saveat_convergence_summary.jl <ll_saveat_check_dir>")
main(ARGS[1])
