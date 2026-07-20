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
    fig = Figure(size = (760, 520))
    ax = Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "EDM_INTERP_SAVEAT",
        ylabel = "harmonic-map rel-L2 vs finest saveat",
        title = "Trajectory-spline convergence — boosted regimes")
    palette = (:crimson, :dodgerblue, :seagreen, :darkorange)
    rows = String[]
    for (i, ((γ, a0), g)) in enumerate(sort(collect(regimes); by = first))
        gs = sort(g; by = c -> c.saveat)
        fine = last(gs)                                       # highest saveat = reference
        pts = NTuple{2, Float64}[]
        for c in gs[1:(end - 1)]
            num = norm(vec(c.h.fields_h) .- vec(fine.h.fields_h))
            den = max(norm(vec(fine.h.fields_h)), eps())
            rel = num / den
            push!(pts, (Float64(c.saveat), rel))
            push!(rows, @sprintf("γ=%g a₀=%g: saveat %d vs %d → rel-L2 %.2e",
                γ, a0, c.saveat, fine.saveat, rel))
        end
        isempty(pts) && continue
        col = palette[(i - 1) % length(palette) + 1]
        scatterlines!(ax, first.(pts), last.(pts); color = col, markersize = 12,
            label = @sprintf("γ=%g, a₀=%g", γ, a0))
    end
    axislegend(ax; position = :rt)
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
