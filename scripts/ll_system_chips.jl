# System-view pickers + comparison declarations for LL campaigns.
#
# classical | Landau–Lifshitz is a VIEW of the same cell (the builder collapses each
# common-disk pair via VIEW_PARAMS), surfaced directly on the pair's `h<n>` field-map chips:
# the classical run's auto-emitted single-value sidecar is REPLACED by two sidecars that reuse
# the two runs' existing field-map PNGs (no re-render), varying only in setup.system. This
# works because LL pairs run total-mode cubes — `field` is constant within the kind, so
# `system` is the single varying setup key the builder renders as one picker (a second varying
# key would flatten the picker into a cross-product — the retired kind="sysview" lesson);
# split-mode pairs are guarded and skipped. Run AFTER any recolor pass: write_field_products
# re-emits the single-value sidecar this script replaces.
#
# The LL − classical diff (common-disk complex difference — cancels shared speckle, isolates
# the coherent RR imprint; the speckle report's bunched-diff pattern) is NOT a view of either
# run: it's emitted as 2-parent `diff` artifacts + a [comparison] declaration per campaign
# (differs = system, along = gamma or a0), surfacing on the comparison tab like the
# Newton-vs-RK4 kernel comparison, with ladder navigation along the campaign axis.
#
#   julia +release --project=scripts scripts/ll_system_chips.jl <campaign_dir>
using TOML, Serialization, Statistics, Printf
using ElectronDynamicsModels
using CairoMakie
using RunManifests

include(joinpath(@__DIR__, "harmonic_products.jl"))   # harmonic_field_style, COMPLABELS

function load_cells(dir)
    cells = []
    for f in sort(readdir(dir))
        (startswith(f, "run_") && endswith(f, ".toml")) || continue
        m = TOML.parsefile(joinpath(dir, f))
        id = m["provenance"]["run_id"]
        hm = joinpath(dir, "hmaps_$id.jls")
        isfile(hm) || continue
        c = m["config"]
        push!(cells, (; id, system = get(c, "system", "classical"), gamma = get(c, "gamma", 0),
            a0 = c["a0"], iters = get(c, "newton_iters", 2),
            n0 = round(Int, get(c, "backscatter_n0", 1)), h = deserialize(hm)))
    end
    return cells
end

function main(dir)
    cells = load_cells(dir)
    groups = Dict()
    for c in cells
        push!(get!(groups, (c.gamma, c.a0, c.iters), []), c)
    end
    style = harmonic_field_style(cap_mult = 4.0)
    pairs = []
    for ((γ, a0, it), g) in groups
        cl = findfirst(c -> c.system == "classical", g)
        ll = findfirst(c -> c.system == "ll", g)
        (cl === nothing || ll === nothing) && continue
        cl, ll = g[cl], g[ll]
        push!(pairs, (; γ, a0, it))
        w₀ = cl.h.w₀
        # Boosted pairs (n0 > 1) label bins in ω_bs = n0·ω₁ units, like the h<n> chips.
        hlabel(n) = cl.n0 == 1 ? "$(n)ω₁" : @sprintf("%.4g ω_bs", n / cl.n0)
        split = hasproperty(cl.h, :fields_far_h) && cl.h.fields_far_h !== nothing
        split && @warn "split-mode pair — a system toggle on h<n> would add a second varying " *
            "setup key (field) and flatten the picker; skipping the view merge" γ a0
        win = hasproperty(cl.h, :window) ? cl.h.window : "hann"
        for (k, n) in enumerate(cl.h.harmonics)
            # classical | Landau–Lifshitz directly on the pair's h<n> chip: two sidecars
            # reusing the runs' own field-map PNGs (write_field_products already rendered
            # both); the classical run's auto single-value sidecar is replaced below.
            if !split
                for (view, r) in (("classical", cl), ("ll", ll))
                    write_derived(
                        dir; kind = "h$n",
                        label = cl.n0 == 1 ? @sprintf("%dω₁ field maps", n) :
                            @sprintf("%.4g ω_bs field maps", n / cl.n0),
                        run_id = cl.id,
                        plot = @sprintf("inverse_thomson_field_h%d_%s.png", n, r.id),
                        source = "hmaps_$(r.id).jls",
                        setup = Dict("field" => "total", "system" => view),
                        plot_params = Dict("apodization" => win),
                        description = "Field harmonic maps at $(hlabel(n)) — toggle the " *
                            "electron model (classical | Landau–Lifshitz, common disk). The " *
                            "LL − classical difference lives on the campaign's comparison " *
                            "card. LL partner run: $(first(ll.id, 8)).",
                    )
                end
                rm(joinpath(dir, "derived_h$(n)_total_$(first(cl.id, 8)).toml"); force = true)
                println("h$n → system picker (classical|ll)")
            end
            # LL − classical (common disk) — a comparison artifact between the two runs, not a
            # view of either: 2 parents route it to the comparison card's matched cell.
            dmaps = ll.h.fields_h[k, :, :, :] .- cl.h.fields_h[k, :, :, :]
            out = joinpath(dir, @sprintf("inverse_thomson_sys_diff_h%d_%s.png", n, first(cl.id, 8)))
            plot_harmonic_grid(
                dmaps, cl.h.x_grid, cl.h.y_grid;
                w₀, labels = COMPLABELS, style...,
                title = @sprintf("γ=%g a₀=%g — diff at %s  (LL − classical, common disk)",
                    γ, a0, hlabel(n)),
                outfile = out,
            )
            write_derived(
                dir; kind = "diff", label = "LL − classical field maps",
                run_id = [cl.id, ll.id], plot = basename(out),
                source = "hmaps_$(ll.id).jls",
                setup = Dict("harmonic" => n),
                plot_params = Dict("rel-L2 (E)" =>
                    round(sqrt(sum(abs2, dmaps[1:3, :, :])) /
                          max(sqrt(sum(abs2, cl.h.fields_h[k, 1:3, :, :])), eps()), sigdigits = 3)),
                description = "Common-disk complex difference LL − classical at $(hlabel(n)) — " *
                    "cancels the shared speckle and isolates the coherent radiation-reaction " *
                    "imprint (rel-L2 ∝ a₀²γ, saturating at the √2 decorrelation ceiling).",
            )
            println("saved → $(basename(out))")
        end
    end
    write_system_comparisons(dir, pairs)
    return
end

# One [comparison] declaration per campaign (differs = system): a ladder along gamma (γ-ladder
# shape: one a₀) or along a0 (pinned to the majority γ), plus a single-cell pair card for any
# off-ladder pair (e.g. the γ=2000 satellite). Sides select raw runs by `where` — the builder
# resolves view-collapsed LL members from the pre-collapse run list.
function write_system_comparisons(dir, pairs)
    isempty(pairs) && return
    camp = basename(abspath(dir))
    side(sys, w) = Dict("label" => sys == "ll" ? "Landau–Lifshitz" : "classical",
        "dir" => camp, "where" => merge(Dict{String, Any}("system" => sys), w))
    decl(w, along, tag) = begin
        out = write_comparison(
            dir; label = "radiation reaction: Landau–Lifshitz vs classical",
            differs = "system", along,
            sides = [side("classical", w), side("ll", w)],
            filename = "comparison_$(camp)_system$(tag).toml",
        )
        println("comparison → $(basename(out))  (along = $along)")
    end
    its = unique(p.it for p in pairs)
    itpin = length(its) == 1 ? Dict{String, Any}("newton_iters" => its[1]) : Dict{String, Any}()
    γs, a0s = unique(p.γ for p in pairs), unique(p.a0 for p in pairs)
    if length(γs) > 1 && length(a0s) == 1
        decl(merge(itpin, Dict{String, Any}("a0" => a0s[1])), "gamma", "")
    else
        γmaj = argmax(γ -> count(p -> p.γ == γ, pairs), γs)
        mains = [p for p in pairs if p.γ == γmaj]
        length(mains) ≥ 2 && decl(merge(itpin, Dict{String, Any}("gamma" => γmaj)), "a0", "")
        for p in pairs
            (length(mains) ≥ 2 && p.γ == γmaj) && continue
            decl(merge(itpin, Dict{String, Any}("gamma" => p.γ, "a0" => p.a0)), "gamma",
                "_g$(Int(p.γ))a$(p.a0)")
        end
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? "." : ARGS[1])
end
