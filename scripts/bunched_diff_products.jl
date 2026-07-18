# Common-disk difference products for prebunched campaigns: every cell shares the base run's
# deterministic sunflower disk (only the bunching Δz differs), so the complex difference
# F_cell − F_base cancels the common speckle and isolates the array's coherent contribution —
# raw-amplitude comparisons across cells are meaningless (each Δz set is an independent speckle
# realization; inverse-speckle-tomography report, Postscript II). Emits one `bunchdiff` chip per
# (bunched cell × harmonic): |ΔF_E| full screen, |ΔF_E| zoomed to the axial feature, and the
# difference phase ∠ΔEˣ; on-axis/median stats in [plot_params].
#
#   julia +release --project=scripts scripts/bunched_diff_products.jl <campaign_dir>
#
# The base cell is the manifest with config.bunch_nb == 0 (exactly one expected). Needs the
# hmaps_<id>.jls of every cell (present in campaign products and the published results tree).
# EDM_DIFF_ZOOM overrides the zoom half-width (w₀ units, default 1).

using TOML, Serialization, Statistics, Printf
using CairoMakie
using RunManifests   # write_derived, write_comparison, check_schema_version

const ZOOM_HW = parse(Float64, get(ENV, "EDM_DIFF_ZOOM", "1.0"))

function load_cells(dir)
    cells = []
    for f in sort(readdir(dir))
        (startswith(f, "run_") && endswith(f, ".toml")) || continue
        m = TOML.parsefile(joinpath(dir, f))
        check_schema_version(m; source = f)
        id = m["provenance"]["run_id"]
        hm = joinpath(dir, "hmaps_$id.jls")
        isfile(hm) || (println("skip $f — no hmaps"); continue)
        push!(cells, (; id, nb = get(m["config"], "bunch_nb", 0), l = get(m["config"], "bunch_l", 0),
            h = deserialize(hm)))
    end
    return cells
end

function main(dir)
    cells = load_cells(dir)
    base = [c for c in cells if c.nb == 0]
    length(base) == 1 || error("expected exactly one base cell (bunch_nb == 0), got $(length(base))")
    base = only(base)
    for c in cells
        c.nb == 0 && continue
        size(c.h.fields_h) == size(base.h.fields_h) && c.h.x_grid == base.h.x_grid ||
            error("grid mismatch between $(c.id[1:8]) and base — cells must share the screen")
        w₀ = base.h.w₀
        xs = c.h.x_grid ./ w₀
        zoom = findall(x -> abs(x) <= ZOOM_HW, xs)
        for (k, n) in enumerate(c.h.harmonics)
            D = c.h.fields_h[k, 1:3, :, :] .- base.h.fields_h[k, 1:3, :, :]
            amp = sqrt.(dropdims(sum(abs2, D; dims = 1); dims = 1))
            med = median(amp)
            ctr = size(amp, 1) ÷ 2
            onax = maximum(amp[ctr-2:ctr+3, ctr-2:ctr+3])
            out = joinpath(dir, @sprintf("inverse_thomson_bunchdiff_h%d_%s-%s.png",
                n, first(c.id, 8), first(base.id, 8)))
            fig = Figure(size = (1550, 520))
            ax1 = Axis(fig[1, 1], title = "|ΔF_E| full screen", xlabel = "x/w₀", ylabel = "y/w₀", aspect = 1)
            heatmap!(ax1, xs, xs, amp)
            ax2 = Axis(fig[1, 2], title = "|ΔF_E|  ±$(ZOOM_HW) w₀", xlabel = "x/w₀", aspect = 1)
            heatmap!(ax2, xs[zoom], xs[zoom], amp[zoom, zoom])
            ax3 = Axis(fig[1, 3], title = "∠ΔEˣ  ±$(ZOOM_HW) w₀", xlabel = "x/w₀", aspect = 1)
            heatmap!(ax3, xs[zoom], xs[zoom], angle.(D[1, zoom, zoom]),
                colormap = :twilight, colorrange = (-π, π))
            Label(fig[0, :], @sprintf(
                "bunched − base at %dω₁  (ℓ = %d, n_b = %d;  on-axis/median = %.2f)",
                n, c.l, c.nb, onax / med), fontsize = 18)
            save(out, fig)
            println("saved → $out")
            write_derived(
                dir; kind = "bunchdiff", label = "array signal vs base (ℓ = $(c.l))",
                run_id = [c.id, base.id], plot = basename(out),
                source = "hmaps_$(c.id).jls",
                setup = Dict("harmonic" => n),
                plot_params = Dict(
                    "on-axis |ΔF|" => round(onax; sigdigits = 3),
                    "median |ΔF|" => round(med; sigdigits = 3),
                    "on-axis/median" => round(onax / med; digits = 2),
                    "zoom hw/w₀" => ZOOM_HW,
                ),
                description = "Complex difference \$F_{cell} - F_{base}\$ of the E components at " *
                    "$(n)ω₁: all cells share the base's deterministic sunflower disk, so the " *
                    "common speckle cancels and the remainder is the bunching array's coherent " *
                    "contribution (never compare bunched cells by raw amplitude — each Δz set " *
                    "re-rolls the speckle). Left: full screen; middle: the axial feature; right: " *
                    "the difference phase. See inverse-speckle-tomography, Postscript II.",
            )
        end
        # Multi-parent chips surface through the comparison tab: declare each bunched cell
        # vs base (lone-run sides via `where`). Declarations go to the results-tree
        # comparisons/ sibling when present (the established home), else the campaign dir.
        campaign = basename(abspath(dir))
        compdir = isdir(joinpath(dir, "..", "comparisons")) ? joinpath(dir, "..", "comparisons") : dir
        write_comparison(
            compdir;
            label = "bunched ℓ = $(c.l) vs base (array signal)",
            differs = "prebunching Δz on the common disk",
            sides = [
                (; label = "ℓ = $(c.l), n_b = $(c.nb)", dir = campaign,
                    script = "inverse_thomson_scattering.jl",
                    where = Dict("bunch_nb" => c.nb, "bunch_l" => c.l)),
                (; label = "base (unbunched)", dir = campaign,
                    script = "inverse_thomson_scattering.jl",
                    where = Dict("bunch_nb" => 0)),
            ],
            filename = "comparison_$(campaign)_l$(c.l).toml",
        )
        println("declared → comparison_$(campaign)_l$(c.l).toml")
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? "." : ARGS[1])
end
