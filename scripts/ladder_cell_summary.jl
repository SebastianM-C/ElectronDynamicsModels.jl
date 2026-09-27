# Per-cell summaries of an a₀-ladder campaign for the dashboard's Summaries tab: (1) the production
# checks across a₀ (observer-window coverage, Coulomb-shelf head/tail, E-vs-B h2/h1 agreement,
# harmonic power falloff to n = 8) and (2) per-cell timings and venue (GPU model, cloud or local;
# trajectory / GPU-kernel / rest seconds). Manifest and sidecar reads only — no cube, no hmaps.
# Inputs per reduced cell (<uuid>.reduced present): run_<uuid>.toml ([config] a0, [window],
# [timing], [provenance] gpu_device / cloud_provider) and the shelf_sentinel sidecars
# derived_shelf_<id8>.toml + derived_falloff_<id8>.toml, and timeseries_<uuid>.jls for the
# window-end field flatness. Writes two [summary] sidecars
# (axis = a0) + PNGs into the FIRST dir; further dirs contribute cells of the same campaign
# (e.g. per-lane download dirs before the merge).
#
#   julia --project=scripts scripts/ladder_cell_summary.jl <campaign_dir> [<dir> ...]
#
# EDM_SUMMARY_EXCLUDE=<id8>[,<id8>...] drops superseded runs (e.g. a cell rerun under a new uuid).
using TOML, Printf, Serialization
using RunManifests: write_summary
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

isempty(ARGS) && error("usage: ladder_cell_summary.jl <campaign_dir> [<dir> ...]")
const OUT = ARGS[1]
const EXCLUDE = filter(!isempty, split(get(ENV, "EDM_SUMMARY_EXCLUDE", ""), ','))

sidecar(dir, kind, id8) = (f = joinpath(dir, "derived_$(kind)_$id8.toml"); isfile(f) ? TOML.parsefile(f) : nothing)
pp(sc) = sc === nothing ? Dict{String, Any}() : get(sc, "plot_params", Dict{String, Any}())

# Public venue label: GPU model + cloud/local, never a host name.
function venue(prov)
    dev = replace(get(prov, "gpu_device", "?"), "NVIDIA " => "", "H100 80GB HBM3" => "H100 SXM")
    return dev * (isempty(get(prov, "cloud_provider", "")) ? " · local" : " · cloud")
end

# Field flatness at the window end: the worst E-component range over the last 30 periods ÷ the
# pixel's burst excursion max|E − E_end| over samples AND components (one amplitude per pixel, so
# the weak Eᶻ is not amplified); max over timeseries pixels. ≪ 1 ⇒ the field has settled before the
# window closes (the window-integrity evidence for cells whose Coulomb-shelf tail fails physically).
function flatness(dir, id)
    f = joinpath(dir, "timeseries_$id.jls"); isfile(f) || return NaN
    ts = deserialize(f); w = max(1, ts.N_samples - 30 * ts.spp + 1):ts.N_samples
    return maximum(ts.pixels) do p
        A = max(maximum(abs, p.E .- p.E[end:end, :]), eps())
        maximum(c -> maximum(@view p.E[w, c]) - minimum(@view p.E[w, c]), 1:3) / A
    end
end
const FLAT_TOL = 1e-6   # window-end flatness bound for accepting a physical shelf tail

function load_cells(dirs)
    cells = []
    for dir in dirs, f in sort(readdir(dir))
        (startswith(f, "run_") && endswith(f, ".toml")) || continue
        m = TOML.parsefile(joinpath(dir, f))
        id = m["provenance"]["run_id"]; id8 = first(id, 8)
        (id8 in EXCLUDE || !isfile(joinpath(dir, "$id.reduced"))) && continue
        shelf, fall = pp(sidecar(dir, "shelf", id8)), pp(sidecar(dir, "falloff", id8))
        isempty(shelf) && @warn "no shelf sidecar" id8
        w, t = get(m, "window", Dict()), get(m, "timing", Dict())
        push!(cells, (;
            id, id8, a0 = Float64(m["config"]["a0"]), venue = venue(m["provenance"]),
            window_ok = get(w, "ok", missing), clipped = get(w, "electrons_clipped", missing),
            margin = min(get(w, "lead_margin_samples", 0), get(w, "tail_margin_samples", 0)),
            shelf, falloff = [get(fall, "P$(n)_over_P1", NaN) for n in 2:8], flat = flatness(dir, id),
            traj = get(t, "trajectories", NaN), kernel = get(t, "kernel", NaN),
            field = get(t, "field", NaN), total = get(t, "total", NaN)))
    end
    ids = [c.id for c in cells]
    allunique(ids) || error("a run appears in more than one input dir")
    return sort(cells; by = c -> c.a0)
end

dev(x) = abs(x - 1)
headdev(s) = max(dev(get(s, "head_min", NaN)), dev(get(s, "head_max", NaN)))
taildev(s) = max(dev(get(s, "tail_min", NaN)), dev(get(s, "tail_max", NaN)))
eb(s) = dev(get(s, "h2_over_h1_E", NaN) / get(s, "h2_over_h1_B", NaN))
const SHELF_TOL = 1e-3   # shelf_sentinel.jl's |−Eᶻ/(N/Z²) − 1| tolerance at both window ends

"""
    shelf_verdict(c) -> :pass | :physical | :fail

Classify one cell's production checks for the panel's marker colour. `c.shelf` is the
shelf_sentinel `[plot_params]` (keys `pass`, `head_min/max`, `tail_min/max`,
`tail_zeros_last_2pct`, `h2_over_h1_E/B`, `window_ok`); `c.window_ok`, `c.clipped` come from
the manifest's [window]; `c.a0` is the cell's a₀; `c.flat` the window-end field flatness (see `flatness`, bound
`FLAT_TOL`). `:physical` marks a sentinel FAIL that is the expected post-pulse physics — the
post-pulse charge state differs from rest (LPWA: displaced forward, tail > 1; numeric: expelled
and drifting, tail < 1), growing like a₀² — rather than a defect; everything else that fails
is `:fail`. Accepted by Sebastian for a₀ ≥ 5 (2026-09-27).
"""
function shelf_verdict(c)
    s = c.shelf
    isempty(s) && return :fail                       # no sentinel sidecar: unchecked
    string(get(s, "pass", "")) == "true" && return :pass
    # Physical only if the tail is the SOLE violated shelf criterion, the field has settled
    # (window-end flatness) and the run is clean; the window's own clipping is judged separately.
    clean = string(get(s, "repo_dirty", "")) == "false" && get(s, "tail_zeros_last_2pct", 1) == 0 &&
            headdev(s) < SHELF_TOL && taildev(s) ≥ SHELF_TOL && c.flat ≤ FLAT_TOL
    return c.a0 ≥ 5 && clean ? :physical : :fail
end
window_verdict(c) = c.window_ok === true && !(c.clipped isa Integer && c.clipped > 0) ? :pass : :fail
verdict(c) = window_verdict(c) === :fail ? :fail : shelf_verdict(c)

cells = load_cells(ARGS)
isempty(cells) && error("no reduced cells found in $(ARGS)")
a0s = [c.a0 for c in cells]; ids = [c.id for c in cells]; tag = cells[1].id8
vs = [verdict(c) for c in cells]; vsh = [shelf_verdict(c) for c in cells]; vw = [window_verdict(c) for c in cells]
vcol = Dict(:pass => :seagreen, :physical => :darkorange, :fail => :crimson)
xs = (; xscale = log10, xlabel = L"a_0")

# ── 1. production checks across a₀ ─────────────────────────────────────────────────────────
fig = Figure(size = (1000, 1250))
axw = Axis(fig[1, 1]; xs..., ylabel = "min window margin\n(samples)",
    title = "production checks per cell: $(basename(abspath(OUT)))")
scatter!(axw, a0s, [c.margin for c in cells]; color = [vcol[v] for v in vw], markersize = 13,
    marker = [c.window_ok === true ? :circle : :xcross for c in cells])
for c in cells
    (c.clipped isa Integer && c.clipped > 0) &&
        text!(axw, c.a0, c.margin; text = " $(c.clipped) clipped", fontsize = 11, align = (:left, :center))
end
axs = Axis(fig[2, 1]; xs..., yscale = log10, ylabel = L"|-E^z/(N/Z^2)-1|")
scatterlines!(axs, a0s, [headdev(c.shelf) for c in cells]; label = "window start", markersize = 10)
scatter!(axs, a0s, [taildev(c.shelf) for c in cells]; label = "window end",
    color = [vcol[v] for v in vsh], markersize = 13, marker = :diamond)
hlines!(axs, [SHELF_TOL]; color = :gray40, linestyle = :dash, label = "sentinel tolerance")
axislegend(axs; position = :lc, framevisible = false)
axe = Axis(fig[3, 1]; xs..., yscale = log10, ylabel = L"|(h_2/h_1)_E\ /\ (h_2/h_1)_B-1|")
scatterlines!(axe, a0s, max.([eb(c.shelf) for c in cells], 1e-16); markersize = 10, color = :purple)
axt = Axis(fig[4, 1]; xs..., yscale = log10, ylabel = "window-end flatness")
scatter!(axt, a0s, max.([c.flat for c in cells], 1e-16); color = [vcol[v] for v in vsh], markersize = 12)
hlines!(axt, [FLAT_TOL]; color = :gray40, linestyle = :dash)
for (c, v) in zip(cells, vsh)
    v === :physical && text!(axs, c.a0, taildev(c.shelf); fontsize = 11, align = (:right, :top),
        text = @sprintf("tail %.5f ", get(c.shelf, "tail_max", NaN)))
end
axf = Axis(fig[5, 1]; xs..., yscale = log10, ylabel = L"P_n/P_1\ \ (E,\ \mathrm{screen})")
for (i, n) in enumerate(2:8)
    y = [c.falloff[i] for c in cells]
    ok = isfinite.(y) .& (y .> 0)
    any(ok) && scatterlines!(axf, a0s[ok], y[ok]; label = L"n = %$(n)", markersize = 7, color = Cycled(i))
end
fig[5, 2] = Legend(fig, axf; framevisible = false)
linkxaxes!(axw, axs, axe, axt, axf)
for a in (axw, axs, axe, axt); a.xlabelvisible = false; end
ylims!(axs, nothing, 3SHELF_TOL * max(1, maximum(c -> taildev(c.shelf), cells) / SHELF_TOL))
Label(fig[6, 1], "window row: window check · shelf and flatness rows: shelf check — green pass · orange physical shelf, accepted · red fail";
    fontsize = 13, tellwidth = false)
out = joinpath(OUT, "ladder_checks_$tag.png")
save(out, fig; px_per_unit = 2)
write_summary(OUT; kind = "ladder_checks", label = "production checks vs a₀", run_ids = ids,
    axis = "a0", plot = basename(out),
    plot_params = Dict(
        "a₀" => a0s, "verdict" => string.(vs), "window" => string.(vw), "shelf" => string.(vsh),
        "window ok" => [string(c.window_ok) for c in cells],
        "electrons clipped" => [c.clipped isa Integer ? c.clipped : -1 for c in cells],
        "shelf |head−1|" => [round(headdev(c.shelf); sigdigits = 3) for c in cells],
        "shelf |tail−1|" => [round(taildev(c.shelf); sigdigits = 3) for c in cells],
        "window-end flatness" => [round(c.flat; sigdigits = 3) for c in cells],
        "P8/P1" => [round(c.falloff[end]; sigdigits = 3) for c in cells]),
    description = "Per-cell production checks against a₀, from each run's `[window]` and the " *
        "`shelf_sentinel.jl` sidecars. **Window**: the smaller of the lead/tail margins (observer " *
        "samples) between the burst and the recording-window edges; a cross marks `ok = false`, " *
        "with the clipped-electron count. **Coulomb shelf**: \$|{-E^z}/(N/Z^2) - 1|\$ at the first " *
        "and last observer samples (the static field of the \$N\$ charges at the screen, uncut by the " *
        "window); dashed = the sentinel's \$10^{-3}\$ tolerance. At \$a_0 \\ge 5\$ the Coulomb-shelf " *
        "criterion is expected to fail physically: the post-pulse charge state differs from rest " *
        "(LPWA: displaced forward, shelf \$> 1\$; numeric: expelled and drifting, shelf \$< 1\$). " *
        "Those cells are orange, *physical shelf, accepted*, with the measured tail value; their window " *
        "integrity is established by **window-end flatness**: the range over the last 30 periods ÷ the " *
        "pixel's peak \$|E - E_\\mathrm{end}|\$ over all components; max over the timeseries pixels " *
        "(dashed: \$10^{-6}\$). **E vs B**: the \$h_2/h_1\$ ratio read from E and from " *
        "B agree when the E side is free of DC-shelf leakage. **Falloff**: screen-integrated E " *
        "power in harmonic bands \$n \\pm 0.5\$ relative to \$n=1\$ (un-windowed spectrum; \$n = 8\$ " *
        "is the Nyquist band at 16 samples per period).")
println("summary → ladder_checks ($(length(cells)) cells) → $out")

# ── 2. timings and venue per cell ──────────────────────────────────────────────────────────
x = 1:length(cells)
rest = [max(c.total - c.traj - c.kernel, 0) for c in cells]
fig = Figure(size = (1100, 760))
ax = Axis(fig[1, 1]; ylabel = "minutes per cell", title = "cell wall time by phase and venue",
    xticks = (x, [@sprintf("%g\n%s", c.a0, replace(c.venue, " · " => "\n")) for c in cells]),
    xticklabelsize = 12, xlabel = L"a_0\ /\ \mathrm{GPU}\ /\ \mathrm{venue}")
stack = vcat(fill(1, length(x)), fill(2, length(x)), fill(3, length(x)))
barplot!(ax, repeat(x, 3), vcat([c.traj for c in cells], [c.kernel for c in cells], rest) ./ 60;
    stack, color = [(:steelblue, :darkorange, :gray60)[s] for s in stack])
Legend(fig[1, 2], [PolyElement(color = c) for c in (:steelblue, :darkorange, :gray60)],
    ["trajectories (CPU)", "field kernel (GPU)", "rest (setup, cube write)"]; framevisible = false)
out = joinpath(OUT, "ladder_timing_$tag.png")
save(out, fig; px_per_unit = 2)
write_summary(OUT; kind = "ladder_timing", label = "per-cell timings and venue", run_ids = ids,
    axis = "a0", plot = basename(out),
    plot_params = Dict(
        "a₀" => a0s, "venue" => [c.venue for c in cells],
        "trajectories [s]" => [round(c.traj; digits = 1) for c in cells],
        "kernel [s]" => [round(c.kernel; digits = 1) for c in cells],
        "field [s]" => [round(c.field; digits = 1) for c in cells],
        "total [s]" => [round(c.total; digits = 1) for c in cells]),
    description = "Wall time of each cell split into its phases from the manifest's `[timing]`: " *
        "trajectory generation (CPU), the GPU field kernel, and the rest (Julia setup, cube " *
        "serialization). The label under each bar gives \$a_0\$, the GPU model and whether the cell " *
        "ran on a cloud VM or a local machine. The reduce (harmonic maps, timeseries, checks) runs " *
        "after the cell, overlapped with the next cell's GPU phase, and is not included.")
println("summary → ladder_timing ($(length(cells)) cells) → $out")
