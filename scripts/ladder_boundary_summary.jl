# Ladder-wide numeric-vs-LPWA summaries for the one-go emission ladder (the Figs 7–8 campaign
# pair): (1) h2/h1 vs a₀ per side with power-law fits, (2) the per-harmonic phase-aligned
# rel-L2 ‖F_lpwa − F_num‖/‖F_num‖ vs a₀, (3) the pattern correlation |⟨F_num, F_lpwa⟩|/(‖·‖‖·‖)
# vs a₀. Every summary is written into BOTH campaign dirs (each with its own run first in
# depends_on, so the Summaries tab shows it on both cards). Definitions follow the ladder
# reports: F = hmaps fields_h[n, comps, :, :] / N_samples (rect reduction), ‖·‖ over components
# × pixels, E = comps 1:3 (B = 4:6 as the check); phase alignment e^{iβ}, β = arg⟨F_lpwa, F_num⟩.
#
#   julia --project=scripts scripts/ladder_boundary_summary.jl <lpwa_dir> [...] -- <numeric_dir> [...]
#
# The first dir of each side receives that side's sidecars + PNGs; extra dirs only contribute
# cells (per-lane dirs before a merge). EDM_SUMMARY_EXCLUDE=<id8>[,...] drops superseded runs.
# Fit ranges (EDM_FIT_LPWA / EDM_FIT_NUM_SMALL / EDM_FIT_NUM_BRANCH = "lo,hi") follow the ladder spec; cells at EDM_FIT_SKIP_A0 (default the crossover-sampling
# dense cells 0.13, 0.16, 0.25) are plotted but left out of the numeric fits.
using TOML, Serialization, Printf, LinearAlgebra
using RunManifests: write_summary
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

sep = findfirst(==("--"), ARGS)
(sep === nothing || sep == 1 || sep == length(ARGS)) &&
    error("usage: ladder_boundary_summary.jl <lpwa_dir> [...] -- <numeric_dir> [...]")
const LDIRS, NDIRS = ARGS[1:(sep - 1)], ARGS[(sep + 1):end]
const EXCLUDE = filter(!isempty, split(get(ENV, "EDM_SUMMARY_EXCLUDE", ""), ','))
const FIT_SKIP = parse.(Float64, split(get(ENV, "EDM_FIT_SKIP_A0", "0.13,0.16,0.25"), ','))
range_env(k, d) = Tuple(parse.(Float64, split(get(ENV, k, d), ',')))   # "lo,hi" in a₀
const R_LPWA = range_env("EDM_FIT_LPWA", "1e-5,1")
const R_SMALL = range_env("EDM_FIT_NUM_SMALL", "1e-5,0.01")
const R_BRANCH = range_env("EDM_FIT_NUM_BRANCH", "0.5,10")

function load(dirs)
    out = Dict{Float64, Any}()
    for d in dirs, f in sort(readdir(d))
        (startswith(f, "run_") && endswith(f, ".toml")) || continue
        m = TOML.parsefile(joinpath(d, f)); id = m["provenance"]["run_id"]
        first(id, 8) in EXCLUDE && continue
        h = joinpath(d, "hmaps_$id.jls")
        (isfile(h) && isfile(joinpath(d, "$id.reduced"))) || continue
        a0 = Float64(m["config"]["a0"])
        haskey(out, a0) && error("two runs at a0 = $a0 ($(out[a0].id), $id): exclude one")
        out[a0] = (; id, hm = deserialize(h), Ns = m["config"]["N_samples"])
    end
    return out
end
L, N = load(LDIRS), load(NDIRS)
isempty(L) && error("no LPWA cells"); isempty(N) && error("no numeric cells")

field(r, n, c) = r.hm.fields_h[findfirst(==(n), r.hm.harmonics), c, :, :] ./ r.Ns
peak(F) = maximum(sqrt.(dropdims(sum(abs2, F; dims = 1); dims = 1)))
ratio(r, c) = norm(field(r, 2, c)) / norm(field(r, 1, c))
peakratio(r, c) = peak(field(r, 2, c)) / peak(field(r, 1, c))

pw(p) = @sprintf("%.3f", p)
rng(r) = @sprintf("%g-%g", r...)   # fit range for legend labels
fmtg(x) = (e = log10(x); isinteger(e) && abs(e) ≥ 3 ? "10^{$(Int(e))}" : @sprintf("%g", x))   # TeX range bound   # fitted exponent for legend labels
# c as TeX mantissa×10^e for legend labels
tex(c) = (e = floor(Int, log10(c)); @sprintf("%.3f\\times10^{%d}", c / 10.0^e, e))

# Least squares of log10 y on log10 a0 over lo ≤ a0 ≤ hi → (p, c) with y ≈ c a0^p.
function powfit(a0s, ys, lo, hi; skip = Float64[])
    k = [i for i in eachindex(a0s) if lo ≤ a0s[i] ≤ hi && !any(isapprox(a0s[i], s) for s in skip)]
    length(k) ≥ 2 || return (NaN, NaN, 0)
    X = hcat(ones(length(k)), log10.(a0s[k])); b = X \ log10.(ys[k])
    return (b[2], 10^b[1], length(k))
end

# Write one summary into both campaign dirs; each copy lists its own side's runs first.
function emit(fig, kind, label; lids, nids, kw...)
    for (dir, ids) in ((LDIRS[1], vcat(lids, nids)), (NDIRS[1], vcat(nids, lids)))
        out = joinpath(dir, "$(kind)_$(first(ids[1], 8)).png")
        save(out, fig; px_per_unit = 2)
        write_summary(dir; kind, label, run_ids = ids, axis = "a0", plot = basename(out), kw...)
    end
    println("summary → $kind (both dirs)")
end

la, na = sort(collect(keys(L))), sort(collect(keys(N)))
pairs = [a for a in la if any(isapprox(a, b) for b in na)]
pn(a) = N[na[findfirst(b -> isapprox(a, b), na)]]
lids, nids = [L[a].id for a in la], [N[a].id for a in na]

# ── 1. h2/h1 vs a₀ per side, with the power-law fits ──────────────────────────────────────
for (fld, c) in ((:E, 1:3), (:B, 4:6))
    yl, yn = [ratio(L[a], c) for a in la], [ratio(N[a], c) for a in na]
    fl = powfit(la, yl, R_LPWA...)
    fs = powfit(na, yn, R_SMALL...; skip = FIT_SKIP)
    fb = powfit(na, yn, R_BRANCH...; skip = FIT_SKIP)
    ax_ = (fs[2] / fb[2])^(1 / (fb[1] - fs[1]))          # crossover of the two numeric segments
    cx = @sprintf("%.3g", ax_)
    fig = Figure(size = (1000, 750))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = L"a_0",
        ylabel = L"\Vert F(2\omega)\Vert\ /\ \Vert F(\omega)\Vert\ \ (%$(fld))", title = "second-to-first harmonic ratio ($fld, norm)")
    g = 10 .^ range(log10(minimum(vcat(la, na))), log10(maximum(vcat(la, na))), 200)
    scatter!(ax, la, yl; markersize = 12, color = Cycled(1), label = "LPWA (analytic)")
    scatter!(ax, na, yn; markersize = 12, color = Cycled(2), marker = :diamond, label = "numeric")
    lines!(ax, g, fl[2] .* g .^ fl[1]; color = Cycled(1), linestyle = :dash,
        label = L"\mathrm{LPWA\ fit}\ %$(rng(R_LPWA)):\ %$(tex(fl[2]))\ a_0^{%$(pw(fl[1]))}")
    gs = g[g .≤ ax_]; gb = g[g .≥ min(0.1, R_BRANCH[1])]
    lines!(ax, gs, fs[2] .* gs .^ fs[1]; color = Cycled(2), linestyle = :dot,
        label = L"\mathrm{numeric}\ %$(rng(R_SMALL)):\ %$(tex(fs[2]))\ a_0^{%$(pw(fs[1]))}\ \ (\mathrm{finite-}N\ \mathrm{residual,\ upper\ bound})")
    lines!(ax, gb, fb[2] .* gb .^ fb[1]; color = Cycled(2), linestyle = :dash,
        label = L"\mathrm{numeric}\ %$(rng(R_BRANCH)):\ %$(tex(fb[2]))\ a_0^{%$(pw(fb[1]))}")
    isfinite(ax_) && vlines!(ax, [ax_]; color = :gray50, linewidth = 1, label = L"\mathrm{laws\ meet\ at}\ a_0 = %$(cx)")
    axislegend(ax; position = :lt, framevisible = false, labelsize = 12)
    devs = [@sprintf("%g: %+.1f%%", a, 100 * (ratio(L[a], c) / (fl[2] * a^fl[1]) - 1)) for a in la if a > 1]
    pk = [@sprintf("%g: %.3f", a, peakratio(L[a], c) / ratio(L[a], c)) for a in la]
    emit(fig, "ladder_h2h1" * (fld === :B ? "_B" : ""),
        "h2/h1 vs a₀ with power-law fits" * (fld === :B ? " — B field" : "");
        lids, nids,
        plot_params = Dict(
            "LPWA a₀" => la, "LPWA h2/h1" => round.(yl; sigdigits = 4),
            "numeric a₀" => na, "numeric h2/h1" => round.(yn; sigdigits = 4),
            "LPWA fit (p, c, n)" => [round(fl[1]; digits = 4), round(fl[2]; sigdigits = 4), fl[3]],
            "numeric small-a₀ fit (p, c, n)" => [round(fs[1]; digits = 4), round(fs[2]; sigdigits = 4), fs[3]],
            "numeric a₀² branch (p, c, n)" => [round(fb[1]; digits = 4), round(fb[2]; sigdigits = 4), fb[3]],
            "numeric laws meet at a₀" => round(ax_; sigdigits = 3),
            "LPWA deviation from fit, a₀ > 1" => devs,
            "LPWA peak/norm ratio of h2/h1" => pk),
        description = "Second-to-first harmonic ratio in the **norm** definition " *
            "\$\\|\\tilde F(2\\omega)\\| / \\|\\tilde F(\\omega)\\|\$ ($fld; norm over components and " *
            "screen pixels, fields \$/N_\\mathrm{samples}\$, rect reduction), per side. The sentinel's " *
            "per-pixel **peak** ratio differs; the peak/norm factor per cell is in the plot " *
            "parameters. Log–log least-squares fits \$y = c\\,a_0^p\$: LPWA over \$$(fmtg(R_LPWA[1])) \\le a_0 \\le $(fmtg(R_LPWA[2]))\$ " *
            "(drawn over the full range; deviations of the \$a_0 > 1\$ cells in the plot parameters); " *
            "numeric \$$(fmtg(R_SMALL[1])) \\le a_0 \\le $(fmtg(R_SMALL[2]))\$ — the finite-\$N\$ residual of a coherent sum (cancelled by the focused beam's \$E_z\$-driven " *
            "longitudinal motion), which decreases with sampling density (\$\\times 492\$ from \$N = 400\$ to " *
            "\$10\\,000\$): an upper bound on the continuum value, not physics " *
            "— and the numeric \$a_0^2\$ branch over " *
            "\$$(fmtg(R_BRANCH[1])) \\le a_0 \\le $(fmtg(R_BRANCH[2]))\$ (dashed down to \$a_0 = 0.1\$). The vertical line marks where the fitted residual and " *
            "\$a_0^2\$ laws meet, \$a_0 = (c_1/c_2)^{1/(p_2-p_1)} \\approx $(@sprintf("%.3g", ax_))\$. The dense cells " *
            "\$a_0 = $(join(FIT_SKIP, ", "))\$ sample the crossover: plotted, not fitted." *
            (fld === :B ? " B-field check of the E-field figure." : ""))
end

# ── 2. per-harmonic phase-aligned rel-L2 and 3. pattern correlation, per a₀ pair ───────────
rows = map(pairs) do a
    l, n = L[a], pn(a)
    map((1, 2)) do h
        Fl, Fn = field(l, h, 1:3), field(n, h, 1:3)
        nn = norm(Fn); β = angle(dot(Fl, Fn))
        (; raw = norm(Fl .- Fn) / nn, phs = norm(cis(β) .* Fl .- Fn) / nn,
           corr = abs(dot(Fl, Fn)) / (nn * norm(Fl)), β = rad2deg(β))
    end
end
ids2 = (vcat([L[a].id for a in pairs]), vcat([pn(a).id for a in pairs]))

fig = Figure(size = (1000, 750))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = L"a_0",
    ylabel = L"\Vert F_\mathrm{LPWA}-F_\mathrm{num}\Vert\ /\ \Vert F_\mathrm{num}\Vert\ \ (E)", title = "LPWA − numeric, per harmonic (phase-aligned)")
for h in (1, 2)
    y = [r[h].phs for r in rows]; yr = [r[h].raw for r in rows]
    scatterlines!(ax, pairs, y; color = Cycled(h), markersize = 12, label = "h$h (phase-aligned)")
    differ = [!isapprox(yr[i], y[i]; rtol = 0.05) for i in eachindex(y)]
    any(differ) && scatter!(ax, pairs[differ], yr[differ]; color = (Makie.wong_colors()[h], 0.35),
        markersize = 12, marker = :circle, label = "h$h raw (where it differs)")
end
axislegend(ax; position = :lt, framevisible = false)
emit(fig, "ladder_relL2", "LPWA − numeric rel-L2 per harmonic vs a₀"; lids = ids2[1], nids = ids2[2],
    plot_params = Dict("a₀" => pairs,
        "h1 phase-aligned" => [round(r[1].phs; sigdigits = 3) for r in rows],
        "h2 phase-aligned" => [round(r[2].phs; sigdigits = 3) for r in rows],
        "h1 raw" => [round(r[1].raw; sigdigits = 3) for r in rows],
        "h2 raw" => [round(r[2].raw; sigdigits = 3) for r in rows],
        "β h1 [deg]" => [round(r[1].β; digits = 2) for r in rows],
        "β h2 [deg]" => [round(r[2].β; digits = 2) for r in rows]),
    description = "Relative L2 difference of the LPWA and numeric screen fields per harmonic, " *
        "\$\\|e^{i\\beta}\\tilde F_\\mathrm{LPWA} - \\tilde F_\\mathrm{num}\\| / \\|\\tilde F_\\mathrm{num}\\|\$ " *
        "(E, norm over components and pixels), after the optimal global phase " *
        "\$\\beta = \\arg\\langle \\tilde F_\\mathrm{LPWA}, \\tilde F_\\mathrm{num}\\rangle\$ (in the plot " *
        "parameters). Light markers show the unaligned value where it differs by more than 5 %.")

fig = Figure(size = (1000, 750))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = L"a_0",
    ylabel = L"1-|\langle F_\mathrm{num},F_\mathrm{LPWA}\rangle|\ /\ (\Vert F_\mathrm{num}\Vert\ \Vert F_\mathrm{LPWA}\Vert)\ \ (E)", title = "screen-pattern mismatch, per harmonic")
for h in (1, 2)
    scatterlines!(ax, pairs, max.(1 .- [r[h].corr for r in rows], 1e-16); color = Cycled(h),
        markersize = 12, label = "h$h")
end
axislegend(ax; position = :lt, framevisible = false)
emit(fig, "ladder_pattern", "LPWA vs numeric pattern correlation vs a₀"; lids = ids2[1], nids = ids2[2],
    plot_params = Dict("a₀" => pairs,
        "corr h1" => [round(r[1].corr; digits = 8) for r in rows],
        "corr h2" => [round(r[2].corr; digits = 8) for r in rows]),
    description = "Shape agreement of the two screen fields per harmonic, independent of overall " *
        "amplitude and phase: the normalized overlap " *
        "\$|\\langle \\tilde F_\\mathrm{num}, \\tilde F_\\mathrm{LPWA}\\rangle| / (\\|\\tilde F_\\mathrm{num}\\|\\,\\|\\tilde F_\\mathrm{LPWA}\\|)\$ " *
        "(E). Plotted as \$1 - \$ overlap on a log axis; the raw overlap is in the plot parameters.")
