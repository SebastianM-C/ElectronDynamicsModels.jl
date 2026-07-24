# Sweep-level boundary summaries for the emission-R3 figure: |ΔF̃| (LPWA − numeric) vs a₀ per
# harmonic — where does the analytic description bend — and h2/h1 vs a₀ for each side separately.
# Reads the a₀-matched hmaps of BOTH campaigns (numeric run_*.toml + hmaps_*.jls staged in one
# dir) and emits [summary] sidecars onto the LPWA campaign dir (axis = a0), where the
# per-pair diff sidecars from compare_lpwa_vs_thomson.jl already live.
#
#   julia +release --project=scripts scripts/boundary_summary.jl <lpwa_dir> <numeric_dir> \
#       [<lpwa_dir₂> <numeric_dir₂> ...]
# Extra pairs extend the ladder across ERAS (e.g. the June split-mode 1e-5..0.1 ladder +
# the 2026-07 total-mode 0.2..10 one). With >1 pair the summary kinds gain a `_full`
# suffix so the single-era cards stay alongside. Outputs land in the FIRST lpwa dir.
#
# Both field families are emitted: E (comps 1:3, kinds boundary_*) and B (comps 4:6, kinds
# boundary_*_B). At small a₀ the transverse E carries an a₀-independent DC-shelf leakage
# into the 2ω bin — identical in both models because it is an artifact of the SHARED
# window/FFT reduction — which dominates the E curves' small-a₀ end; B is leakage-free
# there and shows the physical story (the small-a₀ 2ω floor investigation's "read B" rule).
using TOML, Serialization, Printf
using RunManifests: write_summary
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

(length(ARGS) ≥ 2 && iseven(length(ARGS))) ||
    error("usage: boundary_summary.jl <lpwa_dir> <numeric_dir> [<lpwa_dir₂> <numeric_dir₂> ...]")
pairs = [(ARGS[i], ARGS[i + 1]) for i in 1:2:length(ARGS)]
lpwa_dir = pairs[1][1]                      # output dir for plots + sidecars
full = length(pairs) > 1
K(k) = full ? k * "_full" : k

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

l2(x) = sqrt(sum(abs2, x))

function ladder_rows(comps)
    rows = []
    for (ld, nd) in pairs
        L, T = cells(ld), cells(nd)
        [c.a0 for c in L] == [c.a0 for c in T] ||
            error("a0 ladders differ in ($ld, $nd): $([c.a0 for c in L]) vs $([c.a0 for c in T])")
        for (l, t) in zip(L, T)
            dF = Float64[]; aL = Float64[]; aT = Float64[]; dφ = Float64[]
            for n in (1, 2)
                kl = findfirst(==(n), l.h.harmonics)
                kt = findfirst(==(n), t.h.harmonics)
                (kl === nothing || kt === nothing) && error("harmonic $n absent at a0=$(l.a0)")
                FL = l.h.fields_h[kl, comps, :, :]
                FN = t.h.fields_h[kt, comps, :, :]
                # Optimal GLOBAL phase alignment before differencing: a constant offset between
                # the scripts' window/t₀ conventions rotates the whole complex map and would
                # read as an O(1) "difference" (the June cm_phi0 pairs sit at Δφ≈0 by
                # construction; the pre-PR#62 2026-07 era carries −7π/16). e^{iθ*} =
                # ⟨F_N,F_L⟩/|⟨F_N,F_L⟩|; θ* ships in the plot parameters — bookkeeping, not
                # physics.
                z = sum(conj.(FN) .* FL)
                ph = iszero(z) ? one(z) : z / abs(z)
                push!(dF, l2(FL .* conj(ph) .- FN))
                push!(dφ, rad2deg(angle(ph)))
                push!(aL, l2(FL))
                push!(aT, l2(FN))
            end
            push!(rows, (; l.a0, dF, aL, aT, dφ, lid = l.id, tid = t.id, era = basename(abspath(ld))))
        end
    end
    return sort!(rows, by = r -> r.a0)
end

for (fld, comps) in ((:E, 1:3), (:B, 4:6))
    rows = ladder_rows(comps)
    a0s = [r.a0 for r in rows]
    ids = [id for r in rows for id in (r.lid, r.tid)]
    sfx = fld === :B ? "_B" : ""
    bnote = fld === :B ?
        " B-FIELD variant: at small a₀ the transverse E carries an a₀-independent DC-shelf " *
        "leakage into the 2ω bin (identical in both models — an artifact of the shared " *
        "window/FFT reduction); B is leakage-free there and shows the physical behaviour." : ""

    # ── 1. THE boundary curve (emission R3): |ΔF̃| vs a₀ per harmonic — ABSOLUTE for a single
    # era; RELATIVE (dF/|F_num|) for the cross-era full ladder, because the eras' hmaps carry
    # different DFT amplitude conventions and share no a₀ point to rescale on.
    fig = Figure(size = (760, 520))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "a₀",
        ylabel = full ? "|ΔF̃| / |F̃_num|  (per pair, $fld, screen)" :
            "|ΔF̃|  (complex L2 over $fld, screen)",
        title = full ? "LPWA − numeric: relative harmonic $fld difference vs a₀" :
            "LPWA − numeric: harmonic $fld difference vs a₀")
    val(r, i) = full ? r.dF[i] / max(r.aT[i], eps()) : r.dF[i]
    eras = unique(r.era for r in rows)
    for (i, n) in enumerate((1, 2)), (ei, e) in enumerate(eras)
        sub = [r for r in rows if r.era == e]
        isempty(sub) && continue
        scatterlines!(ax, [r.a0 for r in sub], [val(r, i) for r in sub];
            markersize = 10, color = Cycled(i), marker = ei == 1 ? :circle : :rect,
            label = ei == 1 ? "h$n" : nothing)
    end
    full || lines!(ax, a0s, [rows[1].dF[1] * a0 / a0s[1] for a0 in a0s];
        color = :gray40, linestyle = :dash, label = "∝ a₀")
    axislegend(ax; position = :lt)
    out = joinpath(lpwa_dir, "boundary_dF$(sfx)_$(first(rows[1].lid, 8)).png")
    save(out, fig)
    write_summary(
        lpwa_dir; kind = K("boundary_dF" * sfx),
        label = (full ? "|ΔF̃| (LPWA − numeric) vs a₀ — full ladder, both eras" :
            "|ΔF̃| (LPWA − numeric) vs a₀") * (fld === :B ? " — B field" : ""),
        run_ids = ids, axis = "a0", plot = basename(out),
        plot_params = Dict(
            "a₀ values" => a0s,
            "|ΔF̃| h1" => [round(r.dF[1]; sigdigits = 3) for r in rows],
            "|ΔF̃| h2" => [round(r.dF[2]; sigdigits = 3) for r in rows],
            "rel-L2 h1" => [round(r.dF[1] / max(r.aT[1], eps()); sigdigits = 3) for r in rows],
            "rel-L2 h2" => [round(r.dF[2] / max(r.aT[2], eps()); sigdigits = 3) for r in rows],
            "Δφ h1 [deg]" => [round(r.dφ[1]; sigdigits = 3) for r in rows],
            "Δφ h2 [deg]" => [round(r.dφ[2]; sigdigits = 3) for r in rows]),
        description = (full ?
            "RELATIVE complex L2 |ΔF̃|/|F̃_num| per pair, LPWA − numeric per harmonic vs a₀ " *
            "(log–log), after optimal global phase alignment (the measured convention offset " *
            "Δφ is in the plot parameters). Circles = the June split-mode Ns=6000 era, " *
            "squares = the 2026-07 total-mode Ns=12160 era — the pair-internal relative " *
            "metric is the one quantity that stitches across the eras' different DFT " *
            "amplitude conventions." :
            "Complex L2 of the screen-field difference LPWA − numeric per harmonic, against " *
            "a₀ (log–log; dashed = ∝a₀ guide anchored at the smallest a₀), after optimal " *
            "global phase alignment per pair (Δφ in the plot parameters). rel-L2 per point " *
            "is in the plot parameters.") * bnote,
    )
    println("summary → boundary_dF$sfx ($(length(rows)) pairs)")

    # ── 2. h2/h1 vs a₀, each side separately — where the harmonic ratios diverge. ───────────
    fig = Figure(size = (760, 520))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "a₀", ylabel = "|F̃(2ω)| / |F̃(ω)|  ($fld)",
        title = "second-to-first harmonic ratio ($fld) — LPWA vs numeric")
    scatterlines!(ax, a0s, [r.aL[2] / r.aL[1] for r in rows]; markersize = 10, label = "LPWA (analytic)")
    scatterlines!(ax, a0s, [r.aT[2] / r.aT[1] for r in rows]; markersize = 10, label = "numeric")
    axislegend(ax; position = :lt)
    out = joinpath(lpwa_dir, "boundary_h2h1$(sfx)_$(first(rows[1].lid, 8)).png")
    save(out, fig)
    write_summary(
        lpwa_dir; kind = K("boundary_h2h1" * sfx),
        label = (full ? "h2/h1 vs a₀ — LPWA vs numeric, full ladder" :
            "h2/h1 vs a₀ — LPWA vs numeric") * (fld === :B ? " — B field" : ""),
        run_ids = ids, axis = "a0", plot = basename(out),
        plot_params = Dict(
            "a₀ values" => a0s,
            "h2/h1 LPWA" => [round(r.aL[2] / r.aL[1]; sigdigits = 3) for r in rows],
            "h2/h1 numeric" => [round(r.aT[2] / r.aT[1]; sigdigits = 3) for r in rows]),
        description = "Second-to-first harmonic amplitude ratio (screen L2 over $fld) per " *
            "side. The two curves separating marks which observable bends first as a₀ grows " *
            "— the empirical answer to Q-EM1 in the results interview." *
            (full ? " Spans both eras (June split-mode Ns=6000 + 2026-07 total-mode Ns=12160)." : "") *
            bnote,
    )
    println("summary → boundary_h2h1$sfx ($(length(rows)) pairs)")
end
