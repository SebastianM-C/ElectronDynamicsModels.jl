# Quantitative comparison of two field runs that should agree up to a transverse mirror —
# built for the γ=1 direct/inverse crosscheck (thomson_scattering.jl laser +z, screen +Z
# vs inverse_thomson_scattering.jl laser −z, screen ∓Z), but generic over any two runs on
# the same grid. The reversed laser keeps its helicity/m labels rather than parity-mapping
# them, so the two screens are expected to agree up to one of {identity, x-mirror,
# y-mirror, xy-flip} — DETECTED here (best transform on the |h1| maps), never assumed.
#
#   julia +release --project=scripts scripts/compare_mirror_runs.jl RUN_A.toml RUN_B.toml
#
# Per harmonic × component it reports the relative L2/L∞ difference of the |amplitude|
# maps under the winning transform (magnitudes are insensitive to the component sign
# flips a mirror induces), plus the transverse-E phase windings ℓ of both runs (a mirror
# or conjugation flips the sign — the winding pair diagnoses the convention relation),
# and the powspec caches' relative difference where present. Products: one diff PNG +
# 2-parent derived sidecar per harmonic (kind mirror_h<n>, harmonic selector), overlay
# powspec PNG + sidecar. EDM_OUTDIR overrides the output dir (default: RUN_A's dir).

using TOML, Serialization, LinearAlgebra, Statistics, Printf
using RunManifests: check_schema_version, write_derived
using ElectronDynamicsModels: ring_pixels, phase_winding_fit
using CairoMakie

length(ARGS) == 2 || error("usage: compare_mirror_runs.jl RUN_A.toml RUN_B.toml")
const toml_a, toml_b = abspath(ARGS[1]), abspath(ARGS[2])
const OUTDIR = get(ENV, "EDM_OUTDIR", dirname(toml_a))

run_id(m) = m["provenance"]["run_id"]
function load_side(path)
    m = TOML.parsefile(path)
    check_schema_version(m; source = basename(path))
    hm = get(m["outputs"], "harmonic_maps", "hmaps_$(run_id(m)).jls")
    h = deserialize(joinpath(dirname(path), hm))
    return (; m, h, id = run_id(m))
end
A, B = load_side(toml_a), load_side(toml_b)

# Human-readable side descriptor — physics identity, not run-id hex (which reads like a
# commit hash on the dashboard): "LPWA analytic a₀=0.3" / "inverse Newton a₀=0.3 ε=0.001".
function describe(m)
    s = basename(get(m["provenance"], "script", "?"))
    cfg = m["config"]
    base = startswith(s, "lpwa") ? "LPWA analytic" :
        "inverse " * (get(cfg, "accumulation_alg", "GPUKernelRK4") == "GPUKernelNewton" ?
            "Newton" : "RK4")
    a0 = get(cfg, "a0", nothing)
    ε = get(cfg, "gamma_eps", Float64(get(cfg, "gamma", 1.0)) - 1.0)
    d = base * (a0 === nothing ? "" : " a₀=$(a0)")
    ε == 0 || (d *= " ε=$(ε)")
    return d
end

size(A.h.fields_h) == size(B.h.fields_h) ||
    error("grid/harmonic mismatch: $(size(A.h.fields_h)) vs $(size(B.h.fields_h))")
collect(A.h.harmonics) == collect(B.h.harmonics) ||
    error("harmonic sets differ: $(A.h.harmonics) vs $(B.h.harmonics)")
harmonics = collect(A.h.harmonics)

const COMPLABELS = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")
const TRANSFORMS = (
    ("identity", M -> M),
    ("mirror-x", M -> reverse(M; dims = 1)),
    ("mirror-y", M -> reverse(M; dims = 2)),
    ("flip-xy", M -> reverse(M)),
)

relL2(x, y) = norm(x .- y) / max(norm(x), eps())
relL∞(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(x)), eps())

# ── Transform detection on the |h1| maps, summed over all six components ──
k1 = 1
scores = map(TRANSFORMS) do (name, T)
    s = sum(relL2(abs.(A.h.fields_h[k1, c, :, :]), T(abs.(B.h.fields_h[k1, c, :, :]))) for c in 1:6)
    (name, s)
end
best = argmin(last.(scores))
Tname, Tbest = TRANSFORMS[best]
println("transform scores (Σ_comp rel-L2 of |h$(harmonics[k1])| maps):")
foreach(((n, s),) -> @printf("   %-9s %.6f%s\n", n, s, n == Tname ? "   ← best" : ""), scores)

# ── Per-harmonic metrics under the winning transform + diff figures + sidecars ──
w₀ = A.h.w₀
for (k, n) in enumerate(harmonics)
    errs2 = zeros(6); errsI = zeros(6)
    fig = Figure(size = (1980, 720))
    for c in 1:6
        Ma = abs.(A.h.fields_h[k, c, :, :])
        Mb = Tbest(abs.(B.h.fields_h[k, c, :, :]))
        errs2[c] = relL2(Ma, Mb); errsI[c] = relL∞(Ma, Mb)
        for (row, M, ttl) in ((1, Ma, "A"), (2, Mb, "B∘$(Tname)"), (3, Ma .- Mb, "A − B∘T"))
            ax = Axis(fig[row, c]; title = row == 1 ? COMPLABELS[c] : "",
                ylabel = c == 1 ? ttl : "", xticklabelsvisible = false, yticklabelsvisible = false)
            heatmap!(ax, collect(A.h.x_grid) ./ w₀, collect(A.h.y_grid) ./ w₀, M;
                colormap = row == 3 ? :balance : :viridis)
        end
    end
    Label(fig[0, 1:6], @sprintf("|A(h%d)| — %s vs %s under %s   (rel-L2: %s)",
            n, first(A.id, 8), first(B.id, 8), Tname,
            join((@sprintf("%s=%.1e", COMPLABELS[c], errs2[c]) for c in 1:6), " "));
        fontsize = 16, font = :bold)
    out = joinpath(OUTDIR, "mirror_h$(n)_$(first(A.id, 8))-$(first(B.id, 8)).png")
    save(out, fig)
    println("saved → $out")
    @printf("h%-4d rel-L2 %s\n", n, join((@sprintf("%s=%.2e", COMPLABELS[c], errs2[c]) for c in 1:6), "  "))
    write_derived(
        OUTDIR; kind = "mirror_h$n", label = "mirror check h$n ($Tname)",
        run_id = [A.id, B.id], plot = basename(out), setup = Dict("harmonic" => n),
        plot_params = Dict(
            "transform" => Tname,
            (("relL2 " * COMPLABELS[c]) => round(errs2[c]; sigdigits = 3) for c in 1:6)...,
            (("relLinf " * COMPLABELS[c]) => round(errsI[c]; sigdigits = 3) for c in 1:6)...,
        ),
        description = "|amplitude| maps of both runs at $(n)ω₁ under the detected transform " *
            "($Tname) and their difference. Magnitudes are mirror-sign-insensitive; the " *
            "winding table in the log carries the phase-convention diagnosis.",
    )
end

# ── Phase windings on the transverse E components (ring at 0.3× the half-extent) ──
hw = maximum(abs, A.h.x_grid)
R = 0.3hw
tol = 0.75 * (A.h.x_grid[2] - A.h.x_grid[1])
idxs, az = ring_pixels(A.h.x_grid, A.h.y_grid, R; tol)
println("\nphase windings ℓ (ring R = $(round(R / w₀; sigdigits = 3)) w₀; mirror/conjugation flips the sign):")
for (k, n) in enumerate(harmonics), c in 1:2
    fa = phase_winding_fit(az, angle.(A.h.fields_h[k, c, :, :][idxs]);
        weights = abs.(A.h.fields_h[k, c, :, :][idxs]))
    fb = phase_winding_fit(az, angle.(B.h.fields_h[k, c, :, :][idxs]);
        weights = abs.(B.h.fields_h[k, c, :, :][idxs]))
    @printf("   h%-4d %s   ℓ_A = %+.3f   ℓ_B = %+.3f\n", n, COMPLABELS[c], fa.slope, fb.slope)
end

# ── Powspec caches (present when the runs were reduced by current harmonic_products) ──
psa = joinpath(dirname(toml_a), "powspec_$(A.id).jls")
psb = joinpath(dirname(toml_b), "powspec_$(B.id).jls")
if isfile(psa) && isfile(psb)
    pa, pb = deserialize(psa), deserialize(psb)
    fig = Figure(size = (1400, 700))
    ax = Axis(fig[1, 1]; xlabel = "frequency / ω₁", ylabel = "Σ_pixels |Â|²", yscale = log10,
        title = "power spectra — $(describe(A.m)) (solid) vs $(describe(B.m)) (dashed)")
    # Nyquist = spp/2 in ω₁ multiples ⇒ n(f) = f/fmax · spp/2 (same normalization as the
    # peak-finding convention; robust to the cache's raw frequency units).
    spp = Int(A.m["config"]["samples_per_period"])
    x = collect(pa.freqs) ./ maximum(pa.freqs) .* (spp / 2)
    for c in 1:6
        lines!(ax, x, max.(pa.ps[:, c], 1e-300); linewidth = 1.4, label = COMPLABELS[c])
        lines!(ax, x, max.(pb.ps[:, c], 1e-300); linewidth = 1.4, linestyle = :dash)
    end
    axislegend(ax; position = :rt, nbanks = 2)
    out = joinpath(OUTDIR, "mirror_powspec_$(first(A.id, 8))-$(first(B.id, 8)).png")
    save(out, fig)
    println("saved → $out")
    rel = [relL2(pa.ps[:, c], pb.ps[:, c]) for c in 1:6]
    @printf("powspec rel-L2 per comp: %s\n",
        join((@sprintf("%s=%.2e", COMPLABELS[c], rel[c]) for c in 1:6), "  "))
    write_derived(
        OUTDIR; kind = "mirror_powspec", label = "mirror check powspec",
        run_id = [A.id, B.id], plot = basename(out),
        plot_params = Dict(("relL2 " * COMPLABELS[c]) => round(rel[c]; sigdigits = 3) for c in 1:6),
        description = "Un-windowed power spectra of both runs overlaid (solid vs dashed). " *
            "Integrated spectra are transform-invariant, so any disagreement here is a real " *
            "physics/convention mismatch, not a mirror artifact.",
    )
else
    println("powspec caches not found beside both manifests — skipping the spectrum overlay")
end

# ── comparison-tab declaration: the [comparison] sidecar groups the pair and labels the
# entry in the dashboard's comparison view (the 2-parent mirror_* chips above are routed
# there automatically; this adds the human-readable framing, mirroring compare_lpwa's). ──
using RunManifests: write_comparison
let script_of(m) = get(m["provenance"], "script", "?"),
        out = write_comparison(
        OUTDIR; label = "mirror crosscheck: $(describe(A.m)) vs $(describe(B.m))",
        differs = "propagation direction + screen side (z-mirror pair)",
        sides = [
            (label = describe(A.m), dir = basename(dirname(toml_a)),
                script = script_of(A.m), where = Dict("run" => first(A.id, 8))),
            (label = describe(B.m), dir = basename(dirname(toml_b)),
                script = script_of(B.m), where = Dict("run" => first(B.id, 8))),
        ],
        filename = "comparison_mirror_$(first(A.id, 8))-$(first(B.id, 8)).toml",
    )

    println("comparison → ", basename(out))
end
