# scripts/compare_dmap.jl — |ΔE| difference maps for a mirror-crosscheck run pair.
#
# Usage: julia --project=scripts scripts/compare_dmap.jl RUN_A.toml RUN_B.toml [n]
#
# Loads both runs' hmaps caches, picks the harmonic entry nearest `n` (default 1) on each
# side, re-detects the best transverse transform T ∈ {identity, x-mirror, y-mirror, xy-flip}
# on the |h| maps (same convention as compare_mirror_runs.jl), and renders the per-component
# complex-difference magnitude |F_A − T(F_B)| on the shared grid — the map-level view of the
# powspec rel-L2 verdicts. Emits xchk_dmap_<idA8>-<idB8>.png + a 2-parent derived sidecar
# (kind "xchk_dmap") that joins the pair's standalone comparison card.
using ElectronDynamicsModels
using Serialization
using TOML
using Printf
using Dates
using CairoMakie

length(ARGS) >= 2 || error("usage: compare_dmap.jl RUN_A.toml RUN_B.toml [n]")
const NSEL = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
const COMPLABELS = ["Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ"]

function load_h(mfile, nsel)
    m = TOML.parsefile(mfile)
    id = m["provenance"]["run_id"]
    dir = dirname(mfile)
    h = deserialize(joinpath(dir, first(filter(x -> startswith(x, "hmaps_") && occursin(id, x), readdir(dir)))))
    ks = collect(h.harmonics)
    k = argmin(abs.(Float64.(ks) .- nsel))
    (; id, dir, M = h.fields_h[k, :, :, :], n = ks[k], x = collect(h.x_grid), y = collect(h.y_grid), w₀ = h.w₀)
end

A = load_h(abspath(ARGS[1]), NSEL)
B = load_h(abspath(ARGS[2]), NSEL)
size(A.M) == size(B.M) || error("grid mismatch: $(size(A.M)) vs $(size(B.M))")

# Transform detection on the transverse-E magnitude (sign-insensitive), L2-minimal.
transforms = (
    ("identity", M -> M), ("x-mirror", M -> reverse(M; dims = 2)),
    ("y-mirror", M -> reverse(M; dims = 3)), ("xy-flip", M -> reverse(reverse(M; dims = 2); dims = 3)),
)
magE(M) = sqrt.(abs2.(M[1, :, :]) .+ abs2.(M[2, :, :]))
ref = magE(A.M)
scores = [sum(abs2, magE(t(B.M)) .- ref) for (_, t) in transforms]
ti = argmin(scores)
tname, tf = transforms[ti]
TB = tf(B.M)

# June-convention presentation (compare_lpwa_vs_thomson.jl): per-panel |Δ| normalized to the
# numeric side's own peak (dimensionless colorbar), panel titles carry ‖Δ‖₂/‖F‖₂ — norms,
# not per-pixel ratios, so vortex nodal lines can't blow the display up. This script is the
# MIRROR-AWARE sibling: T is applied to side B before differencing.
using LinearAlgebra: norm

function describe(mfile)
    m = TOML.parsefile(mfile)
    s = basename(get(m["provenance"], "script", "?"))
    cfg = m["config"]
    base = startswith(s, "lpwa") ? "LPWA analytic" :
        "inverse " * (get(cfg, "accumulation_alg", "GPUKernelRK4") == "GPUKernelNewton" ? "Newton" : "RK4")
    a0 = get(cfg, "a0", nothing)
    eps = get(cfg, "gamma_eps", Float64(get(cfg, "gamma", 1.0)) - 1.0)
    d = base * (a0 === nothing ? "" : " a₀=$(a0)")
    eps == 0 || (d *= " ε=$(eps)")
    return d
end
descA, descB = describe(abspath(ARGS[1])), describe(abspath(ARGS[2]))

pair = "$(first(A.id, 8))-$(first(B.id, 8))"
xw, yw = A.x ./ A.w₀, A.y ./ A.w₀
fig = Figure(size = (1080, 680))
Label(fig[0, 1:3], @sprintf("%s vs %s — |ΔF| at %gω₁  (T = %s)", descA, descB, Float64(A.n), tname); fontsize = 17)
relL2 = Float64[]
for comp in 1:6
    a = A.M[comp, :, :]
    b = TB[comp, :, :]
    d = norm(a .- b)
    nrm = norm(a)
    push!(relL2, nrm == 0 ? 0.0 : d / nrm)
    r, c = fldmod1(comp, 3)
    max_a = maximum(abs, a)
    scale = iszero(max_a) ? 1.0 : max_a
    ax = Axis(fig[r, c][1, 1]; title = @sprintf("%s   ‖Δ‖/‖F‖=%.3g", COMPLABELS[comp], relL2[comp]),
        xlabel = "x/w₀", ylabel = "y/w₀", aspect = DataAspect())
    hm = heatmap!(ax, xw, yw, abs.(a .- b) ./ scale; colormap = :inferno)
    Colorbar(fig[r, c][1, 2], hm; label = "|Δ| / peak|F_A|")
end
out = joinpath(A.dir, @sprintf("compare_xchk_h%g_%s.png", Float64(A.n), pair))
save(out, fig)
println("saved → $out   transform=$tname   relL2=", join([@sprintf("%s:%.3g", COMPLABELS[c], relL2[c]) for c in 1:6], " "))

open(joinpath(A.dir, "derived_xchk_dmap_$(pair).toml"), "w") do io
    TOML.print(io, Dict(
        "schema_version" => 1,
        "derived" => Dict(
            "depends_on" => [A.id, B.id], "kind" => "comparison",
            "label" => "$(descA) vs $(descB) |ΔF|",
            "plot" => basename(out), "source" => basename(ARGS[1]),
            "description" => "Per-component |F_A − T(F_B)| at the shared $(A.n)ω₁ bin, each panel " *
                "normalized to that component's own peak (June compare_lpwa_vs_thomson convention); " *
                "panel titles carry ‖Δ‖₂/‖F‖₂. Best transverse transform T = $(tname).",
        ),
        "plot_params" => Dict("transform" => tname, "n" => Float64(A.n),
            ["relL2_$(COMPLABELS[c])" => round(relL2[c]; sigdigits = 3) for c in 1:6]...),
        "provenance" => Dict("host" => readchomp(`hostname`),
            "repo_commit" => (try readchomp(`git rev-parse HEAD`) catch; "unknown" end),
            "script" => "compare_dmap.jl",
            "timestamp_utc" => string(Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS"), "Z")),
        "setup" => Dict("field" => "total"),
    ))
end
println("sidecar → derived_xchk_dmap_$(pair).toml")
