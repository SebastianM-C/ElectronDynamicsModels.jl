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

D = abs.(A.M .- TB)                      # per-component |F_A − T(F_B)|
relL2 = [sqrt(sum(abs2, D[c, :, :]) / max(sum(abs2, A.M[c, :, :]), eps())) for c in 1:6]

pair = "$(first(A.id, 8))-$(first(B.id, 8))"
out = joinpath(A.dir, "xchk_dmap_$(pair).png")
plot_harmonic_grid(
    D, A.x, A.y;
    w₀ = A.w₀, labels = COMPLABELS, transform = identity,
    title = @sprintf("|Δ field| at %gω₁ — %s vs %s (T = %s)", Float64(A.n), first(A.id, 8), first(B.id, 8), tname),
    outfile = out,
)
println("saved → $out   transform=$tname   relL2=", join([@sprintf("%s:%.3g", COMPLABELS[c], relL2[c]) for c in 1:6], " "))

open(joinpath(A.dir, "derived_xchk_dmap_$(pair).toml"), "w") do io
    TOML.print(io, Dict(
        "schema_version" => 1,
        "derived" => Dict(
            "depends_on" => [A.id, B.id], "kind" => "xchk_dmap",
            "label" => "|Δ field| maps at $(A.n)ω₁",
            "plot" => basename(out), "source" => basename(ARGS[1]),
            "description" => "Per-component complex-difference magnitude |F_A − T(F_B)| at the " *
                "shared harmonic bin, best transform T = $(tname) (L2 on |E⊥|). The map-level " *
                "view of the crosscheck's powspec verdicts: structure in these panels is WHERE " *
                "the two runs disagree.",
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
