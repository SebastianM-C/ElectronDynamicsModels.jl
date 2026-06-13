# Quantitative comparison of the analytic-LPWA radiated field against the
# ODE-solved Thomson field, harmonic by harmonic. Driven off the two run
# manifests: the hmaps `.jls` is resolved by relative path from [outputs], and
# the [laser]/[config]/[setup] axes are cross-checked so we never difference two
# runs that aren't the same physical scenario on the same grid.
#
#   julia +release --project=scripts scripts/compare_lpwa_vs_thomson.jl RUN_LPWA.toml RUN_THOMSON.toml
#
# Each manifest's [outputs] must point at an hmaps file (lpwa writes `harmonic_maps`
# directly; for a recovered Thomson run, drop `hmaps_<run_id>.jls` beside its toml).

using TOML, Serialization, LinearAlgebra, Printf
using RunManifests: check_schema_version, write_derived
using CairoMakie

const lpwa_toml, thom_toml = ARGS[1], ARGS[2]
const OUTDIR = get(ENV, "EDM_OUTDIR", ".")

La = TOML.parsefile(lpwa_toml)
Ta = TOML.parsefile(thom_toml)

# Refuse a manifest layout we don't understand before reading any keys from it.
check_schema_version(La; source = "LPWA manifest")
check_schema_version(Ta; source = "Thomson manifest")

run_id(m, path) = get(
    get(m, "provenance", Dict()), "run_id",
    replace(basename(path), r"^run_" => "", r"\.toml$" => "")
)

# Resolve the reduced hmaps .jls relative to the manifest: prefer [outputs].harmonic_maps,
# else the conventional hmaps_<run_id>.jls beside the toml (recovered Thomson maps).
function resolve_hmaps(m, path)
    dir = dirname(abspath(path))
    out = get(m, "outputs", Dict())
    haskey(out, "harmonic_maps") && return joinpath(dir, out["harmonic_maps"])
    cand = joinpath(dir, "hmaps_$(run_id(m, path)).jls")
    isfile(cand) && return cand
    error(
        "$(basename(path)): no [outputs].harmonic_maps and no $(basename(cand)) beside it — " *
            "run the Thomson hmaps recovery first"
    )
end

function check_compatible(a, b)
    laser_params = ("wavelength", "w0", "p", "m", "pol", "profile", "a0", "phi0")
    config_params = ("samples_per_period", "Nx", "Ny", "N", "N_samples")
    # Per-key (NOT whole-dict): the sections carry script-/schema-specific extras — e.g. a
    # pre-dedup Thomson run keeps Nx/N/… in [setup] while the new LPWA [setup] has only the
    # window+depth. Compare the physical axes both manifests share.
    setup_params = ("τi", "τf", "Rmax", "Z")
    for (sec, keys) in (("laser", laser_params), ("config", config_params), ("setup", setup_params))
        for k in keys
            a[sec][k] == b[sec][k] || error("[$sec].$k differs: $(a[sec][k]) vs $(b[sec][k])")
        end
    end
    return nothing
end

check_compatible(La, Ta)

L = deserialize(resolve_hmaps(La, lpwa_toml))   # (; fields_h, harmonics, ffund, x_grid, y_grid, w₀)
T = deserialize(resolve_hmaps(Ta, thom_toml))

# Belt-and-suspenders: the maps themselves must be on a common grid with the same bins.
@assert L.harmonics == T.harmonics "harmonic sets differ: $(L.harmonics) vs $(T.harmonics)"
@assert collect(L.x_grid) ≈ collect(T.x_grid) "x grids differ — screens not co-located"
@assert collect(L.y_grid) ≈ collect(T.y_grid) "y grids differ — screens not co-located"
@assert size(L.fields_h) == size(T.fields_h) "fields_h shapes differ"

lpwa_id, thom_id = run_id(La, lpwa_toml), run_id(Ta, thom_toml)
@printf("comparing LPWA %s  vs  Thomson %s\n", lpwa_id, thom_id)
@printf("  a0 = %s,  harmonics = %s\n", La["laser"]["a0"], L.harmonics)

const complabels = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")

# Absolute (un-normalised) error between two complex (Nx,Ny) maps. Raw |a−b| keeps
# vortex cores / nodal lines in the OAM maps from blowing it up the way a relative
# error would. The scalar is the Frobenius L² of the complex difference.
abserr(a, b) = norm(a .- b)

xw, yw = collect(L.x_grid) ./ L.w₀, collect(L.y_grid) ./ L.w₀

for (k, n) in enumerate(L.harmonics)
    fig = Figure(size = (1080, 680))
    Label(fig[0, 1:3], @sprintf("|F̃_LPWA − F̃_Thomson| at %dω₁", n); fontsize = 18)
    errs = Float64[]
    for comp in 1:6
        a = L.fields_h[k, comp, :, :]
        b = T.fields_h[k, comp, :, :]
        d = abserr(a, b)
        push!(errs, d)
        @printf("h=%dω₁  %-3s   |Δ|₂ = %.4e   (‖ref‖₂ = %.4e)\n", n, complabels[comp], d, norm(b))
        r, c = fldmod1(comp, 3)
        ax = Axis(
            fig[r, c][1, 1]; title = complabels[comp], xlabel = "x/w₀", ylabel = "y/w₀",
            aspect = DataAspect()
        )
        hm = heatmap!(ax, xw, yw, abs.(a .- b); colormap = :inferno)
        Colorbar(fig[r, c][1, 2], hm)
    end
    out = joinpath(OUTDIR, @sprintf("compare_abs_h%d_%s-%s.png", n, first(lpwa_id, 8), first(thom_id, 8)))
    save(out, fig)
    println("saved → ", out)
    # Provenance: bind this comparison to BOTH source runs (depends_on = [LPWA, Thomson]); the
    # builder attaches it under each, with the per-component |Δ|₂ in the description. setup.harmonic
    # makes it ONE parametrized "comparison" chip with a harmonic selector.
    write_derived(
        OUTDIR; kind = "comparison", label = "LPWA vs Thomson |ΔF|",
        run_id = [lpwa_id, thom_id], plot = basename(out), setup = Dict("harmonic" => n),
        description = "|F̃_LPWA − F̃_Thomson|₂ at $(n)ω₁ (a0=$(La["laser"]["a0"])): " *
            join((@sprintf("%s=%.2e", complabels[c], errs[c]) for c in 1:6), ", "),
    )
    println("derived → comparison h$n  (parents $lpwa_id, $thom_id)")
end
