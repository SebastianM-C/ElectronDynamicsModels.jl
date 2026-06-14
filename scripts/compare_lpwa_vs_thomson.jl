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
using ElectronDynamicsModels: ring_pixels, phase_winding_fit
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

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
@printf("comparing LPWA %s  vs  numeric %s\n", lpwa_id, thom_id)
@printf("  a0 = %s,  harmonics = %s\n", La["laser"]["a0"], L.harmonics)

const complabels = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")

# Absolute (un-normalised) error between two complex (Nx,Ny) maps. Raw |a−b| keeps
# vortex cores / nodal lines in the OAM maps from blowing it up the way a relative
# error would. The scalar is the Frobenius L² of the complex difference.
abserr(a, b) = norm(a .- b)

xw, yw = collect(L.x_grid) ./ L.w₀, collect(L.y_grid) ./ L.w₀

for (k, n) in enumerate(L.harmonics)
    fig = Figure(size = (1080, 680))
    Label(fig[0, 1:3], @sprintf("|F̃_LPWA − F̃_numeric| at %dω₁", n); fontsize = 18)
    errs = Float64[]
    for comp in 1:6
        a = L.fields_h[k, comp, :, :]
        b = T.fields_h[k, comp, :, :]
        d = abserr(a, b)
        push!(errs, d)
        nrm = norm(b)
        relL2 = nrm == 0 ? 0.0 : d / nrm   # ‖Δ‖₂/‖F_num‖₂ — dimensionless gap (≈√2 for the transverse fundamental)
        @printf("h=%dω₁  %-3s   |Δ|₂ = %.4e   (‖ref‖₂ = %.4e,  ‖Δ‖/‖F‖ = %.3f)\n", n, complabels[comp], d, nrm, relL2)
        r, c = fldmod1(comp, 3)
        # Plot |Δ| as a fraction of the numeric field's own peak so the colorbar is dimensionless
        max_b = maximum(abs, b)
        scale = iszero(max_b) ? 1.0 : max_b
        ax = Axis(
            fig[r, c][1, 1]; title = @sprintf("%s   ‖Δ‖/‖F‖=%.2f", complabels[comp], relL2),
            xlabel = "x/w₀", ylabel = "y/w₀", aspect = DataAspect()
        )
        hm = heatmap!(ax, xw, yw, abs.(a .- b) ./ scale; colormap = :inferno)
        Colorbar(fig[r, c][1, 2], hm; label = "|Δ| / peak|F_num|")
    end
    out = joinpath(OUTDIR, @sprintf("compare_abs_h%d_%s-%s.png", n, first(lpwa_id, 8), first(thom_id, 8)))
    save(out, fig)
    println("saved → ", out)
    # Provenance: bind this comparison to BOTH source runs (depends_on = [LPWA, Thomson]); the
    # builder attaches it under each, with the per-component |Δ|₂ in the description. setup.harmonic
    # makes it ONE parametrized "comparison" chip with a harmonic selector.
    write_derived(
        OUTDIR; kind = "comparison", label = "LPWA vs numeric |ΔF|",
        run_id = [lpwa_id, thom_id], plot = basename(out), setup = Dict("harmonic" => n),
        description = "|F̃_LPWA − F̃_numeric|₂ at $(n)ω₁ (a0=$(La["laser"]["a0"])): " *
            join((@sprintf("%s=%.2e", complabels[c], errs[c]) for c in 1:6), ", "),
    )
    println("derived → comparison h$n  (parents $lpwa_id, $thom_id)")
end

# ── Eᶻ physical-field comparison (the Re part the heatmaps plot) ──
# (a) ratio Re(Eᶻ_L)/Re(Eᶻ_T) over the screen, masked at the Thomson node floor → NaN (white);
# (b) the physical Re(Eᶻ) overlaid along a few rays, LPWA vs Thomson. Both two-parent comparison
# derived. Eᶻ is component 3 (Eˣ Eʸ Eᶻ = 1 2 3).
EZ = 3
xg, yg = collect(L.x_grid), collect(L.y_grid)   # ascending screen axes (asserted == Thomson's)
ρmax = maximum(abs, xg)
ρs = range(0, ρmax; length = length(xg))
φovl = range(0, π / 2; length = 4)              # 0°,30°,60°,90° — distinct rays within Eᶻ's 120° (ℓ≈3) period

# bilinear sample of grid array M at (px, py); NaN outside the screen
function bilin(M, px, py)
    (px < xg[1] || px > xg[end] || py < yg[1] || py > yg[end]) && return NaN
    i = clamp(searchsortedlast(xg, px), 1, length(xg) - 1)
    j = clamp(searchsortedlast(yg, py), 1, length(yg) - 1)
    tx = (px - xg[i]) / (xg[i + 1] - xg[i])
    ty = (py - yg[j]) / (yg[j + 1] - yg[j])
    return (1 - tx) * (1 - ty) * M[i, j] + tx * (1 - ty) * M[i + 1, j] +
        (1 - tx) * ty * M[i, j + 1] + tx * ty * M[i + 1, j + 1]
end

for (k, n) in enumerate(L.harmonics)
    ReL = real.(L.fields_h[k, EZ, :, :])
    ReT = real.(T.fields_h[k, EZ, :, :])

    # (a) ratio map: white where |Re(Eᶻ_T)| is below the node floor (NaN) or the ratio ≈ 0
    floorT = 1.0e-3 * maximum(abs, ReT)
    ratio = map((l, t) -> abs(t) < floorT ? NaN : l / t, ReL, ReT)
    figr = Figure(size = (560, 520))
    axr = Axis(
        figr[1, 1]; aspect = DataAspect(), xlabel = "x/w₀", ylabel = "y/w₀",
        title = @sprintf("Re(Eᶻ) LPWA/numeric at %dω₁ (a0=%s)", n, La["laser"]["a0"])
    )
    hmr = heatmap!(
        axr, xw, yw, ratio;
        colormap = cgrad([:white, :steelblue, :firebrick]), colorrange = (0, 2), nan_color = :white
    )
    Colorbar(figr[1, 2], hmr; label = "Re(Eᶻ_LPWA)/Re(Eᶻ_num)")
    rout = joinpath(OUTDIR, @sprintf("compare_ez_ratio_h%d_%s-%s.png", n, first(lpwa_id, 8), first(thom_id, 8)))
    save(rout, figr)
    println("saved → ", rout)
    write_derived(
        OUTDIR; kind = "ez_ratio", label = "Eᶻ ratio LPWA/numeric",
        run_id = [lpwa_id, thom_id], plot = basename(rout), setup = Dict("harmonic" => n),
        description = "Re(Eᶻ_LPWA)/Re(Eᶻ_numeric) over the screen at $(n)ω₁ (a0=$(La["laser"]["a0"])); white = |Re(Eᶻ_numeric)| below the node floor."
    )

    # (b) overlay along rays: LPWA (solid, semi-transparent so the numeric dashed line shows
    # through where the two coincide) vs numeric (dashed), one colour per φ
    figo = Figure(size = (780, 480))
    axo = Axis(
        figo[1, 1]; xlabel = "ρ/w₀", ylabel = "Re(Eᶻ)",
        title = @sprintf("Re(Eᶻ) along rays at %dω₁ — LPWA solid, numeric dashed", n)
    )
    for (j, φ) in enumerate(φovl)
        rl = [bilin(ReL, ρ * cos(φ), ρ * sin(φ)) for ρ in ρs]
        rt = [bilin(ReT, ρ * cos(φ), ρ * sin(φ)) for ρ in ρs]
        lines!(
            axo, ρs ./ L.w₀, rl; color = Cycled(j), alpha = 0.5, linewidth = 3,
            label = @sprintf("φ=%.0f°", rad2deg(φ))
        )
        lines!(axo, ρs ./ L.w₀, rt; color = Cycled(j), linestyle = :dash, linewidth = 1.5)
    end
    axislegend(axo; labelsize = 9)
    oout = joinpath(OUTDIR, @sprintf("compare_ez_overlay_h%d_%s-%s.png", n, first(lpwa_id, 8), first(thom_id, 8)))
    save(oout, figo)
    println("saved → ", oout)
    write_derived(
        OUTDIR; kind = "ez_overlay", label = "Eᶻ along rays LPWA vs numeric",
        run_id = [lpwa_id, thom_id], plot = basename(oout), setup = Dict("harmonic" => n),
        description = "Re(Eᶻ) vs ρ along $(length(φovl)) rays ($(round(Int, rad2deg(first(φovl))))–$(round(Int, rad2deg(last(φovl))))°, within Eᶻ's 120° azimuthal period) at $(n)ω₁ (a0=$(La["laser"]["a0"])); LPWA solid, numeric dashed."
    )
    println("derived → ez_ratio + ez_overlay h$n  (parents $lpwa_id, $thom_id)")
end

# ── Transverse ∠F offset: is LPWA − numeric a constant phase b? ──
# For Eˣ, Eʸ on R/w₀ ∈ {4, 12}, sample the SAME ring pixels on both fields (shared azimuth order ⇒
# common unwrap reference), fit ∠F ≈ slope·φ + b, and report Δb = b_LPWA − b_numeric (nearest branch,
# in (−π,π]) per (component, radius). A Δb constant across the two radii ⇒ a radius-independent
# phase offset — the suspected origin of the constant √2 transverse gap (see lpwa-vs-numeric notes).
let
    offset_comps = (1, 2)                          # Eˣ, Eʸ
    offset_radii = L.w₀ .* (4, 12)
    tol = 0.5 * (xg[2] - xg[1])
    palette = (:dodgerblue, :crimson)
    ringcolor(i) = palette[(i - 1) % length(palette) + 1]
    for (k, n) in enumerate(L.harmonics)
        fig = Figure(size = (980, 460))
        Label(fig[0, 1:2], @sprintf("∠F unwrapped vs φ — LPWA solid, numeric dashed (%dω₁, a0=%s)", n, La["laser"]["a0"]); fontsize = 16)
        pp = Dict{String, Any}("R/w₀" => collect(offset_radii ./ L.w₀))
        for (ci, comp) in enumerate(offset_comps)
            ax = Axis(fig[1, ci]; xlabel = "azimuth φ", ylabel = "∠F (unwrapped)", title = complabels[comp])
            db = Float64[]; slL = Float64[]; slT = Float64[]
            for (ri, R) in enumerate(offset_radii)
                idxs, az = ring_pixels(L.x_grid, L.y_grid, R; tol)
                if isempty(idxs)
                    push!(db, NaN); push!(slL, NaN); push!(slT, NaN)
                    continue
                end
                aL = L.fields_h[k, comp, :, :][idxs]
                aT = T.fields_h[k, comp, :, :][idxs]
                fL = phase_winding_fit(az, angle.(aL); weights = abs.(aL))
                fT = phase_winding_fit(az, angle.(aT); weights = abs.(aT))
                raw = fL.intercept - fT.intercept
                shift = 2π * round(raw / 2π)        # align numeric to LPWA's 2π branch
                push!(db, raw - shift); push!(slL, fL.slope); push!(slT, fT.slope)
                col = ringcolor(ri)
                lines!(ax, az, fL.unwrapped; color = col, alpha = 0.6, linewidth = 2.5,
                    label = @sprintf("R/w₀=%d", round(Int, R / L.w₀)))
                lines!(ax, az, fT.unwrapped .+ shift; color = col, linestyle = :dash, linewidth = 1.5)
            end
            ci == 1 && axislegend(ax; labelsize = 9, position = :lt)
            pp["Δb/π $(complabels[comp])"] = round.(db ./ π; sigdigits = 3)
            pp["ℓ_LPWA $(complabels[comp])"] = round.(slL; sigdigits = 3)
            pp["ℓ_num $(complabels[comp])"] = round.(slT; sigdigits = 3)
        end
        out = joinpath(OUTDIR, @sprintf("compare_phase_offset_h%d_%s-%s.png", n, first(lpwa_id, 8), first(thom_id, 8)))
        save(out, fig)
        println("saved → ", out)
        write_derived(
            OUTDIR; kind = "phase_offset", label = "∠F offset LPWA vs numeric",
            run_id = [lpwa_id, thom_id], plot = basename(out), setup = Dict("harmonic" => n),
            plot_params = pp,
            description = "Δb = b_LPWA − b_numeric (nearest 2π branch, in (−π,π]) of the ∠F ≈ slope·φ + b " *
                "fit on Eˣ,Eʸ at R/w₀=4,12, $(n)ω₁ (a0=$(La["laser"]["a0"])). Constant Δb across R ⇒ a pure " *
                "radius-independent phase offset; a slope (ℓ) mismatch — see ℓ_LPWA vs ℓ_num in the " *
                "plot parameters — means the offset is φ-dependent, not a constant b.",
        )
        println("derived → phase_offset h$n  (parents $lpwa_id, $thom_id)")
    end
end
