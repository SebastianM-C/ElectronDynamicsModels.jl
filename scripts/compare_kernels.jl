# Numeric A/B of two retarded-time GPU kernels (GPUKernelNewton vs GPUKernelRK4) run on
# IDENTICAL physics — same laser, screen, electrons, tolerances; the ONLY thing that differs
# is how the retarded time is solved on the GPU. Any disagreement is therefore purely
# numeric: the maps rendered here characterise each kernel's numerical floor, not physics.
# Per harmonic it renders |Δ| difference maps for B and E (total field and, when both runs
# saved the split, far field alone), normalized two ways — raw |Δ| and |Δ|/rms(ref) — so the
# SPATIAL STRUCTURE of the disagreement (smooth vs speckly, ring-following vs uniform) is
# visible. B is the featured field: E is known to carry the DC-shelf spectral-leakage floor
# (see the smalla0-floor notes), so its large-n harmonics don't resolve the kernel gap.
#
#   julia +release --project=scripts scripts/compare_kernels.jl RUN_A.toml RUN_REF.toml [RUN_ALT.toml]
#
# RUN_A = the kernel under test (Newton), RUN_REF = the reference kernel (RK4, the
# denominator of every relative number). The optional RUN_ALT is a same-kernel sibling of
# RUN_A differing only in iteration count (e.g. newton_iters=1 vs 2); its gap vs RUN_A is
# reported in the sidecars as the iteration-convergence floor.

using TOML, Serialization, LinearAlgebra, Statistics, Printf
using RunManifests: check_schema_version, write_derived, write_comparison
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

const newt_toml, ref_toml = ARGS[1], ARGS[2]
const alt_toml = length(ARGS) >= 3 ? ARGS[3] : nothing
const OUTDIR = get(ENV, "EDM_OUTDIR", ".")

Na = TOML.parsefile(newt_toml)
Ra = TOML.parsefile(ref_toml)
Aa = alt_toml === nothing ? nothing : TOML.parsefile(alt_toml)

check_schema_version(Na; source = "kernel-under-test manifest")
check_schema_version(Ra; source = "reference-kernel manifest")
Aa === nothing || check_schema_version(Aa; source = "alt-iterations manifest")

run_id(m, path) = get(
    get(m, "provenance", Dict()), "run_id",
    replace(basename(path), r"^run_" => "", r"\.toml$" => "")
)
script_of(m) = basename(get(get(m, "provenance", Dict()), "script", ""))
kernel_of(m) = string(get(m["config"], "accumulation_alg", "GPUKernelRK4"))

# Resolve the reduced hmaps .jls relative to the manifest: prefer [outputs].harmonic_maps,
# else the conventional hmaps_<run_id>.jls beside the toml.
function resolve_hmaps(m, path)
    dir = dirname(abspath(path))
    out = get(m, "outputs", Dict())
    haskey(out, "harmonic_maps") && return joinpath(dir, out["harmonic_maps"])
    cand = joinpath(dir, "hmaps_$(run_id(m, path)).jls")
    isfile(cand) && return cand
    error("$(basename(path)): no [outputs].harmonic_maps and no $(basename(cand)) beside it")
end

# Identical physics is the whole premise — refuse any pair that differs in anything except
# the kernel knobs. Unlike the LPWA comparison there is no convention gap to tolerate:
# both runs come from the same script, so pol/phi0/everything must match EXACTLY.
function check_compatible(a, b; expect_kernel_diff::Bool)
    laser_params = ("wavelength", "w0", "p", "m", "pol", "profile", "a0", "phi0",
        "temporal_width", "focus_position")
    config_params = ("samples_per_period", "Nx", "Ny", "N", "N_samples", "initial_phase",
        "mode", "observable", "reltol", "interp_saveat", "sync_per_electron")
    setup_params = ("τi", "τf", "Rmax", "Z")
    for (sec, keys) in (("laser", laser_params), ("config", config_params), ("setup", setup_params))
        for k in keys
            (haskey(a[sec], k) || haskey(b[sec], k)) || continue   # absent on both: older schema
            a[sec][k] == b[sec][k] || error("[$sec].$k differs: $(a[sec][k]) vs $(b[sec][k])")
        end
    end
    if expect_kernel_diff
        kernel_of(a) != kernel_of(b) ||
            @warn "both runs use the same retarded-time kernel — this compares iteration counts, not kernels" kernel = kernel_of(a)
    else
        kernel_of(a) == kernel_of(b) ||
            error("alt-iterations run uses a different kernel: $(kernel_of(a)) vs $(kernel_of(b))")
    end
    return nothing
end

check_compatible(Na, Ra; expect_kernel_diff = true)
Aa === nothing || check_compatible(Aa, Na; expect_kernel_diff = false)

N = deserialize(resolve_hmaps(Na, newt_toml))   # (; fields_h, fields_far_h, harmonics, ffund, x_grid, y_grid, w₀, …)
R = deserialize(resolve_hmaps(Ra, ref_toml))
A = Aa === nothing ? nothing : deserialize(resolve_hmaps(Aa, alt_toml))

@assert collect(N.x_grid) ≈ collect(R.x_grid) "x grids differ — screens not co-located"
@assert collect(N.y_grid) ≈ collect(R.y_grid) "y grids differ — screens not co-located"
@assert size(N.fields_h)[2:end] == size(R.fields_h)[2:end] "fields_h non-harmonic dims differ"

common_harmonics = A === nothing ? intersect(N.harmonics, R.harmonics) :
    intersect(N.harmonics, R.harmonics, A.harmonics)
isempty(common_harmonics) && error("no shared harmonics: $(N.harmonics) vs $(R.harmonics)")
k_of(H, n) = findfirst(==(n), H.harmonics)

newt_id, ref_id = run_id(Na, newt_toml), run_id(Ra, ref_toml)
const a0 = Na["laser"]["a0"]
@printf("comparing %s %s  vs  %s %s (reference)\n", kernel_of(Na), newt_id, kernel_of(Ra), ref_id)
Aa === nothing || @printf(
    "  iteration floor vs %s (newton_iters=%s)\n", run_id(Aa, alt_toml), Aa["config"]["newton_iters"]
)
@printf("  a0 = %s,  harmonics = %s\n", a0, Tuple(common_harmonics))

const complabels = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")

xw, yw = collect(N.x_grid) ./ N.w₀, collect(N.y_grid) ./ N.w₀

# Per-harmonic |Δ| maps of one field's 3 components between the two kernels. Top row: raw
# |Δ| (each panel's own colorbar — the fields span decades between components). Bottom row:
# |Δ|/rms(ref component) — one dimensionless texture scale, so the structure of the floor is
# comparable across components/harmonics. `Amaps` (optional) only feeds the iteration-floor
# scalar in the sidecar. One parametrized chip (harmonic selector) bound to BOTH parents.
function diff_maps(Nmaps, Rmaps, Amaps; comps, field, kind, label, file_tag, variant)
    for n in common_harmonics
        kN, kR = k_of(N, n), k_of(R, n)
        gN, gR = Nmaps[kN, comps, :, :], Rmaps[kR, comps, :, :]
        rel_group = norm(gN .- gR) / norm(gR)
        max_ratio = maximum(abs, gN .- gR) / maximum(abs, gR)
        # |Δ| vs |ref| pixel correlation: ≈1 → the disagreement rides ON the physical rings
        # (amplitude-proportional); ≈0 → a structureless (speckle/floor) pattern.
        ring_corr = cor(vec(abs.(gN .- gR)), vec(abs.(gR)))
        it_floor = Amaps === nothing ? nothing :
            norm(Amaps[k_of(A, n), comps, :, :] .- gN) / norm(gN)

        fig = Figure(size = (1150, 680))
        Label(
            fig[0, 1:3],
            @sprintf("|%s̃_Newton − %s̃_RK4| at %dω₁ (a0=%s, %s)", field, field, n, a0, variant);
            fontsize = 18
        )
        rels = Float64[]
        for (j, comp) in enumerate(comps)
            a = Nmaps[kN, comp, :, :]
            b = Rmaps[kR, comp, :, :]
            d = abs.(a .- b)
            nrm = norm(b)
            rel = nrm == 0 ? 0.0 : norm(a .- b) / nrm
            push!(rels, rel)
            rms = nrm / sqrt(length(b))
            ax1 = Axis(
                fig[1, j][1, 1]; title = @sprintf("%s   ‖Δ‖/‖ref‖=%.2e", complabels[comp], rel),
                xlabel = "x/w₀", ylabel = "y/w₀", aspect = DataAspect()
            )
            hm1 = heatmap!(ax1, xw, yw, d; colormap = :inferno)
            Colorbar(fig[1, j][1, 2], hm1; label = "|Δ|")
            ax2 = Axis(fig[2, j][1, 1]; xlabel = "x/w₀", ylabel = "y/w₀", aspect = DataAspect())
            hm2 = heatmap!(ax2, xw, yw, d ./ rms; colormap = :inferno)
            Colorbar(fig[2, j][1, 2], hm2; label = "|Δ| / rms(ref)")
        end
        out = joinpath(OUTDIR, @sprintf("compare_kernel_%s_h%d_%s-%s.png", file_tag, n, first(newt_id, 8), first(ref_id, 8)))
        save(out, fig)
        println("saved → ", out)
        @printf(
            "h=%dω₁  %s %-11s  relL2 = %.3e   max|Δ|/max|ref| = %.3e   corr(|Δ|,|ref|) = %+.2f%s\n",
            n, field, "($variant)", rel_group, max_ratio, ring_corr,
            it_floor === nothing ? "" : @sprintf("   it-floor = %.3e", it_floor)
        )
        pp = Dict{String, Any}(
            "relL2 ‖Δ‖/‖ref‖" => round(rel_group; sigdigits = 3),
            "relL2 per comp" => round.(rels; sigdigits = 3),
            "max|Δ|/max|ref|" => round(max_ratio; sigdigits = 3),
            "corr(|Δ|,|ref|)" => round(ring_corr; sigdigits = 3),
        )
        it_floor === nothing || (pp["relL2 it-pair (Newton $(Aa["config"]["newton_iters"]) vs $(Na["config"]["newton_iters"]) iters)"] =
            round(it_floor; sigdigits = 3))
        trust = comps == 4:6 ?
            "$field is the trusted field for floor comparisons (E carries the DC-shelf spectral-leakage floor)." :
            "Caveat: E carries the known DC-shelf spectral-leakage floor, so its high harmonics saturate on leakage, not on the kernel gap."
        write_derived(
            OUTDIR; kind, label,
            run_id = [newt_id, ref_id], plot = basename(out), setup = Dict("harmonic" => n),
            plot_params = pp,
            description = "|$(field)̃_Newton − $(field)̃_RK4| at $(n)ω₁ (a0=$a0, $variant): identical physics, " *
                "only the retarded-time GPU kernel differs, so this map IS the kernels' numeric disagreement. " *
                "Top row raw |Δ|; bottom row |Δ|/rms(ref component) (one dimensionless texture scale). " *
                @sprintf("rel-L2 = %.2e, max|Δ|/max|ref| = %.2e; corr(|Δ|,|ref|) = %.2f ", rel_group, max_ratio, ring_corr) *
                "(≈1 → the gap rides on the physical rings, ≈0 → structureless floor). " *
                (it_floor === nothing ? "" :
                @sprintf("Newton iteration-convergence floor (it%s vs it%s): rel-L2 = %.2e. ",
                    Aa["config"]["newton_iters"], Na["config"]["newton_iters"], it_floor)) * trust,
        )
        println("derived → $kind h$n  (parents $newt_id, $ref_id)")
    end
    return nothing
end

# B first — the featured, trusted field — then E; total field, then the far-field split.
diff_maps(N.fields_h, R.fields_h, A === nothing ? nothing : A.fields_h;
    comps = 4:6, field = "B", kind = "kernel_diff_B", label = "Newton vs RK4 |ΔB|",
    file_tag = "dB", variant = "total field")
diff_maps(N.fields_h, R.fields_h, A === nothing ? nothing : A.fields_h;
    comps = 1:3, field = "E", kind = "kernel_diff_E", label = "Newton vs RK4 |ΔE|",
    file_tag = "dE", variant = "total field")

Nfar = hasproperty(N, :fields_far_h) ? N.fields_far_h : nothing
Rfar = hasproperty(R, :fields_far_h) ? R.fields_far_h : nothing
Afar = (A !== nothing && hasproperty(A, :fields_far_h)) ? A.fields_far_h : nothing
if Nfar !== nothing && Rfar !== nothing
    @assert size(Nfar)[2:end] == size(Rfar)[2:end] "far-field non-harmonic dims differ"
    diff_maps(Nfar, Rfar, Afar;
        comps = 4:6, field = "B", kind = "kernel_diff_B_far", label = "Newton vs RK4 |ΔB| (far field)",
        file_tag = "dB_far", variant = "far field")
    diff_maps(Nfar, Rfar, Afar;
        comps = 1:3, field = "E", kind = "kernel_diff_E_far", label = "Newton vs RK4 |ΔE| (far field)",
        file_tag = "dE_far", variant = "far field")
else
    println("far-field maps absent on one/both runs — skipping far comparison")
end

# ── Comparison declaration: the first-class A-vs-B relationship the dashboard reads. ──
# Both sides live in ONE campaign dir (the kernel is a sweep axis there, not a dir split),
# so each side carries a `where` filter of canonical-param constraints selecting its runs —
# the builder pairs the two filtered sides cell-by-cell along the shared a0 axis. Idempotent:
# every a0-pair re-emits the SAME declaration under one fixed filename.
let
    function side_spec(m, path)
        alg = kernel_of(m)
        w = Dict{String, Any}("accumulation_alg" => alg)
        lbl = alg
        if alg == "GPUKernelNewton"
            w["newton_iters"] = m["config"]["newton_iters"]
            lbl = "Newton ($(m["config"]["newton_iters"]) iters)"
        elseif alg == "GPUKernelRK4"
            w["n_substeps"] = m["config"]["n_substeps"]
            lbl = "RK4 ($(m["config"]["n_substeps"]) substep)"
        end
        return Dict{String, Any}(
            "label" => lbl, "dir" => basename(dirname(abspath(path))),
            "script" => script_of(m), "where" => w,
        )
    end
    sides = [side_spec(Na, newt_toml), side_spec(Ra, ref_toml)]
    out = write_comparison(
        OUTDIR; label = "kernel numerics: Newton vs RK4", differs = "retarded-time kernel",
        along = "a0", sides,
        filename = "comparison_$(sides[1]["dir"])_kernels.toml",
    )
    println("comparison → ", basename(out), "  (", sides[1]["label"], "  vs  ", sides[2]["label"], ")")
end
