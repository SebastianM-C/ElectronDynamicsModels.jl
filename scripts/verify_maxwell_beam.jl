# Verify the INCIDENT beam satisfies vacuum Maxwell — the analytic-laser sibling of
# verify_lorenz_gauge.jl (which checks the *radiated* 4-potential). The LG/Gauss lasers
# are closed-form paraxial-corrected fields, so two kinds of residual coexist:
#
#   • paraxial TRUNCATION — genuine to the model, scales down with (λ/w₀)ⁿ;
#   • convention BUGS — sign/term errors that do NOT scale away with w₀/λ.
#
# The two diagnostics here separate them:
#   1. residual maps  — |∇·E|, |∇·B|, |∇×E + ∂B/∂t|, |∇×B − (1/c²)∂E/∂t| (central
#      differences) over the transverse plane, one row per beam configuration
#      (k = ±ẑ × σ∓/linear). A bug lights up specific configurations/components.
#   2. scaling plot   — worst residual vs w₀/λ on log-log axes with a (λ/w₀)² guide.
#      Truncation follows the guide; a bug plateaus. The 2026-07-30 Bz parity error
#      (overall kz_sign on B[3] made every k=−ẑ beam non-Maxwellian, |∇·B| ~ 2e-3 of
#      the gradient scale vs ~1e-5 for +ẑ) is exactly the class this catches — it was
#      found by the γ=1 direct/inverse crosscheck and would sit flat in panel 2.
#
# Residuals are normalized by the configuration's max field-gradient magnitude, so
# every number is dimensionless and comparable across configs and beam sizes.
#
#   julia --project=scripts scripts/verify_maxwell_beam.jl
#
# ENV: EDM_OUTDIR (default .), EDM_MAXWELL_NGRID (map resolution/side, default 61),
#      EDM_MAXWELL_W0S (comma-sep w₀/λ list for the scaling panel, default 15,37.5,75,150),
#      EDM_ASSERT_DIVB (fail with nonzero exit if any config's |∇·B|/scale exceeds this;
#      "" = report only). EDM_DERIVED_FROM optionally links the dashboard node to a run.

using ElectronDynamicsModels
using ModelingToolkit
using StaticArrays
using LinearAlgebra
using Statistics
using Printf
using CairoMakie

const c = 137.03599908330932
const OUTDIR = get(ENV, "EDM_OUTDIR", ".")
const NGRID = parse(Int, get(ENV, "EDM_MAXWELL_NGRID", "61"))
const W0S = parse.(Float64, split(get(ENV, "EDM_MAXWELL_W0S", "15,37.5,75,150"), ","))
const ASSERT_DIVB = get(ENV, "EDM_ASSERT_DIVB", "")
mkpath(OUTDIR)

ω = 0.057
λ = 2π * c / ω

const CONFIGS = (
    (label = "k+ σ−", kdir = [0, 0, 1], pol = :circular_minus),
    (label = "k− σ−", kdir = [0, 0, -1], pol = :circular_minus),
    (label = "k− σ+", kdir = [0, 0, -1], pol = :circular_plus),
    (label = "k+ lin", kdir = [0, 0, 1], pol = :linear),
    (label = "k− lin", kdir = [0, 0, -1], pol = :linear),
)

function make_evaluator(kdir, pol, w₀)
    @named world = Worldline(:τ, :atomic)
    @named laser = LaguerreGaussLaser(; wavelength = λ, a0 = 0.3, beam_waist = w₀,
        radial_index = 2, azimuthal_index = -2, world, temporal_profile = :gaussian,
        temporal_width = 150 / ω, focus_position = 0.0, polarization = pol,
        initial_phase = -π / 2, k_direction = kdir)
    return FieldEvaluator(laser)
end

# Central-difference Maxwell residuals at one spacetime point p = [t, x, y, z]
# (FieldEvaluator's first coordinate is TIME, in the solver's time units — the envelope
# argument is (t − t₀) − kz_sign·z/c). Spatial step h (length), temporal step ht (time).
# All residuals are expressed in E-gradient units (Ampère's is rescaled by c) and
# normalized by the local max gradient, so every number is dimensionless.
function residuals(fe, p::SVector{4, Float64}, h::Float64, ht::Float64)
    ev(dp) = fe(p .+ dp)
    step(i) = SVector{4}(ntuple(j -> j == i ? (i == 1 ? ht : h) : 0.0, 4))
    rp = ntuple(i -> ev(step(i)), 4)
    rm = ntuple(i -> ev(-step(i)), 4)
    dE = ntuple(i -> (rp[i].E .- rm[i].E) ./ (i == 1 ? 2ht : 2h), 4)   # ∂E/∂(t,x,y,z)
    dB = ntuple(i -> (rp[i].B .- rm[i].B) ./ (i == 1 ? 2ht : 2h), 4)
    divE = dE[2][1] + dE[3][2] + dE[4][3]
    divB = dB[2][1] + dB[3][2] + dB[4][3]
    curlE = SVector(dE[3][3] - dE[4][2], dE[4][1] - dE[2][3], dE[2][2] - dE[3][1])
    curlB = SVector(dB[3][3] - dB[4][2], dB[4][1] - dB[2][3], dB[2][2] - dB[3][1])
    faraday = curlE .+ dB[1]                     # ∇×E + ∂B/∂t
    ampere = c .* (curlB .- dE[1] ./ c^2)        # c·(∇×B − (1/c²)∂E/∂t) → E-gradient units
    scale = max(maximum(x -> maximum(abs, x), dE[2:4]),
        c * maximum(x -> maximum(abs, x), dB[2:4]))
    return (; divE = abs(divE), divB = abs(divB) * c,   # c·|∇·B| → E-gradient units
        faraday = norm(faraday), ampere = norm(ampere), scale)
end

const RES_KEYS = (:divE, :divB, :faraday, :ampere)
const RES_LABELS = ("|∇·E|", "|∇·B|", "|∇×E + ∂B/∂t|", "|∇×B − ∂E/∂t/c²|")

# ── 1. residual maps on the transverse plane (z = 0.2λ, mid-pulse) ──
w₀_map = 75λ
h = λ / 50                     # spatial stencil
const T = 2π / ω
ht = T / 64                    # temporal stencil (FieldEvaluator's first coordinate is TIME)
t₀p = 5T                       # pulse-center time (LaguerreGaussLaser n_cycles = 5 default)
xs = range(-1.5w₀_map, 1.5w₀_map; length = NGRID)
fig = Figure(size = (380 * 4 + 140, 300 * length(CONFIGS) + 90))
worst_map = Dict{String, Dict{Symbol, Float64}}()
for (ic, cfg) in enumerate(CONFIGS)
    fe = make_evaluator(cfg.kdir, cfg.pol, w₀_map)
    maps = Dict(k => Matrix{Float64}(undef, NGRID, NGRID) for k in RES_KEYS)
    smax = 0.0
    for (i, x) in enumerate(xs), (j, y) in enumerate(xs)
        r = residuals(fe, SVector(t₀p + 0.1T, x, y, 0.2λ), h, ht)
        for k in RES_KEYS
            maps[k][i, j] = getproperty(r, k)
        end
        smax = max(smax, r.scale)
    end
    worst_map[cfg.label] = Dict(k => maximum(maps[k]) / smax for k in RES_KEYS)
    for (kk, k) in enumerate(RES_KEYS)
        ax = Axis(fig[ic, kk]; title = ic == 1 ? RES_LABELS[kk] : "",
            ylabel = kk == 1 ? cfg.label : "",
            xticklabelsvisible = ic == length(CONFIGS), yticklabelsvisible = kk == 1)
        hm = heatmap!(ax, xs ./ w₀_map, xs ./ w₀_map, log10.(maps[k] ./ smax .+ 1e-16);
            colormap = :inferno, colorrange = (-8, -1))
        ic == length(CONFIGS) && kk == length(RES_KEYS) &&
            Colorbar(fig[1:length(CONFIGS), 5], hm; label = "log₁₀ residual / max|∇F|")
    end
end
Label(fig[0, 1:4], "Incident-beam Maxwell residuals — LG p=2 m=−2, w₀ = 75λ, a0 = 0.3 (normalized)";
    fontsize = 18, font = :bold)
mapfile = joinpath(OUTDIR, "maxwell_beam_maps.png")
save(mapfile, fig)
println("saved → $mapfile")

# ── 2. scaling with w₀/λ: truncation falls as (λ/w₀)², a convention bug stays flat ──
scal = Dict(cfg.label => Dict(k => Float64[] for k in RES_KEYS) for cfg in CONFIGS)
for w0f in W0S
    w₀ = w0f * λ
    probes = (SVector(t₀p + 0.1T, 0.7w₀, 0.3w₀, 0.2λ), SVector(t₀p + 0.4T, 1.2w₀, -0.5w₀, -1.5λ),
        SVector(t₀p - 0.7T, 0.4w₀, 0.9w₀, 2.2λ))
    for cfg in CONFIGS
        fe = make_evaluator(cfg.kdir, cfg.pol, w₀)
        rs = [residuals(fe, p, h, ht) for p in probes]
        smax = maximum(r -> r.scale, rs)
        for k in RES_KEYS
            push!(scal[cfg.label][k], maximum(r -> getproperty(r, k), rs) / smax)
        end
    end
end
fig2 = Figure(size = (420 * 4, 400))
for (kk, k) in enumerate(RES_KEYS)
    ax = Axis(fig2[1, kk]; title = RES_LABELS[kk], xlabel = "w₀ / λ",
        ylabel = kk == 1 ? "residual / max|∇F|" : "", xscale = log10, yscale = log10)
    for cfg in CONFIGS
        scatterlines!(ax, W0S, max.(scal[cfg.label][k], 1e-16); label = cfg.label)
    end
    guide = (W0S[1] ./ W0S) .^ 2 .* maximum(scal[CONFIGS[1].label][k][1])
    lines!(ax, W0S, guide; color = :gray, linestyle = :dash, label = "(λ/w₀)²")
    kk == length(RES_KEYS) && axislegend(ax; position = :lb, labelsize = 10)
end
Label(fig2[0, 1:4], "Maxwell residual scaling — paraxial truncation follows the guide; a convention bug plateaus";
    fontsize = 16, font = :bold)
scalfile = joinpath(OUTDIR, "maxwell_beam_scaling.png")
save(scalfile, fig2)
println("saved → $scalfile")

# ── report + optional assertion ──
println("\nworst residual / scale on the w₀ = 75λ map:")
@printf("%-8s %12s %12s %12s %12s\n", "config", RES_KEYS...)
for cfg in CONFIGS
    w = worst_map[cfg.label]
    @printf("%-8s %12.2e %12.2e %12.2e %12.2e\n", cfg.label, (w[k] for k in RES_KEYS)...)
end
if !isempty(ASSERT_DIVB)
    thr = parse(Float64, ASSERT_DIVB)
    bad = [cfg.label for cfg in CONFIGS if worst_map[cfg.label][:divB] > thr]
    isempty(bad) || error("∇·B residual above $(thr) for: $(join(bad, ", "))")
    println("assert: all configs satisfy |∇·B|/scale ≤ $thr")
end

# ── dashboard metadata (standalone verification node, deterministic id) ──
using UUIDs
using RunManifests
let V = string(uuid5(UUID("0000000e-d42a-4000-8000-000000000000"),
            "maxwell_beam_$(NGRID)_$(join(W0S, '-'))")),
        parent = get(ENV, "EDM_DERIVED_FROM", "")

    write_run_manifest(OUTDIR; run_id = V, script = basename(PROGRAM_FILE),
        derived_from = isempty(parent) ? nothing : parent,
        config = Dict("a0" => 0.3, "initial_phase" => -π / 2, "N" => 0, "N_samples" => 0,
            "samples_per_period" => 0, "n_substeps" => 0),
        laser = Dict("wavelength" => λ, "a0" => 0.3, "w0" => w₀_map, "p" => 2, "m" => -2,
            "pol" => "all", "profile" => "gaussian", "temporal_width" => 150 / ω,
            "focus_position" => 0.0, "phi0" => -π / 2),
        setup = Dict("ngrid" => NGRID, "w0_scan" => W0S))
    write_derived(OUTDIR; kind = "maxwell_maps", label = "Maxwell residual maps",
        run_id = V, plot = basename(mapfile), source = basename(mapfile),
        plot_params = Dict("worst divB $(cfg.label)" => round(worst_map[cfg.label][:divB]; sigdigits = 3)
            for cfg in CONFIGS),
        description = "Normalized |∇·E|, |∇·B|, Faraday and Ampère residuals of the incident " *
            "LG beam over the transverse plane, one row per (k̂, polarization) configuration.")
    write_derived(OUTDIR; kind = "maxwell_scaling", label = "Maxwell residual scaling",
        run_id = V, plot = basename(scalfile), source = basename(scalfile),
        description = "Worst normalized Maxwell residuals vs w₀/λ: paraxial truncation follows " *
            "the (λ/w₀)² guide; convention bugs plateau (how the 2026-07-30 reversed-beam Bz " *
            "parity error would present).")
    println("dashboard metadata → verification node $V", isempty(parent) ? "" : " (derived_from $parent)")
end
