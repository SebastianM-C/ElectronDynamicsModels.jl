# Backfill the emission-time profile (emissiontime_<uuid>.jls + chip + emission_time sidecar) for runs that
# predate its run-time emission (#150): rebuild each run's trajectory ensemble from run_<uuid>.toml, sample it
# through the same kind of interpolants the solver integrates, and write the products under the ORIGINAL uuid.
# Deterministic from the manifest (the cube is never touched):
#   • thomson_scattering.jl runs: the forward ODE solve (same MTK model, [config] tolerances, uniform saveat,
#     sunflower disc at z = 0, [setup] solve span) → trajectory_interpolants (4-acceleration spline a_itp);
#   • lpwa.jl runs ([config].trajectory_source = "lpwa_analytic"): lpwa.jl's analytic orbit on its knot grid.
# The .reduced marker is enriched with the new cache so it publishes; the sidecar carries a provenance note.
#
#   julia +1.12 --project=scripts -t auto scripts/emission_time_backfill.jl <campaign_dir> [uuid8 ...]
#
# Idempotent: runs whose emissiontime_<uuid>.jls exists are skipped. EDM_ELECTRON_BATCH (default 1000)
# bounds the number of trajectories resident at once for the ODE path.
using TOML, Dates, Serialization, Printf, LinearAlgebra, StaticArrays
using CairoMakie, RunManifests
using ElectronDynamicsModels, ModelingToolkit, OrdinaryDiffEqVerner, SciMLBase, SymbolicIndexingInterface
using DataInterpolations
using HypergeometricFunctions: pochhammer, _₁F₁
using SpecialFunctions
include(joinpath(@__DIR__, "trajectory_products.jl"))   # emission_time_acc / emission_time / write_emission_time

const c = 137.03599908330932
const BATCH = parse(Int, get(ENV, "EDM_ELECTRON_BATCH", "1000"))
const COMMIT = try readchomp(`git -C $(pkgdir(ElectronDynamicsModels)) rev-parse --short HEAD`) catch; "unknown" end


# Same marker enrichment as gammatau_backfill.jl: replace-or-append the entry byte-exact, atomic tmp + mv.
function enrich_marker!(dir, uuid, files)
    path = joinpath(dir, "$(uuid).reduced")
    m = (isfile(path) && filesize(path) > 0) ? TOML.parsefile(path) : Dict{String, Any}(
        "run_id" => string(uuid), "reduced_at" => string(now()), "host" => gethostname(), "reduction" => Any[])
    red = get!(m, "reduction", Any[])
    for f in files
        bn = basename(f)
        entry = Dict{String, Any}("file" => bn, "bytes" => filesize(joinpath(dir, bn)))
        i = findfirst(e -> get(e, "file", "") == bn, red)
        i === nothing ? push!(red, entry) : (red[i] = entry)
    end
    tmp = path * ".tmp"
    open(io -> TOML.print(io, m; sorted = true), tmp, "w")
    mv(tmp, path; force = true)
    return path
end

# Forward ODE path (thomson_scattering.jl): the ensemble in batches, each folded into the accumulator.
function numeric_emission!(acc, m)
    cfg, las, st = m["config"], m["laser"], m["setup"]
    λ, a₀, w₀ = las["wavelength"], las["a0"], las["w0"]
    ω = 2π * c / λ
    τi_solve, τf_solve = st["τi_solve"], st["τf_solve"]
    @named world = Worldline(:τ, :atomic)
    @named laser = LaguerreGaussLaser(; wavelength = λ, a0 = a₀, beam_waist = w₀, radial_index = Int(las["p"]),
        azimuthal_index = Int(las["m"]), world, temporal_profile = Symbol(las["profile"]),
        temporal_width = las["temporal_width"], focus_position = las["focus_position"],
        polarization = Symbol(las["pol"]), initial_phase = cfg["initial_phase"])
    @named elec = ClassicalElectron(; laser)
    sys = mtkcompile(elec)
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(sys, [sys.x => [τi_solve * c, 0.0, 0.0, 0.0], sys.u => [c, 0.0, 0.0, 0.0]],
        (τi_solve, τf_solve), u0_constructor = SVector{8}, fully_determined = true)
    setx = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
    R₀ = st["Rmax"] .* sunflower(cfg["N"], 2)
    saveat = collect(τi_solve:((2π / ω) / parse(Float64, string(cfg["interp_saveat"]))):τf_solve)
    for lo in 1:BATCH:cfg["N"]
        rng = lo:min(lo + BATCH - 1, cfg["N"])
        pf(p, ctx) = (r = R₀[rng[ctx.sim_id]]; (u0, pp) = setx(p, SVector{8}(τi_solve * c, r[1], r[2], 0.0, c, 0.0, 0.0, 0.0)); remake(p; u0, p = pp))
        sol = solve(EnsembleProblem(prob; prob_func = pf, safetycopy = false), Vern9(), EnsembleThreads();
            trajectories = length(rng), reltol = cfg["reltol"], abstol = cfg["abstol"], saveat)
        fold_emission!(acc, emission_time(ElectronDynamicsModels.trajectory_interpolants(sol), acc.τs, c, 2π / ω))
    end
    return 2π / ω
end

# Analytic LPWA path (lpwa.jl, verbatim physics and knot grid).
function lpwa_emission!(acc, m)
    cfg, st = m["config"], m["setup"]
    a₀, ϕ₀ = cfg["a0"], cfg["initial_phase"]
    qme, p, mm, ω, s = -1.0, 2, -2, 0.057, 150
    λ = c * 2π / ω; w₀ = 75λ; k = ω / c; τ = 150 / ω
    A₀ = a₀ * c / qme * sqrt(pochhammer(p + 1, abs(mm))) / √2
    A(ρ) = A₀ * (√2 * ρ / w₀)^abs(mm) * _₁F₁(-p, abs(mm) + 1, 2 * (ρ / w₀)^2) * exp(-(ρ / w₀)^2)
    function trajectory(τₚ, ℜ₀)
        x₀, y₀, z₀ = ℜ₀; ρ₀ = norm(ℜ₀); φ = -mm * atan(y₀, x₀) + ϕ₀ + π / 2
        u⁰ = c; χ = k * u⁰ * τₚ; a = A(ρ₀) * qme / c
        Δx = inv(k) * a * s * exp(-(χ / s)^2) * real(im * cis(φ + χ) * dawson(s / 2 + im * χ / s))
        ẋ = -u⁰ * a * exp(-(χ / s)^2) * cos(φ + χ)
        Δy = inv(k) * a * s * exp(-(χ / s)^2) * real(cis(φ + χ) * dawson(s / 2 + im * χ / s))
        ẏ = -u⁰ * a * exp(-(χ / s)^2) * sin(φ + χ)
        Δz = inv(2k) * a^2 * s / 2 * sqrt(π / 2) * (1 + erf(sqrt(2) * χ / s))
        ż = u⁰ / 2 * a^2 * exp(-2 * (χ / s)^2)
        return SVector{8}(c * τₚ + Δz, x₀ + Δx, y₀ + Δy, z₀ + Δz, c + ż, ẋ, ẏ, ż)
    end
    τi_solve, τf_solve = st["τi_solve"], st["τf_solve"]
    Nτ = round(Int, 10_000 * (τf_solve - τi_solve) / (16τ))
    τs = collect(range(τi_solve, τf_solve, length = Nτ))
    x_idxs, u_idxs = SA[1, 2, 3, 4], SA[5, 6, 7, 8]
    xi = [[r..., 0.0] for r in (st["Rmax"] .* sunflower(cfg["N"], 2))]
    for lo in 1:BATCH:cfg["N"]
        rng = lo:min(lo + BATCH - 1, cfg["N"])
        trajs = Vector{Any}(undef, length(rng))
        Threads.@threads for j in eachindex(rng)
            us = [trajectory(τₚ, xi[rng[j]]) for τₚ in τs]
            itp = CubicSpline(us, τs; extrapolation = ExtrapolationType.Extension)
            a_itp = CubicSpline([DataInterpolations.derivative(itp, τₚ)[u_idxs] for τₚ in τs], τs;
                extrapolation = ExtrapolationType.Extension)
            trajs[j] = ElectronDynamicsModels.TrajectoryInterpolant(itp, a_itp, x_idxs, u_idxs, 1.0)
        end
        fold_emission!(acc, emission_time(identity.(trajs), acc.τs, c, 2π / ω))
    end
    return 2π / ω
end

function backfill_run(dir, mfile)
    m = TOML.parsefile(joinpath(dir, mfile))
    uuid = m["provenance"]["run_id"]
    out = joinpath(dir, "emissiontime_$(uuid).jls")
    isfile(out) && (println("skip $(first(uuid, 8)) — emissiontime_ exists"); return)
    isfile(joinpath(dir, "$(uuid).reduced")) || (println("skip $(first(uuid, 8)) — not reduced"); return)
    cfg, st = m["config"], m["setup"]
    lpwa = get(cfg, "trajectory_source", "") == "lpwa_analytic"
    ω = lpwa ? 0.057 : 2π * c / m["laser"]["wavelength"]
    acc = emission_time_acc(st["τi_solve"], st["τf_solve"], (2π / ω) / 16)
    t0 = time()
    T = lpwa ? lpwa_emission!(acc, m) : numeric_emission!(acc, m)
    note = "backfill: re-solved from the manifest at $(COMMIT) on $(Dates.today()) ($(lpwa ? "analytic LPWA orbit" : "forward ODE ensemble"))"
    write_emission_time(dir, uuid, acc; T, window_periods = cfg["N_samples"] / cfg["samples_per_period"], note)
    enrich_marker!(dir, uuid, [out])
    @printf "%s a₀ = %-6g %s  %.0f s\n" first(uuid, 8) cfg["a0"] (lpwa ? "lpwa" : "numeric") time() - t0
    flush(stdout)
end

function main_backfill(dir, only)
    for f in sort(filter(f -> startswith(f, "run_") && endswith(f, ".toml"), readdir(dir)))
        isempty(only) || any(u -> startswith(f[5:end], u), only) || continue
        backfill_run(dir, f)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_backfill(isempty(ARGS) ? "." : ARGS[1], ARGS[2:end])
end
