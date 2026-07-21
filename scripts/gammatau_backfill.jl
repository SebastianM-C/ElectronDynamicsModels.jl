# Backfill trajectory-side products for EXISTING inverse-Thomson runs that predate the
# run-time gammatau/ic emission (PR #51): re-solve each run's trajectory ensemble from its
# run_<uuid>.toml on CPU — deterministic from [config]/[laser]/[setup] (same solver, same
# tolerances, same uniform saveat; the field cube is never touched) — then emit, under the
# ORIGINAL run uuid:
#   • gammatau_<uuid>.jls   γ(τ)/γ₀ trace through the production-spec trajectory splines
#   • ic_<uuid>.jls + chip  as-run initial conditions (write_ic_products)
#   • the classical|LL γ(τ)/γ₀ comparison chips (ll_system_chips.gamma_drain_product)
# and enrich each run's .reduced marker with the new caches (atomic tmp+mv; these runs'
# reduce already finished, so appending to the final marker is safe), so stage_products
# ships them to the storagebox and build_status tracks them.
#
#   julia +release --project=scripts -t auto scripts/gammatau_backfill.jl <campaign_dir>
#
# Idempotent: runs whose gammatau_ + ic_ caches already exist are skipped (pair chips are
# re-rendered — cheap). Env: EDM_GAMMA_TRACE_OVERSAMPLE (default 4), same as the solver.

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner   # explicit Vern9 — pure ODE, no implicit-init solver needed
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using Serialization
using Statistics
using Printf
using TOML
using Dates
using CairoMakie
using RunManifests

include(joinpath(@__DIR__, "trajectory_products.jl"))   # gamma_trace + write_ic_products
include(joinpath(@__DIR__, "ll_system_chips.jl"))       # gamma_drain_product (main() is guarded)

const c = 137.03599908330932
const GAMMA_TRACE_OS = parse(Int, get(ENV, "EDM_GAMMA_TRACE_OVERSAMPLE", "4"))
GAMMA_TRACE_OS > 0 || error("EDM_GAMMA_TRACE_OVERSAMPLE must be > 0 for a backfill")

# Sunflower distribution — identical to inverse_thomson_scattering.jl.
const ϕgold = (1 + √5) / 2
sf_radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
function sunflower(n, α)
    b = round(Int, α * sqrt(n))
    return [[sf_radius(k, n, b) * cos(k * 2π / ϕgold^2),
             sf_radius(k, n, b) * sin(k * 2π / ϕgold^2)] for k in 1:n]
end

# Append `files` to the run's FINAL .reduced marker (record_reduction! writes .partial for the
# in-flight reduce; here the reduce is long done, so we edit the committed marker — atomically,
# tmp+mv, so a status collector never sees a half-written TOML). Legacy empty sentinels get the
# full header. Idempotent per basename.
function enrich_marker!(dir, uuid, files)
    path = joinpath(dir, "$(uuid).reduced")
    m = (isfile(path) && filesize(path) > 0) ? TOML.parsefile(path) : Dict{String, Any}(
        "run_id" => string(uuid), "reduced_at" => string(now()), "host" => gethostname(),
        "reduction" => Any[])
    red = get!(m, "reduction", Any[])
    for f in files
        bn = basename(f)
        any(e -> get(e, "file", "") == bn, red) && continue
        push!(red, Dict{String, Any}("file" => bn, "bytes" => filesize(joinpath(dir, bn))))
    end
    tmp = path * ".tmp"
    open(io -> TOML.print(io, m; sorted = true), tmp, "w")
    mv(tmp, path; force = true)
    return path
end

function backfill_run(dir, mfile)
    m = TOML.parsefile(mfile)
    uuid = m["provenance"]["run_id"]
    cfg, las, set = m["config"], m["laser"], m["setup"]
    if get(cfg, "scattering", "") != "inverse"
        println("skip $(first(uuid, 8)) — not an inverse run")
        return nothing
    end
    gt, ic = joinpath(dir, "gammatau_$uuid.jls"), joinpath(dir, "ic_$uuid.jls")
    if isfile(gt) && isfile(ic)
        println("skip $(first(uuid, 8)) — caches already present")
        return uuid
    end

    γ0 = Float64(cfg["gamma"])
    λ = las["wavelength"]
    ω = 2π * c / λ
    τ = las["temporal_width"]
    w₀ = las["w0"]
    Rmax = set["Rmax"]
    N = Int(cfg["N"])
    β = sqrt(1 - 1 / γ0^2)
    u⁰_t, u³_z = γ0 * c, c * sqrt(γ0^2 - 1)
    τi, τf = -cfg["tspan_tau"] * τ, cfg["tspan_tau"] * τ

    @named world = Worldline(:τ, :atomic)
    @named laser = LaguerreGaussLaser(;
        wavelength = λ, a0 = cfg["a0"], beam_waist = w₀,
        radial_index = Int(las["p"]), azimuthal_index = Int(las["m"]),
        world, temporal_profile = Symbol(las["profile"]), temporal_width = τ,
        focus_position = las["focus_position"], polarization = Symbol(las["pol"]),
        initial_phase = las["phi0"], k_direction = [0, 0, -1],
    )
    elec = if get(cfg, "system", "classical") == "ll"
        @named elec = LandauLifshitzElectron(; laser)
    else
        @named elec = ClassicalElectron(; laser)
    end
    sys = mtkcompile(elec)

    R₀ = Rmax * sunflower(N, 2)
    nb, l, chirp = Int(cfg["bunch_nb"]), Int(cfg["bunch_l"]), Float64(cfg["bunch_chirp"])
    Z = set["Z"]
    chirp == 0 || error("bunch_chirp ≠ 0 backfill not supported (chirp needs the p=2 |m|=2 u_rel)")
    bunch_dz(r) = nb == 0 ? 0.0 :
        ((1 + β) / 2) * ((r[1]^2 + r[2]^2) / (2Z) + l * atan(r[2], r[1]) / (2π) * λ / nb)
    xμ = [[u⁰_t * τi, r..., u³_z * τi + bunch_dz(r)] for r in R₀]

    u⁰ = [u⁰_t, 0.0, 0.0, u³_z]
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        sys, [sys.x => SVector{4}(xμ[1]...), sys.u => SVector{4}(u⁰...)], (τi, τf),
        u0_constructor = SVector{8}, fully_determined = true
    )
    set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
    function prob_func(prob, ctx)
        u0, p = set_x(prob, SVector{8}(xμ[ctx.sim_id]..., u⁰...))
        return remake(prob; u0, p)
    end
    saveat = collect(τi:((2π / ω) / (γ0 * (1 + β)) / parse(Float64, cfg["interp_saveat"])):τf)
    t_solve = @elapsed solution = solve(
        EnsembleProblem(prob; prob_func, safetycopy = false), Vern9(), EnsembleThreads();
        reltol = cfg["reltol"], abstol = cfg["abstol"], dtmax = cfg["dtmax"],
        trajectories = N, saveat
    )
    trajs = trajectory_interpolants(solution)
    @info "trajectories re-solved" first(uuid, 8) cfg["system"] t_solve knots = length(saveat)

    knot_dt = saveat[2] - saveat[1]
    γτ_grid = collect(τi:(knot_dt / GAMMA_TRACE_OS):τf)
    γmean, γmin, γmax, γdrain = gamma_trace(trajs, γτ_grid, c, γ0, τf)
    serialize(gt, (; τs = γτ_grid, γ0, ω, τ_pulse = τ,
        γmean, γmin, γmax, drain = γdrain, oversample = GAMMA_TRACE_OS,
        knots_per_period = parse(Float64, cfg["interp_saveat"])))
    @info "γ(τ)/γ₀ trace serialized" first(uuid, 8) mean_drain = sum(γdrain) / N

    write_ic_products(xμ, u⁰, [bunch_dz(r) for r in R₀], dir, uuid;
        γ0, λ, w₀, nb, l, chirp)
    enrich_marker!(dir, uuid, [gt, ic])
    return uuid
end

function main_backfill(dir)
    mfiles = sort(filter(f -> startswith(basename(f), "run_") && endswith(f, ".toml"),
        readdir(dir; join = true)))
    isempty(mfiles) && error("no run_*.toml in $dir")
    done = String[]
    for mf in mfiles
        u = backfill_run(dir, mf)
        u === nothing || push!(done, u)
        GC.gc()
    end
    # Pair pass: the classical|LL γ(τ)/γ₀ overlay chips (2-parent, comparison card).
    cells = []
    for mf in mfiles
        m = TOML.parsefile(mf)
        c_ = m["config"]
        push!(cells, (; id = m["provenance"]["run_id"],
            system = get(c_, "system", "classical"), gamma = c_["gamma"], a0 = c_["a0"],
            iters = get(c_, "newton_iters", 2)))
    end
    groups = Dict()
    for c_ in cells
        push!(get!(groups, (c_.gamma, c_.a0, c_.iters), []), c_)
    end
    for g in values(groups)
        cl = findfirst(d -> d.system == "classical", g)
        ll = findfirst(d -> d.system == "ll", g)
        (cl === nothing || ll === nothing) && continue
        gamma_drain_product(dir, g[cl], g[ll], g[cl].gamma, g[cl].a0)
    end
    println("backfilled $(length(done)) runs in $dir")
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_backfill(isempty(ARGS) ? "." : ARGS[1])
end
