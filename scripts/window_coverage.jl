# scripts/window_coverage.jl — observer-window coverage chip for an ARCHIVED run (CPU only):
# re-solves the run's electrons from its manifest (same reconstruction as plot_pixel_traces.jl /
# analyze_trajectories.jl), rebuilds the production screen + observer window, and runs
# `window_coverage`: does every pixel see every electron's full history inside the sampled
# window? A run whose window outlives a trajectory's retarded image silently lacks that
# electron's tail contribution in the cube (the kernel skips the clipped slots). Solver runs made
# after the [window] section existed already carry the verdict; this chip backfills older runs
# and draws the per-electron margins.
#
#   julia --project=scripts scripts/window_coverage.jl <run_manifest.toml>
#   env: EDM_N       cap on re-solved electrons (uniformly spaced subset incl. the outermost;
#                    default = all, exact); EDM_OUTDIR (default: next to the manifest)
# Writes window_coverage_<run_id>.png + derived_window_coverage_<id8>.toml (dashboard chip).

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using SciMLBase
using SymbolicIndexingInterface: setsym_oop
using StaticArrays
using CairoMakie
using TOML
using Printf
using RunManifests   # screen_halfwidth, window_start

length(ARGS) == 1 || error("usage: window_coverage.jl <run_manifest.toml>")
const MFILE = abspath(ARGS[1])
isfile(MFILE) || error("no manifest at $MFILE")
const OUTDIR = get(ENV, "EDM_OUTDIR", dirname(MFILE))
const N_CAP = parse(Int, get(ENV, "EDM_N", "0"))   # 0 = all

const c = 137.03599908330932   # speed of light, atomic units (as in the solver scripts)

# ── Reconstruct the run (mirrors analyze_trajectories.jl `reconstruct`) ──
m = TOML.parsefile(MFILE)
laser_p, cfg, setup, prov = m["laser"], m["config"], m["setup"], m["provenance"]
λ = Float64(laser_p["wavelength"])
w₀ = Float64(laser_p["w0"])
pol = Symbol(laser_p["pol"])
τ0 = Float64(laser_p["temporal_width"])
a₀ = Float64(cfg["a0"])
φ₀ = Float64(laser_p["phi0"])
Nfull = Int(cfg["N"])
Rmax = Float64(setup["Rmax"])
τi, τf = Float64(setup["τi"]), Float64(setup["τf"])
Z = Float64(setup["Z"])
Nx = Int(cfg["Nx"])
Ny = Int(get(cfg, "Ny", Nx))
N_samples = Int(cfg["N_samples"])
spp = Int(cfg["samples_per_period"])
HALFW = screen_halfwidth(m)   # a.u. (absolute half-width)
reltol = Float64(get(cfg, "reltol", 1.0e-12))
abstol = Float64(cfg["abstol"])
interp_saveat = string(get(cfg, "interp_saveat", "adaptive"))
ω = 2π * c / λ
RUN_TAG = String(prov["run_id"])
inverse = get(cfg, "scattering", "") == "inverse"
γboost = inverse ? Float64(get(cfg, "gamma", 1.0)) : 1.0
u⁰_t = γboost * c
u³_z = inverse ? c * sqrt(γboost^2 - 1) : 0.0
βz = u³_z / u⁰_t
dtmax_cfg = Float64(get(cfg, "dtmax", Inf))
dtmax_kw = isfinite(dtmax_cfg) ? (; dtmax = dtmax_cfg) : (;)
saveat_kw = interp_saveat == "adaptive" ? (;) :
    (; saveat = collect(τi:((2π / ω) / (γboost * (1 + βz)) / parse(Float64, interp_saveat)):τf))
inverse && Int(get(cfg, "bunch_nb", 0)) > 0 &&
    @warn "bunched inverse run (bunch_nb > 0): Δz offsets are NOT reconstructed"

@named world = Worldline(:τ, :atomic)
@named laser = LaguerreGaussLaser(;
    wavelength = λ, a0 = a₀, beam_waist = w₀,
    radial_index = Int(laser_p["p"]), azimuthal_index = Int(laser_p["m"]),
    world, temporal_profile = Symbol(laser_p["profile"]), temporal_width = τ0,
    focus_position = Float64(laser_p["focus_position"]), polarization = pol,
    initial_phase = φ₀,
    (inverse ? (; k_direction = [0, 0, -1]) : (;))...,
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)
prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, [sys.x => [u⁰_t * τi, 0.0, 0.0, u³_z * τi], sys.u => [u⁰_t, 0.0, 0.0, u³_z]], (τi, τf);
    u0_constructor = SVector{8}, fully_determined = true
)

const ϕgold = (1 + √5) / 2
function sunflower(n, α)
    points = Vector{Vector{Float64}}()
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
        push!(points, [r * cos(k * 2π / ϕgold^2), r * sin(k * 2π / ϕgold^2)])
    end
    return points
end
R₀ = Rmax * sunflower(Nfull, 2)
xμ = [[u⁰_t * τi, r..., u³_z * τi] for r in R₀]
# Electron subset: all (exact) or EDM_N uniformly spaced sunflower indices (the sunflower is
# radially ordered, so a uniform index subset spans r₀ = 0 … Rmax including the outermost).
solve_idx = (N_CAP <= 0 || N_CAP >= Nfull) ? collect(1:Nfull) :
    unique(round.(Int, range(1, Nfull; length = N_CAP)))
r0 = [hypot(xμ[i][2], xμ[i][3]) for i in solve_idx]

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
function prob_func(prob, ctx)
    i = solve_idx[ctx.sim_id]
    u0, p = set_x(prob, SVector{8}(SVector{4}(xμ[i]...)..., SVector{4}(u⁰_t, 0.0, 0.0, u³_z)...))
    return remake(prob; u0, p)
end
@info "window coverage" MFILE RUN_TAG Nfull length(solve_idx) Nx Ny N_samples
t_traj = @elapsed sol = solve(
    EnsembleProblem(prob; prob_func, safetycopy = false), Vern9(), EnsembleThreads();
    reltol, abstol, trajectories = length(solve_idx), saveat_kw..., dtmax_kw...
)
@info "trajectories solved" t_traj
trajs = trajectory_interpolants(sol)

# ── The production screen + observer window ──
x⁰_samples = range(start = window_start(m), step = c * (2π / ω / spp), length = N_samples)
screen = ObserverScreen(LinRange(-HALFW, HALFW, Nx), LinRange(-HALFW, HALFW, Ny), Z, x⁰_samples; c)

t_cov = @elapsed cov = window_coverage(trajs, screen)
scale = Nfull / length(solve_idx)   # subset → whole-run extrapolation (exact when scale == 1)
@info "coverage" cov.ok cov.slot_fill cov.electrons_clipped cov.lead_margin_samples cov.tail_margin_samples t_cov
cov.ok || @warn "observer window NOT fully covered — $(cov.electrons_clipped)/$(length(solve_idx)) re-solved electrons are clipped; the cube lacks their contribution in the clipped slots" cov.slots_dropped cov.worst_electron

# ── Figure: per-electron margins vs initial radius ──
lead = [p.lead_margin for p in cov.per_electron]
tail = [p.tail_margin for p in cov.per_electron]
fig = Figure(size = (900, 380))
ax = Axis(fig[1, 1]; xlabel = "electron initial radius r₀ / Rmax", ylabel = "margin [observer samples]",
    title = @sprintf("Observer-window coverage — %s  (N_samples = %d, fill = %s)", first(RUN_TAG, 8), N_samples,
        cov.slots_executed === missing ? "?" : @sprintf("%.4f", cov.slot_fill)))
hlines!(ax, [0.0]; color = :red, linestyle = :dash, label = "clipping threshold")
scatter!(ax, r0 ./ Rmax, tail; markersize = 6, label = "tail margin (window end vs τf image)")
scatter!(ax, r0 ./ Rmax, lead; markersize = 6, marker = :utriangle, label = "lead margin (window start vs τi image)")
axislegend(ax; position = :rb)
pngfile = joinpath(OUTDIR, "window_coverage_$(RUN_TAG).png")
save(pngfile, fig)
println("figure → $pngfile")

# ── Derived sidecar (same shape as plot_pixel_traces.jl's) ──
repo_commit = try
    readchomp(`git -C $(pkgdir(ElectronDynamicsModels)) rev-parse HEAD`)
catch
    "unknown"
end
idtag = first(RUN_TAG, 8)
verdict = cov.ok ? "fully covered" : "$(cov.electrons_clipped) of $(length(solve_idx)) electrons clipped"
sidecar = Dict(
    "schema_version" => 1,
    "derived" => Dict(
        "depends_on" => [RUN_TAG],
        "kind" => "window_coverage",
        "label" => "observer-window coverage: $verdict",
        "plot" => basename(pngfile),
        "source" => basename(MFILE),
        "description" => "Does every pixel see every electron's full history inside the sampled " *
            "observer window? Host-side evaluation of the field kernels' own arrival-window formula " *
            "at the extremal pixels of each re-solved electron; negative margins mark slots the GPU " *
            "kernel skipped (missing contributions in the cube). Lead margin $(cov.lead_margin_samples), " *
            "tail margin $(cov.tail_margin_samples) samples.",
    ),
    "coverage" => Dict(
        "ok" => cov.ok,
        "electrons_solved" => length(solve_idx),
        "electrons_clipped" => cov.electrons_clipped,
        "electrons_clipped_scaled" => round(Int, cov.electrons_clipped * scale),
        "lead_margin_samples" => cov.lead_margin_samples,
        "tail_margin_samples" => cov.tail_margin_samples,
        "worst_electron" => cov.worst_electron == 0 ? 0 : solve_idx[cov.worst_electron],
        "slots_nominal" => Nfull * N_samples * Nx * Ny,
        (cov.slots_executed === missing ? () :
            ("slot_fill" => Float64(cov.slot_fill),
             "slots_executed_scaled" => round(Int, cov.slots_executed * scale)))...,
    ),
    "provenance" => Dict(
        "host" => readchomp(`hostname`), "repo_commit" => repo_commit,
        "script" => "window_coverage.jl",
        "timestamp" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())),
    ),
    "setup" => Dict("electrons" => N_CAP <= 0 ? "all" : string(length(solve_idx))),
)
scfile = joinpath(OUTDIR, "derived_window_coverage_$(idtag).toml")
open(scfile, "w") do io
    TOML.print(io, sidecar)
end
println("sidecar → $scfile")
