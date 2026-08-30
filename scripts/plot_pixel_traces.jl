# scripts/plot_pixel_traces.jl — per-electron E(t) traces at selected screen pixels.
#
# Usage: julia --project=scripts scripts/plot_pixel_traces.jl <run_manifest.toml>
#
# Reconstructs the run from its manifest (same pattern as analyze_trajectories.jl — the
# recorded wavelength/w0/Rmax carry any EDM_OMEGA_SCALE/EDM_SCREEN_HALFW geometry as-is),
# re-solves a handful of electron trajectories, and evaluates each electron's individual
# Liénard–Wiechert E(t) at a row of pixels along +x — the center pixel plus a few at given
# radii — via the CPU reference accumulate_field. The full-N coherent sum (scaled by 1/N,
# so coherent structure lands on the per-electron amplitude) is overlaid for comparison.
#
# ENV knobs:
#   EDM_TRACE_RADII      pixel radii as fractions of the screen half-extent, default "0,0.25,0.5,0.75,1.0"
#   EDM_TRACE_ELECTRONS  sunflower indices (csv); default 5 electrons at r/Rmax ≈ 0, ¼, ½, ¾, 1
#                        (k = N·f² since the sunflower radius grows as √k)
#   EDM_TRACE_TOTAL      1 (default) → also accumulate the full-N sum; 0 → skip (much faster)
#   EDM_OUTDIR           output dir, default = the manifest's directory
#
# Writes pixeltraces_<tag>.png + pixeltraces_<tag>.jls (per-electron + total traces, grids).

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner   # explicit Vern9 — trajectories + the retarded-time ODE
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using Printf
using CairoMakie
using Serialization
using TOML

const c = 137.03599908330932   # speed of light, atomic units (repo convention)

length(ARGS) == 1 || error("usage: plot_pixel_traces.jl <run_manifest.toml>")
const MFILE = abspath(ARGS[1])
isfile(MFILE) || error("no manifest at $MFILE")

const RADII = parse.(Float64, split(get(ENV, "EDM_TRACE_RADII", "0,0.25,0.5,0.75,1.0"), ","))
all(0 .<= RADII .<= 1) || error("EDM_TRACE_RADII are fractions of the screen half-extent (0…1)")
const TRACE_TOTAL = get(ENV, "EDM_TRACE_TOTAL", "1") == "1"
const OUTDIR = get(ENV, "EDM_OUTDIR", dirname(MFILE))

# ── Reconstruct the run (mirrors analyze_trajectories.jl reconstruct) ──
m = TOML.parsefile(MFILE)
laser_p, cfg, setup = m["laser"], m["config"], m["setup"]

λ = Float64(laser_p["wavelength"])
w₀ = Float64(laser_p["w0"])
pol = Symbol(laser_p["pol"])
τ0 = Float64(laser_p["temporal_width"])
a₀ = Float64(cfg["a0"])
φ₀ = Float64(laser_p["phi0"])
N = Int(cfg["N"])
Rmax = Float64(setup["Rmax"])
τi, τf = Float64(setup["τi"]), Float64(setup["τf"])
Z = Float64(setup["Z"])
N_samples = Int(cfg["N_samples"])
spp = Int(cfg["samples_per_period"])
# Mini-screen half-width = the RUN's screen. Current manifests record the zoom knob as
# [setup].screen_hw (absolute) / [config].screen_hw_w0; the old cfg.screen_halfw key missed
# them, so zoomed-screen runs (hw < 25 w₀) fell back to the 25 w₀ production framing and the
# outer trace rows sampled pixels OFF the cube's screen — beyond the narrow window's corner
# budget, so their bursts overran the window end and the chips showed a spurious end-of-pulse
# cutoff (bridge-refix γ=2/2.5, 2026-08-30). The real screen corner is contained by
# construction (tail margin intact), so on-screen rows never clip.
HALFW = "screen_hw" in keys(setup) ? Float64(setup["screen_hw"]) / w₀ :
    Float64(get(cfg, "screen_hw_w0", get(cfg, "screen_halfw", 25.0)))
reltol = Float64(get(cfg, "reltol", 1.0e-12))
abstol = Float64(cfg["abstol"])
interp_saveat = string(get(cfg, "interp_saveat", "adaptive"))
ω = 2π * c / λ
RUN_TAG = m["provenance"]["run_id"]

# Inverse-Thomson manifests (scattering=inverse): reversed laser + boosted electrons, the
# meet-at-origin timing of inverse_thomson_scattering.jl. Thomson manifests: γ=1, β=0 — all
# expressions below collapse to the original rest-electron forms.
γboost = Float64(get(cfg, "gamma", 1.0))
inverse = get(cfg, "scattering", "") == "inverse"
inverse && string(get(cfg, "window", "full")) != "full" &&
    @warn "narrow-window inverse manifest: mini-screen uses the FULL-window corner anchor"
u⁰_t = (inverse ? γboost : 1.0) * c
u³_z = inverse ? c * sqrt(γboost^2 - 1) : 0.0
βz = u³_z / u⁰_t
dtmax_cfg = Float64(get(cfg, "dtmax", Inf))
dtmax_kw = isfinite(dtmax_cfg) ? (; dtmax = dtmax_cfg) : (;)

# k = N·f² ⇒ r/Rmax ≈ f: radially uniform picks from the √k sunflower spiral.
sel = if isempty(get(ENV, "EDM_TRACE_ELECTRONS", ""))
    unique(clamp.([max(1, round(Int, N * f^2)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)], 1, N))
else
    parse.(Int, split(ENV["EDM_TRACE_ELECTRONS"], ","))
end
all(1 .<= sel .<= N) || error("EDM_TRACE_ELECTRONS out of 1:$N")

@info "pixel traces" MFILE RUN_TAG a₀ ω N sel RADII TRACE_TOTAL

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

R₀ = Rmax * sunflower(N, 2)
xμ = [[u⁰_t * τi, r..., u³_z * τi] for r in R₀]
r0 = [hypot(x[2], x[3]) for x in xμ]

# Solve only what the traces need: the full ensemble when the coherent sum is on,
# else just the selected electrons.
solve_idx = TRACE_TOTAL ? collect(1:N) : sel
set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
function prob_func(prob, ctx)
    i = solve_idx[ctx.sim_id]
    u0, p = set_x(prob, SVector{8}(SVector{4}(xμ[i]...)..., SVector{4}(u⁰_t, 0.0, 0.0, u³_z)...))
    return remake(prob; u0, p)
end
# Knots divide the PROPER-TIME carrier period T/(γ(1+β)) — the inverse script's convention;
# γ=1, β=0 (thomson) reduces to the original T/knots spacing.
saveat_kw = interp_saveat == "adaptive" ? (;) :
    (; saveat = collect(τi:((2π / ω) / (γboost * (1 + βz)) / parse(Float64, interp_saveat)):τf))
t_traj = @elapsed sol = solve(
    EnsembleProblem(prob; prob_func, safetycopy = false), Vern9(), EnsembleThreads();
    reltol, abstol, trajectories = length(solve_idx), saveat_kw..., dtmax_kw...
)
@info "trajectories solved" t_traj length(solve_idx)

trajs = trajectory_interpolants(sol)
traj_of = Dict(zip(solve_idx, trajs))

# ── Mini-screen: the requested pixels along +x (center first), production time window ──
xs = [f * HALFW * w₀ for f in RADII]
# Prefer the RECORDED window start (setup.x0_start, present on current manifests): exact for
# both :full and :narrow (burst-centred) windows — the corner-anchor formula is the pre-knob
# reconstruction fallback and is wrong for narrow runs.
x⁰_samples = range(
    start = haskey(m["setup"], "x0_start") ? Float64(m["setup"]["x0_start"]) :
        c * τi + hypot(Z, HALFW * w₀ + Rmax), step = c * (2π / ω / spp),
    length = N_samples,
)
screen = ObserverScreen(xs, [0.0], Z, x⁰_samples; c)   # Ny = 1: one y = 0 row of pixels

# Retarded-time ODE tolerances (CPU reference path; the trajectories carry the production spline).
ret_kw = (; reltol = 1.0e-10, abstol = 1.0e-12)

E_sel = Array{Float64}(undef, length(sel), N_samples, 3, length(xs))
for (ie, k) in enumerate(sel)
    fld = accumulate_field([traj_of[k]], screen, Vern9(); mode = Val(:total), ret_kw...)
    E_sel[ie, :, :, :] = fld.E[:, :, :, 1]
end
@info "per-electron traces done" length(sel)

E_tot = nothing
if TRACE_TOTAL
    t_tot = @elapsed fld_all = accumulate_field(trajs, screen, Vern9(); mode = Val(:total), ret_kw...)
    E_tot = fld_all.E[:, :, :, 1]
    @info "full-N coherent sum done" t_tot N
end

# ── Plot: rows = pixels, cols = components; per-electron lines + total/N overlay.
# Full-window figure (envelopes) + a ±EDM_TRACE_ZOOM_PERIODS zoom (carrier + relative phases —
# the coherence content the traces exist for; invisible at the full window's scale). ──
const ZOOM = parse(Float64, get(ENV, "EDM_TRACE_ZOOM_PERIODS", "3"))
T_units = λ                     # x⁰ is a length (c·t): one laser period ⇔ one wavelength
npix, ncomp = length(xs), 3
colors = [Makie.cgrad(:viridis)[f] for f in range(0, 0.85, length = length(sel))]
osc = get(cfg, "omega_scale", 1.0)

function build_fig(xlim)
    fig = Figure(size = (1560, 250 * npix + 110))
    for (ip, xpix) in enumerate(xs)
        R_pix = hypot(xpix, Z)
        t_rel = (collect(x⁰_samples) .- R_pix) ./ T_units
        for ic in 1:ncomp
            ax = Axis(fig[ip, ic];
                xlabel = ip == npix ? "(t − R_pix/c) / T_laser" : "",
                ylabel = ic == 1 ? @sprintf("r = %.2f·halfw\nE (a.u.)", RADII[ip]) : "",
                title = ip == 1 ? ("Ex", "Ey", "Ez")[ic] : "",
                ytickformat = vs -> [@sprintf("%.1e", v) for v in vs])
            for (ie, k) in enumerate(sel)
                lines!(ax, t_rel, E_sel[ie, :, ic, ip]; color = colors[ie], linewidth = 1.0)
            end
            E_tot === nothing || lines!(ax, t_rel, E_tot[:, ic, ip] ./ N;
                color = :black, linestyle = :dash, linewidth = 1.2)
            xlim === nothing || xlims!(ax, -xlim, xlim)
        end
    end
    leg_entries = [LineElement(; color = colors[ie]) for ie in eachindex(sel)]
    leg_labels = [@sprintf("e#%d  (r₀ = %.2f Rmax)", k, r0[k] / Rmax) for k in sel]
    if E_tot !== nothing
        push!(leg_entries, LineElement(; color = :black, linestyle = :dash))
        push!(leg_labels, "coherent sum / N")
    end
    Legend(fig[npix + 1, 1:ncomp], leg_entries, leg_labels; orientation = :horizontal, framevisible = false)
    Label(fig[0, 1:ncomp],
        @sprintf("Per-electron E(t) at screen pixels — %s  (a0=%g, ω-scale=%.4g, N=%d)%s",
            RUN_TAG, a₀, osc, N, xlim === nothing ? "" : @sprintf("  [zoom ±%g T]", xlim));
        fontsize = 18, font = :bold)
    return fig
end

pngfile = joinpath(OUTDIR, "pixeltraces_$(RUN_TAG).png")
save(pngfile, build_fig(nothing))
println("saved → $pngfile")
zoomfile = joinpath(OUTDIR, "pixeltraces_zoom_$(RUN_TAG).png")
save(zoomfile, build_fig(ZOOM))
println("saved → $zoomfile")

# Derived sidecars (same schema as harmonic_products' derived_h*.toml) so the dashboard
# stager/builder picks the traces up as chips of the parent run.
repo_commit = try
    readchomp(`git -C $(pkgdir(ElectronDynamicsModels)) rev-parse HEAD`)
catch
    "unknown"
end
idtag = first(RUN_TAG, 8)
for (kind, label, png, extra) in (
        ("pixeltraces", "per-electron E(t) traces",
            basename(pngfile), "Full observation window (envelopes)."),
        ("pixeltraces_zoom", "per-electron E(t) traces (zoom)",
            basename(zoomfile), @sprintf("Zoom ±%g laser periods around the peak (carrier + relative phases).", ZOOM)),
    )
    sidecar = Dict(
        "schema_version" => 1,
        "derived" => Dict(
            "depends_on" => [RUN_TAG],
            "kind" => kind,
            "label" => label,
            "plot" => png,
            "source" => basename(MFILE),
            "description" => "Individual Liénard–Wiechert E(t) of $(length(sel)) electrons " *
                "(r₀/Rmax ≈ $(join([@sprintf("%.2f", r0[k] / Rmax) for k in sel], ", "))) at pixels " *
                "r/halfw = $(join(RADII, ", ")) along +x, with the full-N coherent sum scaled by 1/N. " * extra,
        ),
        "plot_params" => Dict(
            "radii_frac" => RADII, "electrons" => sel,
            "zoom_periods" => kind == "pixeltraces_zoom" ? ZOOM : 0.0,
        ),
        "provenance" => Dict(
            "host" => readchomp(`hostname`), "repo_commit" => repo_commit,
            "script" => "plot_pixel_traces.jl",
            "timestamp" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())),
        ),
        "setup" => Dict("field" => "total"),
    )
    scfile = joinpath(OUTDIR, "derived_$(kind)_$(idtag).toml")
    open(scfile, "w") do io
        TOML.print(io, sidecar)
    end
    println("sidecar → $scfile")
end

jlsfile = joinpath(OUTDIR, "pixeltraces_$(RUN_TAG).jls")
serialize(jlsfile, (; x⁰_samples = collect(x⁰_samples), xs, RADII, sel, r0_sel = r0[sel],
    E_sel, E_tot, N, a₀, ω, λ, Z, HALFW, w₀, Rmax, run_id = RUN_TAG))
println("serialized → $jlsfile")
