# CPU trajectory analyses for thomson_scattering.jl runs, reconstructed from run_*.toml
# manifests. The big field/trajectory data of the source runs is NOT needed — trajectories
# are cheap to recompute, so only the run parameters (the small run_*.toml) are required.
#
# Two analysis modes (EDM_ANALYSIS):
#
# "trajectories" (default) — electron-worldline visualization (the φ₀=−π/2 "cm" campaign).
#   Same laser + sunflower electron distribution as scripts/thomson_scattering.jl, re-solved
#   as a dense CPU ensemble. Per source run (= per a₀) it writes TWO PNGs — the worldlines
#   projected onto the zx and zy planes (axes in units of w₀), each with marginal histograms
#   of the FINAL electron positions (at τf) — plus a run_<uuid>.toml manifest.
#
#   Worldlines are sampled with the solution's DENSE interpolation via symbolic-expression
#   indexing — `sol(ts; idxs = sys.x[4] / w₀)` — the same `idxs` machinery SciMLBase's Makie
#   plot recipe uses internally (we call it directly so each line can be coloured by its
#   initial radius r₀/w₀, which the recipe's label-only PlotSpecs can't carry).
#
#   Coordinate mode (EDM_COORDS): the transverse motion here is a real ~0.008·w₀ ponderomotive
#   displacement — larger than the ~0.001·w₀ forward z-drift — but it's only ~0.25% of the
#   ±3.25·w₀ initial disk, so on ABSOLUTE transverse axes it looks frozen. "displacement"
#   plots (x−x₀, y−y₀) vs z, centring every electron; "absolute" keeps beam-frame positions.
#
# "spline-error" — anatomy of the small-a₀ 2ω floor (see the section header further down).
#   Per electron, measures the CubicSpline interpolants the production radiation path actually
#   feeds accumulate_field (state + acceleration splines through the SAVED trajectory knots)
#   against the dense adaptive Vern9 continuous extension they are built from — then shows how
#   those per-electron interpolation errors are distributed in time, how far the ensemble mean
#   is knocked down by their mutual (destructive) interference, and that the surviving coherent
#   residual concentrates at 2ω for adaptive knots but not for uniform knots: the origin of the
#   h2/h1[B] floor and of the EDM_INTERP_SAVEAT fix.
#
# ENV knobs (all optional):
#   EDM_ANALYSIS         "trajectories" (default) or "spline-error"
#   EDM_SOURCE_CAMPAIGN  dir of run_*.toml to take parameters from
#                        (default /storage/pool/smc/thomson_runs/field_campaign_cm_phi0)
#   EDM_SOURCE_RUN       substring of the manifest filename to select specific run(s); "" = all.
#                        Useful for spline-error on campaigns like floor_probe whose manifests
#                        differ only in [config].interp_saveat (ignored on reconstruction).
#   EDM_INITIAL_PHASE    "manifest" (default) ⇒ each run's own [config].initial_phase (−π/2 for
#                        the cm_phi0 source); or a numeric value to override every run
#   EDM_OUTDIR           output dir (default <pkg>/runs/field_campaign_cm_phi0, or
#                        <pkg>/runs/spline_error in spline-error mode)
#   EDM_N                cap electrons per run (default = manifest N, but 400 in spline-error
#                        mode where the error statistics converge fast and the stored error
#                        matrix is nt×N; a smaller N re-generates a coarser full-radius disk)
#
# trajectories mode:
#   EDM_COORDS           "absolute" (default) or "displacement" (plot x−x₀, y−y₀ vs z)
#   EDM_N_PLOT_TRAJ      worldlines actually drawn (default 800; histograms always use all N)
#   EDM_NPATH            dense τ samples per drawn worldline (default 10000 displacement / 1500 absolute)
#
# spline-error mode:
#   EDM_ERR_SPP          fine-grid samples per laser period for the error signals (default 32)
#   EDM_ERR_SAVEAT       knots/period of the uniform-saveat comparison variants (default 16)
#   EDM_ERR_COMPONENT    spatial acceleration component analyzed: "x" (default), "y", "z"
#   EDM_ERR_N_TRACE      per-electron error traces drawn / knot-spacing lines (default 16)
#   EDM_ERR_ZOOM         zoomed-window width in laser periods, centred on the envelope peak (default 3)
#   EDM_ERR_EXPORT       "" (default, off) | "1" ⇒ splerr_report_<comp>_<base>_<id>.json in
#                        EDM_OUTDIR | a filename/path — serialize the report payload (per-electron
#                        full-res zoom traces + knot times, envelopes, spectra, 2ω-band traces)
#                        for the interactive dashboard report; also adds the splerr_zoomcase_*
#                        figure. Pure retention/serialization: no computed quantity changes.
#   EDM_ERR_EXPORT_PERIODS  full-resolution export window in laser periods, centred on the
#                        envelope peak (default 48 ≈ the pulse core)
#   EDM_ERR_EXPORT_IDS   comma-separated electron indices to export; "" (default) ⇒ 4 electrons
#                        radius-uniform over the disk (r₀/Rmax ≈ 0, 1/3, 2/3, 1)
#   EDM_ERR_RELTOL       override the solve reltol ("" = manifest value) — the knot-DENSITY
#   EDM_ERR_ABSTOL       override the solve abstol ("" = manifest value)   levers
#
#   julia --project=scripts scripts/analyze_trajectories.jl

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner   # explicit Vern9 — pure ODE, no implicit-init solver needed
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using DataInterpolations     # CubicSpline — the production trajectory interpolant
using FFTW                   # error spectra (spline-error mode)
using JSON                   # report-payload export (spline-error mode, EDM_ERR_EXPORT)
using LinearAlgebra
using Statistics
using Printf
using CairoMakie
using TOML
using UUIDs

include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts
include(joinpath(@__DIR__, "manifest.jl"))      # RunManifests: write_run_manifest

# Atomic units
const c = 137.03599908330932

# ── Configuration (ENV-overridable) ──
const ANALYSIS = get(ENV, "EDM_ANALYSIS", "trajectories")      # "trajectories" | "spline-error"
ANALYSIS in ("trajectories", "spline-error") || error("EDM_ANALYSIS must be \"trajectories\" or \"spline-error\", got \"$ANALYSIS\"")
const SOURCE_CAMPAIGN = get(ENV, "EDM_SOURCE_CAMPAIGN", "/storage/pool/smc/thomson_runs/field_campaign_cm_phi0")
const SOURCE_RUN = get(ENV, "EDM_SOURCE_RUN", "")              # manifest-filename substring filter; "" = all
const PHASE_SPEC = get(ENV, "EDM_INITIAL_PHASE", "manifest")   # "manifest" ⇒ each run's own φ₀; or a numeric override
const COORDS = get(ENV, "EDM_COORDS", "absolute")                  # "absolute" | "displacement"
const OUTDIR = get(ENV, "EDM_OUTDIR", joinpath(pkgdir(ElectronDynamicsModels), "runs",
    ANALYSIS == "spline-error" ? "spline_error" : "field_campaign_cm_phi0"))
# Spline-error statistics converge fast with N and keep an nt×N error matrix in memory, so that
# mode defaults to 400 electrons; EDM_N still overrides in both directions and in both modes.
const N_CAP = haskey(ENV, "EDM_N") ? parse(Int, ENV["EDM_N"]) :
    ANALYSIS == "spline-error" ? 400 : typemax(Int)
const N_PLOT_TRAJ = parse(Int, get(ENV, "EDM_N_PLOT_TRAJ", "800"))
# Dense samples/worldline. Displacement zooms in on the (circular, circular_minus) quiver, which
# needs ~26 pts/optical-cycle to render smoothly; absolute hides the quiver under the disk so 1500
# suffices. Default tracks COORDS; override with EDM_NPATH.
const NPATH = parse(Int, get(ENV, "EDM_NPATH", COORDS == "displacement" ? "10000" : "1500"))
COORDS in ("absolute", "displacement") || error("EDM_COORDS must be \"absolute\" or \"displacement\", got \"$COORDS\"")
# spline-error mode
const ERR_SPP = parse(Int, get(ENV, "EDM_ERR_SPP", "32"))            # fine-grid samples/period
const ERR_SAVEAT = parse(Int, get(ENV, "EDM_ERR_SAVEAT", "16"))      # uniform-variant knots/period
const ERR_COMPONENT = get(ENV, "EDM_ERR_COMPONENT", "x")             # acceleration component
const ERR_N_TRACE = parse(Int, get(ENV, "EDM_ERR_N_TRACE", "16"))    # traces drawn
const ERR_ZOOM = parse(Float64, get(ENV, "EDM_ERR_ZOOM", "3"))       # zoom window, laser periods
ERR_COMPONENT in ("x", "y", "z") || error("EDM_ERR_COMPONENT must be \"x\", \"y\" or \"z\", got \"$ERR_COMPONENT\"")
# report export (all off unless EDM_ERR_EXPORT is set)
const ERR_EXPORT = get(ENV, "EDM_ERR_EXPORT", "")                    # "" | "1" | filename/path
const ERR_EXPORT_PERIODS = parse(Float64, get(ENV, "EDM_ERR_EXPORT_PERIODS", "48"))
const ERR_EXPORT_IDS = get(ENV, "EDM_ERR_EXPORT_IDS", "")            # "" ⇒ radius-uniform 4
# solve-tolerance overrides ("" ⇒ the manifest's own values) — the reltol/abstol levers:
# tighter/looser stepping changes the adaptive knot DENSITY but not its dynamics-locked
# placement, so these test density-vs-placement without touching the source campaign.
const ERR_RELTOL = get(ENV, "EDM_ERR_RELTOL", "")
const ERR_ABSTOL = get(ENV, "EDM_ERR_ABSTOL", "")

# φ₀ for a given manifest: the numeric override (same for every run), or the manifest's own value.
phi0_for(manifest) = PHASE_SPEC == "manifest" ?
    Float64(manifest["config"]["initial_phase"]) : parse(Float64, PHASE_SPEC)

# Display φ₀ as a tidy multiple of π where it is one, else a plain number.
function phi_label(φ)
    for (val, str) in ((-π / 2, "−π/2"), (π / 2, "π/2"), (0.0, "0"), (π, "π"), (-π, "−π"))
        isapprox(φ, val; atol = 1e-9) && return str
    end
    return @sprintf("%.4g", φ)
end

# ── Sunflower distribution + per-a₀ tolerance — identical to thomson_scattering.jl ──
const GOLDEN = (1 + √5) / 2
radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
function sunflower(n, α)
    pts = Vector{NTuple{2, Float64}}(undef, n)
    stride = 2π / GOLDEN^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        θ = k * stride
        pts[k] = (r * cos(θ), r * sin(θ))
    end
    return pts
end
function abserr(a₀)
    amp = log10(a₀)
    expo = -amp^2 / 27 + 32amp / 27 - 220 / 27
    return 10^expo
end

# ── One marginal-histogram figure: worldlines in a (z, transverse) plane + final-position hists ──
# `zpath`/`tpath`/`cval` are the NaN-separated polylines (one NaN between trajectories) and the
# matching per-vertex colour (initial radius r₀/w₀). `zf`/`tf` are ALL electrons' final positions.
function marginal_figure(; zpath, tpath, cval, zf, tf, crange, zlabel, tlabel, title)
    fig = Figure(size = (760, 720))
    Label(fig[0, 1:2], title; fontsize = 15, font = :bold)
    ax_top = Axis(fig[1, 1]; ylabel = "count")
    ax = Axis(fig[2, 1]; xlabel = zlabel, ylabel = tlabel)
    # count axis on TOP of the right histogram so its ticks/label don't collide with the main
    # axis' transverse tick labels at the column boundary.
    ax_right = Axis(fig[2, 2]; xlabel = "count", xaxisposition = :top)

    lines!(ax, zpath, tpath; color = cval, colormap = :viridis, colorrange = crange,
        alpha = 0.35, linewidth = 0.6)
    hist!(ax_top, zf; bins = 60, color = :steelblue)
    hist!(ax_right, tf; bins = 60, direction = :x, color = :steelblue)

    linkxaxes!(ax, ax_top)       # top histogram shares the z (beam-axis) range
    linkyaxes!(ax, ax_right)     # right histogram shares the transverse range
    hidexdecorations!(ax_top; grid = false)     # z labels live on the main axis; keep counts
    hideydecorations!(ax_right; grid = false)   # transverse labels live on the main axis; keep counts

    Colorbar(fig[3, 1]; colormap = :viridis, colorrange = crange,
        vertical = false, flipaxis = false, height = 12, label = L"r_0/w_0")

    rowsize!(fig.layout, 1, Relative(0.18))
    colsize!(fig.layout, 2, Relative(0.18))
    colgap!(fig.layout, 6)
    rowgap!(fig.layout, 6)
    return fig
end

# ── Shared reconstruction: one run_*.toml manifest → model, ODE problem, ensemble, disk ──
# Everything both analysis modes need: the exact thomson_scattering.jl laser + sunflower
# distribution rebuilt from the run parameters, with φ₀ per PHASE_SPEC. Solver tolerances are
# the manifest's own [config].reltol/abstol where recorded, with the thomson defaults as
# fallback (only the spline-error mode uses them — knot patterns depend on the tolerances).
function reconstruct(mfile)
    m = TOML.parsefile(mfile)
    laser_p, cfg, setup, prov = m["laser"], m["config"], m["setup"], m["provenance"]

    λ = Float64(laser_p["wavelength"])
    w₀ = Float64(laser_p["w0"])
    m_az = Int(laser_p["m"])
    p_rad = Int(laser_p["p"])
    pol = Symbol(laser_p["pol"])
    profile = Symbol(laser_p["profile"])
    τ0 = Float64(laser_p["temporal_width"])
    z_focus = Float64(laser_p["focus_position"])
    a₀ = Float64(cfg["a0"])
    Nfull = Int(cfg["N"])
    Rmax = Float64(setup["Rmax"])
    τi = Float64(setup["τi"])
    τf = Float64(setup["τf"])
    φ₀ = phi0_for(m)
    parent_run_id = get(prov, "run_id", "unknown")
    N = min(Nfull, N_CAP)
    reltol = Float64(get(cfg, "reltol", 1.0e-12))
    abstol = Float64(get(cfg, "abstol", abserr(a₀)))

    # Laser + electron model — same construction as thomson_scattering.jl, φ₀ overridden.
    @named world = Worldline(:τ, :atomic)
    @named laser = LaguerreGaussLaser(;
        wavelength = λ, a0 = a₀, beam_waist = w₀,
        radial_index = p_rad, azimuthal_index = m_az,
        world, temporal_profile = profile, temporal_width = τ0,
        focus_position = z_focus, polarization = pol, initial_phase = φ₀,
    )
    @named elec = ClassicalElectron(; laser)
    sys = mtkcompile(elec)

    tspan = (τi, τf)
    x⁰ = [τi * c, 0.0, 0.0, 0.0]
    u⁰ = [c, 0.0, 0.0, 0.0]
    u0 = [sys.x => x⁰, sys.u => u⁰]
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        sys, u0, tspan; u0_constructor = SVector{8}, fully_determined = true
    )

    # Initial transverse positions (sunflower scaled to Rmax) + per-electron initial radius.
    R₀ = sunflower(N, 2)
    xμ = [[τi * c, Rmax * p[1], Rmax * p[2], 0.0] for p in R₀]
    r0 = [hypot(x[2], x[3]) for x in xμ]

    set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
    function prob_func(prob, ctx)
        i = ctx.sim_id
        x_new = SVector{4}(xμ[i]...)
        u_new = SVector{4}(c, 0.0, 0.0, 0.0)
        u0, p = set_x(prob, SVector{8}(x_new..., u_new...))
        return remake(prob; u0, p)
    end
    ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)

    return (; m, λ, w₀, ω = 2π * c / λ, m_az, p_rad, pol, profile, τ0, z_focus,
        a₀, N, Rmax, τi, τf, φ₀, parent_run_id, reltol, abstol, sys, prob, ensemble, xμ, r0)
end

# ── Analyze one source run: reconstruct, re-solve, plot zx + zy, write manifest ──
function analyze_run(mfile)
    rec = reconstruct(mfile)
    (; λ, w₀, m_az, p_rad, pol, profile, τ0, z_focus,
        a₀, N, Rmax, τi, τf, φ₀, parent_run_id, sys, prob, ensemble, xμ, r0) = rec

    @info "run" file = basename(mfile) a₀ φ₀ N COORDS parent_run_id

    # Dense solve (no saveat) so the worldlines can be sampled with the continuous interpolation;
    # only ~1.3k steps/trajectory ⇒ <1 GB for 10⁴ electrons.
    t_solve = @elapsed solution = solve(
        ensemble, Vern9(), EnsembleThreads();
        reltol = 1.0e-12, abstol = abserr(a₀), trajectories = N,
    )
    sols = solution.u
    @info "trajectories solved" t_solve steps_per_traj = round(Int, sum(length(s.t) for s in sols) / N)

    # Symbolic-expression getters with the w₀ scaling baked in (the SciMLBase `idxs` machinery).
    ez = sys.x[4] / w₀      # z/w₀          (beam axis; z₀ = 0)
    ex = sys.x[2] / w₀      # x/w₀          (transverse)
    ey = sys.x[3] / w₀      # y/w₀          (transverse)
    gz, gx, gy = getsym(prob, ez), getsym(prob, ex), getsym(prob, ey)

    # Initial transverse positions in w₀ units (for the displacement framing; z₀ = 0).
    x0w = [xμ[i][2] / w₀ for i in 1:N]
    y0w = [xμ[i][3] / w₀ for i in 1:N]
    disp = COORDS == "displacement"

    # Final positions of ALL electrons (last saved point = τf), in w₀ units (centred if displacement).
    zf = [last(gz(s)) for s in sols]
    xf = [last(gx(s)) - (disp ? x0w[i] : 0.0) for (i, s) in enumerate(sols)]
    yf = [last(gy(s)) - (disp ? y0w[i] : 0.0) for (i, s) in enumerate(sols)]

    # Worldlines drawn: an evenly-spaced subsample over the (radius-ordered) ensemble, densely
    # interpolated at NPATH τ points and joined into NaN-separated polylines with a per-vertex
    # colour (r₀/w₀). z is shared between the two planes; transverse is x or y (centred if displacement).
    ts = range(τi, τf; length = NPATH)
    plot_idx = unique(round.(Int, range(1, N; length = min(N_PLOT_TRAJ, N))))
    zpath = Float64[]; xpath = Float64[]; ypath = Float64[]; cval = Float64[]
    for i in plot_idx
        zs = sols[i](ts; idxs = ez).u
        xs = sols[i](ts; idxs = ex).u
        ys = sols[i](ts; idxs = ey).u
        disp && (xs = xs .- x0w[i]; ys = ys .- y0w[i])
        r = r0[i] / w₀
        append!(zpath, zs); push!(zpath, NaN)
        append!(xpath, xs); push!(xpath, NaN)
        append!(ypath, ys); push!(ypath, NaN)
        append!(cval, fill(r, length(zs))); push!(cval, r)
    end
    crange = (0.0, Rmax / w₀)

    philab = phi_label(φ₀)
    ctag = disp ? "disp" : "abs"
    base = @sprintf("%s_a0_%.0e_phi%.4f", ctag, a₀, φ₀)
    id8 = first(string(uuid4()), 8)
    mkpath(OUTDIR)

    zlabel = L"z/w_0"
    plots = String[]
    for (plane, tpath, tf, tlabel) in (
            ("zx", xpath, xf, disp ? L"(x-x_0)/w_0" : L"x/w_0"),
            ("zy", ypath, yf, disp ? L"(y-y_0)/w_0" : L"y/w_0"),
        )
        title = @sprintf("Electron trajectories — %s plane%s  (a₀=%.0e, φ₀=%s, N=%d)",
            plane, disp ? " (displacement)" : "", a₀, philab, N)
        fig = marginal_figure(; zpath, tpath, cval, zf, tf, crange, zlabel, tlabel, title)
        fname = "traj_$(plane)_$(base)_$(id8).png"
        save(joinpath(OUTDIR, fname), fig)
        push!(plots, fname)
        println("saved → $(joinpath(OUTDIR, fname))")
    end

    # Standalone analysis-node manifest (new φ₀ ⇒ a fresh computation, not a derived field product).
    run_id = string(uuid4())
    config = Dict(
        "a0" => a₀, "initial_phase" => φ₀, "N" => N, "n_plot_traj" => length(plot_idx),
        "coords" => COORDS, "kind" => "trajectories",
        "source_campaign" => basename(rstrip(SOURCE_CAMPAIGN, '/')), "source_run" => parent_run_id,
    )
    laser_out = Dict(
        "wavelength" => λ, "a0" => a₀, "w0" => w₀, "m" => m_az, "p" => p_rad,
        "pol" => string(pol), "profile" => string(profile),
        "temporal_width" => τ0, "focus_position" => z_focus, "phi0" => φ₀,
    )
    setup_out = Dict("Rmax" => Rmax, "τi" => τi, "τf" => τf)
    write_run_manifest(OUTDIR; run_id, script = basename(PROGRAM_FILE),
        config, laser = laser_out, setup = setup_out, plots)
    println("manifest → $(joinpath(OUTDIR, "run_$(run_id).toml"))")
    return run_id
end

# ════════════════════════════════════════════════════════════════════════════════════════
# Spline-error analysis (EDM_ANALYSIS=spline-error) — anatomy of the small-a₀ 2ω floor.
#
# The production radiation path never sees the ODE solution itself: accumulate_field sees
# CubicSplines through the SAVED trajectory knots — the state spline `itp` and the
# acceleration spline `a_itp` (the radiated far field is ∝ 𝔞; kernel_rk4.jl evaluates
# `a_itp(τ)`). With the default adaptive output the knots are Vern9's own steps: irregular,
# and PLACED BY THE DYNAMICS — the knot-spacing zoom figure shows the spacing is
# phase-locked to the laser period and stratified by initial radius. Each electron's
# between-knot interpolation error is then percent-scale (relative to its own acceleration)
# and carries both phase-locked distortion lines (odd harmonics of its ω tone, for the
# circular-polarization campaigns) and broadband knot noise. What the ensemble sum does to
# those errors is the crux (a₀=1e-5 floor_probe evidence, δaˣ and δaᶻ):
#   - transverse: the per-electron 3ω/5ω lines carry each electron's azimuthal phase, so
#     the vortex winding cancels them like any incoherent term; the broadband noise only
#     interferes destructively down to the 1/√N shot level. The surviving residue is flat
#     across the spectrum — the 2ω "line" the h2[B] maps read is just that noise floor at
#     the 2ω bin (hence the speckle h2 maps and the run-to-run 1.05–2.9e-6 scatter).
#   - longitudinal: the |E|²-driven quiver is azimuth-independent, so part of the spline
#     error is phase-locked ACROSS electrons — at 2ω the coherent mean survives a factor
#     ~3–8 above the 1/√N expectation and buries the physical 2ω of aᶻ by ~2.5 decades.
# Uniform knots (EDM_INTERP_SAVEAT) break the dynamics lock: the error is tied to a fixed
# grid instead, drops ~3 decades, and its coherent 2ω residue collapses (~6 decades).
#
# This mode measures exactly that error, per electron and per instant, against the dense
# Vern9 continuous extension (the "truth" the splines are built from), for three variants:
#
#   adaptive       TrajectoryInterpolant(dense solve) — the production path without
#                  EDM_INTERP_SAVEAT (knots = Vern9's own steps).
#   uniform prod   TrajectoryInterpolant(saveat solve) — the production path WITH
#                  EDM_INTERP_SAVEAT = ERR_SAVEAT knots/period. NB on a saveat (non-dense)
#                  solution `sol(t, Val{1})` — the a_itp knot source — falls back to the
#                  local linear-segment slope: a smooth ≈ωh/2 phase/amplitude bias common to
#                  every electron, which mirrors the physical lines in the error spectrum
#                  but creates no NEW line (and largely cancels in the h2/h1 ratio). Shown
#                  for honesty and quantified by the aknot_rms_rel_prod diagnostic.
#   uniform ideal  CubicSplines through the SAME uniform knot times but with state values
#                  AND acceleration knots sampled from the dense solution — isolates knot
#                  PLACEMENT (uniformity) from knot quality; the apples-to-apples control
#                  for the adaptive variant.
#
# Per source run it writes four PNGs + a run_<uuid>.toml manifest:
#   splerr_time_*   full-record error envelopes (log) + a zoomed window of the individual
#                   per-electron errors and of the surviving ensemble mean vs the ±rms/√N
#                   incoherent band
#   splerr_heat_*   electron × time error map (adaptive): per-electron knot stripes differ,
#                   common vertical banding = the coherent residual
#   splerr_spec_*   error spectra: coherent |FFT⟨δa⟩| vs the incoherent expectation
#                   mean|FFT δa|/√N and the physical spectrum; the coherent adaptive value
#                   at 2ω over the physical fundamental is the predicted floor
#   splerr_knots_*  adaptive knot spacing vs time — why the errors phase-lock
# ════════════════════════════════════════════════════════════════════════════════════════

const VARIANTS = (:adaptive, :uniform_prod, :uniform_ideal)

poslog(x) = max.(x, 1e-300)   # keep log-axis data strictly positive

# Per-period max of |sig| (the fine grid has exactly spp samples/period) → log-plot envelope.
period_env(sig, spp) = [maximum(abs, view(sig, ((k - 1) * spp + 1):(k * spp)))
                        for k in 1:(length(sig) ÷ spp)]

# Peak of a spectrum within a band of harmonic orders (robust to bin placement and leakage).
# NB max over a band of noisy bins carries Rayleigh max-statistics bias (~3× the mean level
# for ±0.25 order at these record lengths) — the same bias any h2-bin readout of a noise
# floor has. band_median gives the unbiased local broadband level; a narrow band (half≈0.02)
# isolates an actual phase-locked line.
function band_max(S, orders, center; half = 0.25)
    lo = searchsortedfirst(orders, center - half)
    hi = searchsortedlast(orders, center + half)
    return maximum(view(S, lo:hi))
end
function band_median(S, orders, center; half = 0.25)
    lo = searchsortedfirst(orders, center - half)
    hi = searchsortedlast(orders, center + half)
    return median(view(S, lo:hi))
end

# Zero-phase band-limited component: keep only the rfft bins with harmonic order in
# center ± half. Deliberately unwindowed — the signals here are compact (pulse ≪ record) so
# band leakage is negligible; do NOT reuse for signals that fill the record.
function bandpass(sig, orders, center; half = 0.25)
    F = rfft(sig)
    keep = (orders .>= center - half) .& (orders .<= center + half)
    F[.!keep] .= 0
    return irfft(F, length(sig))
end

# Compact JSON payload: 5 significant digits ≫ anything the report figures resolve.
rd(x::Real) = round(x; sigdigits = 5)
rd(v::AbstractArray) = rd.(v)

# CubicSplines through uniform knot times with state values AND acceleration knots sampled
# from the DENSE solution's continuous extension — the knot-placement-only "uniform ideal"
# variant (same structure radiation.jl builds, ideal knot values).
function ideal_uniform_interpolant(sol, ts, x_idxs, u_idxs, K)
    itp = CubicSpline(sol(ts).u, ts; extrapolation = ExtrapolationType.Extension)
    a_knots = [SVector{4}(v[u_idxs]) for v in sol(ts, Val{1}).u]
    a_itp = CubicSpline(a_knots, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, x_idxs, u_idxs, K)
end

# ── FIG 1: error worldlines — envelopes (full record) + zoomed traces + surviving mean ──
function splerr_time_figure(; tsT, zoom, pcT, spp, A0, arms, μ, rms, traces, tracecol, crange, N, title)
    fig = Figure(size = (1200, 940))
    Label(fig[0, 1:2], title; fontsize = 15, font = :bold)
    cols = ((:adaptive, "adaptive knots (production default)"),
        (:uniform_ideal, @sprintf("uniform knots (ideal, %d/period)", ERR_SAVEAT)))
    ax_env = Axis[]; ax_ind = Axis[]; ax_mean = Axis[]
    for (j, (v, name)) in enumerate(cols)
        # row 1: full-record per-period envelopes (log)
        ax1 = Axis(fig[1, j]; title = name, yscale = log10,
            ylabel = j == 1 ? L"|\delta a| / a_\mathrm{peak}" : "")
        lines!(ax1, pcT, poslog(period_env(arms, spp) ./ A0);
            color = (:gray, 0.5), label = L"physical rms $|a|$")
        lines!(ax1, pcT, poslog(period_env(rms[v], spp) ./ A0);
            color = :steelblue, label = "rms over electrons")
        lines!(ax1, pcT, poslog(period_env(μ[v], spp) ./ A0);
            color = :crimson, label = "ensemble mean (coherent)")
        lines!(ax1, pcT, poslog(period_env(rms[v], spp) ./ (A0 * sqrt(N)));
            color = :black, linestyle = :dot, label = L"rms$/\sqrt{N}$ (incoherent expectation)")
        ylims!(ax1, 1e-14, 30)   # clip the pulse-edge tails (~1e-30): keep the informative decades
        j == 1 && axislegend(ax1; position = :lb, framevisible = false, labelsize = 9)
        push!(ax_env, ax1)
        # row 2: individual per-electron errors, zoomed
        ax2 = Axis(fig[2, j]; ylabel = j == 1 ? L"\delta a / a_\mathrm{peak}" : "")
        for (i, δ) in traces[v]
            lines!(ax2, tsT[zoom], δ[zoom] ./ A0; color = (tracecol(i), 0.55), linewidth = 0.8)
        end
        lines!(ax2, tsT[zoom], μ[v][zoom] ./ A0; color = :black, linewidth = 1.4)
        push!(ax_ind, ax2)
        # row 3: the surviving ensemble mean alone vs the ±rms/√N incoherent band
        ax3 = Axis(fig[3, j]; xlabel = L"\tau/T",
            ylabel = j == 1 ? L"\langle\delta a\rangle / a_\mathrm{peak}" : "")
        band!(ax3, tsT[zoom], -rms[v][zoom] ./ (A0 * sqrt(N)), rms[v][zoom] ./ (A0 * sqrt(N));
            color = (:gray, 0.35), label = L"\pm\,\mathrm{rms}/\sqrt{N}")
        lines!(ax3, tsT[zoom], μ[v][zoom] ./ A0;
            color = :crimson, linewidth = 1.2, label = "ensemble mean")
        j == 1 && axislegend(ax3; position = :lt, framevisible = false, labelsize = 9)
        push!(ax_mean, ax3)
        linkxaxes!(ax2, ax3)
        hidexdecorations!(ax2; grid = false)
    end
    linkyaxes!(ax_env...); linkyaxes!(ax_ind...); linkyaxes!(ax_mean...)
    Colorbar(fig[4, 1:2]; colormap = :viridis, colorrange = crange,
        vertical = false, flipaxis = false, height = 10, label = L"r_0/w_0")
    rowgap!(fig.layout, 8)
    return fig
end

# ── FIG 2: electron × time error map (adaptive) + the shared (coherent) column mean ──
function splerr_heat_figure(; tsT, zoom, E, A0, N, title)
    fig = Figure(size = (960, 700))
    Label(fig[0, 1:2], title; fontsize = 15, font = :bold)
    M = E[zoom, :] ./ A0
    cmax = quantile(abs.(vec(M)), 0.995)
    ax = Axis(fig[1, 1]; ylabel = "electron # (radius-ordered)")
    hm = heatmap!(ax, tsT[zoom], 1:N, M; colormap = :RdBu, colorrange = (-cmax, cmax))
    Colorbar(fig[1, 2], hm; label = L"\delta a / a_\mathrm{peak}")
    ax2 = Axis(fig[2, 1]; xlabel = L"\tau/T", ylabel = "mean")
    lines!(ax2, tsT[zoom], vec(mean(M; dims = 2)); color = :crimson)
    linkxaxes!(ax, ax2)
    hidexdecorations!(ax; grid = false)
    rowsize!(fig.layout, 2, Relative(0.16))
    rowgap!(fig.layout, 6)
    return fig
end

# ── FIG 3: error spectra — what survives the ensemble sum, and where ──
function splerr_spec_figure(; orders, Sphys, Scoh, Sinc, Sω, N, floor2, phys2, title)
    fig = Figure(size = (1200, 560))
    Label(fig[0, 1:2], title; fontsize = 15, font = :bold)
    sel = 2:searchsortedlast(orders, 6.0)   # skip DC, show up to the 6th harmonic
    cols = ((:adaptive, "adaptive knots (production default)"),
        (:uniform_ideal, @sprintf("uniform knots (ideal, %d/period)", ERR_SAVEAT)))
    axs = Axis[]
    for (j, (v, name)) in enumerate(cols)
        ax = Axis(fig[1, j]; title = name, xlabel = L"harmonic order $f/\omega$",
            yscale = log10, ylabel = j == 1 ? L"|a(f)| / |a_\mathrm{phys}(\omega)|" : "")
        vlines!(ax, [1.0, 2.0]; color = (:black, 0.12))
        lines!(ax, orders[sel], poslog(Sphys[sel] ./ Sω);
            color = (:gray, 0.55), label = @sprintf("physical ⟨a⟩  (2ω: %.1e)", phys2))
        lines!(ax, orders[sel], poslog(Sinc[v][sel] ./ (Sω * sqrt(N)));
            color = :steelblue, linestyle = :dash,
            label = L"incoherent expectation $\mathrm{mean}|\delta a(f)| / \sqrt{N}$")
        lines!(ax, orders[sel], poslog(Scoh[v][sel] ./ Sω);
            color = :crimson, label = @sprintf("coherent ⟨δa⟩  (2ω: %.1e)", floor2[v]))
        if v == :uniform_ideal
            lines!(ax, orders[sel], poslog(Scoh[:uniform_prod][sel] ./ Sω);
                color = :seagreen, linestyle = :dot,
                label = @sprintf("production saveat path  (2ω: %.1e)", floor2[:uniform_prod]))
        end
        axislegend(ax; position = :rt, framevisible = false, labelsize = 9)
        push!(axs, ax)
    end
    linkyaxes!(axs...)
    return fig
end

# ── FIG 4: adaptive knot spacing vs time — the step-size modulation that phase-locks errors ──
function splerr_knot_figure(; sols, idxs, T, zoomT, tracecol, crange, title)
    fig = Figure(size = (1100, 540))
    Label(fig[0, 1:2], title; fontsize = 15, font = :bold)
    # Full record on log y (bulk spacing ~0.2T vs ~20T ramps where the pulse is off); the zoom
    # panel autoscales linearly around the in-pulse values to expose per-period modulation.
    ax1 = Axis(fig[1, 1]; xlabel = L"\tau/T", ylabel = L"\Delta\tau_\mathrm{knot}/T",
        title = "full record", yscale = log10)
    ax2 = Axis(fig[1, 2]; xlabel = L"\tau/T", title = @sprintf("zoom (%g periods)", ERR_ZOOM))
    for i in idxs
        τk = sols[i].t
        mid = (τk[1:(end - 1)] .+ τk[2:end]) ./ (2T)
        dτ = diff(τk) ./ T
        lines!(ax1, mid, dτ; color = (tracecol(i), 0.45), linewidth = 0.7)
        insel = findall(m -> zoomT[1] ≤ m ≤ zoomT[2], mid)
        isempty(insel) || scatterlines!(ax2, mid[insel], dτ[insel];
            color = tracecol(i), markersize = 5, linewidth = 0.9)
    end
    for ax in (ax1, ax2)
        hlines!(ax, [1 / ERR_SAVEAT]; color = :black, linestyle = :dash)
    end
    Colorbar(fig[2, 1:2]; colormap = :viridis, colorrange = crange,
        vertical = false, flipaxis = false, height = 10, label = L"r_0/w_0")
    return fig
end

# ── FIG 5 (with EDM_ERR_EXPORT): individual splines under the microscope ──
# Per export electron: left, the adaptive error vs the uniform-ideal control (comparable
# magnitudes — the knot-PLACEMENT comparison) with the adaptive knots marked; right, the
# production-saveat error at its own scale (its linear-slope a-knot bias is ~decades larger).
# Both columns carry a faint a(t) scaled into the error band for phase reference. The static
# counterpart of the dashboard report's interactive explorer.
function splerr_zoomcase_figure(; ts, T, ipk, ids, r0w, A0, A_tr, E_adap, E_prod_tr,
        E_ideal_tr, sols, title)
    fig = Figure(size = (1250, 185 * length(ids) + 90))
    Label(fig[0, 1:2], title; fontsize = 15, font = :bold)
    w = searchsortedfirst(ts, ts[ipk] - 2T):searchsortedlast(ts, ts[ipk] + 2T)
    tsT = ts[w] ./ T
    aref(sig, δmax) = sig .* (0.9 * δmax / max(maximum(abs, sig), 1e-300))
    nid = length(ids)
    axl = Axis[]; axr = Axis[]
    for (k, i) in enumerate(ids)
        δa = view(E_adap, w, i) ./ A0
        δi = E_ideal_tr[i][w] ./ A0
        δp = E_prod_tr[i][w] ./ A0
        at = view(A_tr[i], w)
        ax1 = Axis(fig[k, 1]; ylabel = L"\delta a / a_\mathrm{peak}", titlesize = 11,
            titlealign = :left, xlabel = k == nid ? L"\tau/T" : "",
            title = k == 1 ? @sprintf("adaptive vs uniform ideal — electron %d (r₀/w₀ = %.2f)", i, r0w[i]) :
                @sprintf("electron %d (r₀/w₀ = %.2f)", i, r0w[i]))
        τk = filter(t -> ts[w[1]] <= t <= ts[w[end]], sols[i].t) ./ T
        vlines!(ax1, τk; color = (:crimson, 0.18), linewidth = 0.6)
        lines!(ax1, tsT, aref(at, maximum(abs, δa)); color = (:gray, 0.35), label = "a(t) (scaled)")
        lines!(ax1, tsT, δa; color = :crimson, linewidth = 1.0, label = "adaptive")
        lines!(ax1, tsT, δi; color = :steelblue, linewidth = 1.0,
            label = @sprintf("uniform ideal (%d/period)", ERR_SAVEAT))
        ax2 = Axis(fig[k, 2]; titlesize = 11, titlealign = :left,
            xlabel = k == nid ? L"\tau/T" : "",
            title = k == 1 ? "production saveat (own scale)" : "")
        lines!(ax2, tsT, aref(at, maximum(abs, δp)); color = (:gray, 0.35))
        lines!(ax2, tsT, δp; color = :seagreen, linewidth = 1.0,
            label = @sprintf("uniform saveat, production (%d/period)", ERR_SAVEAT))
        k == 1 && axislegend(ax1; position = :rt, framevisible = false, labelsize = 8)
        k == 1 && axislegend(ax2; position = :rt, framevisible = false, labelsize = 8)
        k < nid && (hidexdecorations!(ax1; grid = false); hidexdecorations!(ax2; grid = false))
        push!(axl, ax1); push!(axr, ax2)
    end
    linkxaxes!(axl..., axr...)
    return fig
end

# ── Report-payload export (EDM_ERR_EXPORT): everything the interactive dashboard report
# needs, serialized once so the report never recomputes physics. Trace arrays are
# pre-normalized (time domain ÷ A0, spectra ÷ Sω) and rounded to 5 sigdigits; uniform time
# grids are stored as (t0T, dtT/dT) pairs, never as arrays. Knot times keep 6 decimals of
# τ/T so consecutive differences (the spacing figures) stay clean.
function write_splerr_export(fexp; component, config, laser, setup, run_id, T, τi, spp,
        np_tot, orders, N, arms, μ, rms, μ2, ā2, Sphys, Scoh, Sinc, Sω, A0,
        export_ids, r0, Rmax, w₀, A_tr, E_adap, E_prod_tr, E_ideal_tr, sols, saveat, ts, wexp)
    kT(t) = round(t / T; digits = 6)
    env(sig) = rd(period_env(sig, spp) ./ A0)   # period_env is already |·|-max per period
    sel = 1:searchsortedlast(orders, 6.5)       # DC … 6.5ω — all the report's spectra show
    electrons = map(export_ids) do i
        (; i, r0_w0 = rd(r0[i] / w₀), r0_Rmax = rd(r0[i] / Rmax),
            a = rd(A_tr[i][wexp] ./ A0),
            err = (; adaptive = rd(E_adap[wexp, i] ./ A0),
                uniform_prod = rd(E_prod_tr[i][wexp] ./ A0),
                uniform_ideal = rd(E_ideal_tr[i][wexp] ./ A0)),
            knots_adaptive = kT.(filter(t -> ts[first(wexp)] <= t <= ts[last(wexp)], sols[i].t)),
            knots_adaptive_full = kT.(sols[i].t))
    end
    payload = (;
        schema = 1, kind = "splerr_report", component, run_id, T, spp, N,
        config, laser, setup,
        norm = (; A0 = round(A0; sigdigits = 8), S_omega = round(Sω; sigdigits = 8)),
        period_env = (;
            np = np_tot, t0T = kT(τi + T / 2), dT = 1.0,
            arms = env(arms),
            mu = (; (v => env(μ[v]) for v in VARIANTS)...),
            rms = (; (v => env(rms[v]) for v in VARIANTS)...),
            mu2w = (; (v => env(μ2[v]) for v in VARIANTS)...),
            phys2w = env(ā2)),
        spectrum = (;
            dorder = orders[2] - orders[1],
            phys = rd(Sphys[sel] ./ Sω),
            coh = (; (v => rd(Scoh[v][sel] ./ Sω) for v in VARIANTS)...),
            inc = (; (v => rd(Sinc[v][sel] ./ Sω) for v in VARIANTS)...)),
        zoom = (;
            t0T = kT(ts[first(wexp)]), dtT = 1 / spp,
            electrons,
            uniform_knots = (; t0T = kT(saveat[1]), dT = 1 / ERR_SAVEAT)),
        band2w = (;
            t0T = kT(ts[first(wexp)]), dtT = 1 / spp,
            phys = rd(ā2[wexp] ./ A0),
            mu = (; (v => rd(μ2[v][wexp] ./ A0) for v in VARIANTS)...)),
    )
    open(io -> JSON.json(io, payload), fexp, "w")
    return fexp
end

# ── Spline-error analysis of one source run ──
function analyze_spline_error(mfile)
    rec = reconstruct(mfile)
    (; λ, w₀, m_az, p_rad, pol, profile, τ0, z_focus,
        a₀, N, Rmax, τi, τf, φ₀, parent_run_id, reltol, abstol, ensemble, r0) = rec
    isempty(ERR_RELTOL) || (reltol = parse(Float64, ERR_RELTOL))
    isempty(ERR_ABSTOL) || (abstol = parse(Float64, ERR_ABSTOL))
    T = λ / c   # laser period; τ ≈ t at these a₀ (γ − 1 ~ a₀²), so harmonics read directly
    comp = findfirst(==(ERR_COMPONENT), ("x", "y", "z"))

    @info "spline-error run" file = basename(mfile) a₀ φ₀ N reltol abstol parent_run_id

    # The two production solves: identical adaptive stepping, only the SAVED output differs
    # (saveat is filled from the in-step interpolant; it does not alter the step sequence).
    saveat = collect(τi:(T / ERR_SAVEAT):τf)
    t_solve = @elapsed begin
        sol_dense = solve(ensemble, Vern9(), EnsembleThreads();
            reltol, abstol, trajectories = N)
        sol_unif = solve(ensemble, Vern9(), EnsembleThreads();
            reltol, abstol, trajectories = N, saveat)
    end
    sols = sol_dense.u
    kpp = mean(length(s.t) - 1 for s in sols) / ((τf - τi) / T)
    @info "trajectories solved (dense + saveat)" t_solve knots_per_period_adaptive = round(kpp; digits = 2)

    # Production interpolants, exactly as accumulate_field receives them (radiation.jl).
    trajs_adap = trajectory_interpolants(sol_dense)
    trajs_prod = trajectory_interpolants(sol_unif)
    x_idxs = trajs_adap[1].x_idxs
    u_idxs = trajs_adap[1].u_idxs
    K = trajs_adap[1].K
    ai = u_idxs[1 + comp]   # state index of uᶜ ⇒ that component of du = 𝔞ᶜ

    # Fine uniform grid: ERR_SPP samples/period over whole periods (clean envelopes/FFT bins).
    np_tot = floor(Int, (τf - τi) / T)
    nt = np_tot * ERR_SPP
    ts = collect(τi .+ (0:(nt - 1)) .* (T / ERR_SPP))
    win = 0.5 .- 0.5 .* cos.(2π .* (0:(nt - 1)) ./ (nt - 1))   # Hann
    nf = nt ÷ 2 + 1
    orders = (0:(nf - 1)) .* (ERR_SPP / nt)   # rfft bin → harmonic order f/ω

    trace_list = unique(round.(Int, range(1, N; length = min(ERR_N_TRACE, N))))
    # Electrons whose full-resolution traces are exported for the interactive report:
    # radius-uniform over the disk (index-uniform would cluster at the rim, r ∝ √k).
    export_ids = ERR_EXPORT == "" ? Int[] :
        !isempty(ERR_EXPORT_IDS) ?
        sort(unique(clamp.(parse.(Int, split(ERR_EXPORT_IDS, ",")), 1, N))) :
        unique([argmin(abs.(r0 ./ Rmax .- s)) for s in (0.0, 1 / 3, 2 / 3, 1.0)])
    export_set = Set(export_ids)
    Σa = zeros(nt)
    Σa² = zeros(nt)
    Σδ = Dict(v => zeros(nt) for v in VARIANTS)
    Σδ² = Dict(v => zeros(nt) for v in VARIANTS)
    ΣS = Dict(v => zeros(nf) for v in VARIANTS)
    E_adap = Matrix{Float64}(undef, nt, N)      # full adaptive error (heatmap + traces)
    E_ideal_tr = Dict{Int, Vector{Float64}}()   # uniform-ideal traces (subset only)
    A_tr = Dict{Int, Vector{Float64}}()         # dense-truth acceleration (export subset)
    E_prod_tr = Dict{Int, Vector{Float64}}()    # uniform-prod traces (export subset)

    t_err = @elapsed for i in 1:N
        sol = sols[i]
        atrue = [v[ai] for v in sol(ts, Val{1}).u]   # dense Vern9 derivative = truth
        Σa .+= atrue
        Σa² .+= abs2.(atrue)
        i in export_set && (A_tr[i] = atrue)
        traj_ideal = ideal_uniform_interpolant(sol, saveat, x_idxs, u_idxs, K)
        for (v, traj) in ((:adaptive, trajs_adap[i]), (:uniform_prod, trajs_prod[i]),
                (:uniform_ideal, traj_ideal))
            δ = [traj.a_itp(t)[1 + comp] for t in ts] .- atrue
            Σδ[v] .+= δ
            Σδ²[v] .+= abs2.(δ)
            ΣS[v] .+= abs.(rfft(win .* δ))
            v == :adaptive && (E_adap[:, i] = δ)
            v == :uniform_prod && i in export_set && (E_prod_tr[i] = δ)
            v == :uniform_ideal && (i in trace_list || i in export_set) && (E_ideal_tr[i] = δ)
        end
    end
    @info "spline errors measured" t_err nt

    # Two physical yardsticks: ā (coherent ensemble mean — what the field sum sees; the vortex
    # phase winding cancels most of it) and arms (typical single-electron scale — what a
    # per-electron relative error should be read against). Time-domain/heatmap figures are
    # normalized by A0 = peak(arms); spectra by the coherent fundamental Sω (the h1 analog).
    ā = Σa ./ N
    arms = sqrt.(Σa² ./ N)
    A0 = maximum(arms)                                     # peak typical-electron acceleration
    A0coh = maximum(abs, ā)                                # peak coherent-mean acceleration
    μ = Dict(v => Σδ[v] ./ N for v in VARIANTS)            # coherent (ensemble-mean) error
    rms = Dict(v => sqrt.(Σδ²[v] ./ N) for v in VARIANTS)  # typical single-electron error
    Sphys = abs.(rfft(win .* ā))
    Scoh = Dict(v => abs.(rfft(win .* μ[v])) for v in VARIANTS)
    Sinc = Dict(v => ΣS[v] ./ N for v in VARIANTS)
    Sω = band_max(Sphys, orders, 1.0)          # physical fundamental (h1 proxy)
    phys2 = band_max(Sphys, orders, 2.0) / Sω  # physical 2ω (the signal the floor buries)
    # h2-bin readout analog (max over 2±0.25 — carries the same max-statistics bias as any
    # harmonic-map bin readout of a noise floor), plus the discriminating pair: the unbiased
    # local broadband level and a narrow line window at exactly 2ω.
    floor2 = Dict(v => band_max(Scoh[v], orders, 2.0) / Sω for v in VARIANTS)
    broad2 = Dict(v => band_median(Scoh[v], orders, 2.0) / Sω for v in VARIANTS)
    line2 = Dict(v => band_max(Scoh[v], orders, 2.0; half = 0.02) / Sω for v in VARIANTS)
    inc2 = Dict(v => band_max(Sinc[v], orders, 2.0) / (Sω * sqrt(N)) for v in VARIANTS)
    broad2_inc = band_median(Sinc[:adaptive], orders, 2.0) / (Sω * sqrt(N))
    coh2 = broad2[:adaptive] / broad2_inc   # ≈1 ⇒ incoherent residue; ≫1 ⇒ phase-locked line
    @info "2ω budget (acceleration proxy of h2/h1)" phys2 floor2_adaptive = floor2[:adaptive] broad2_adaptive = broad2[:adaptive] line2_adaptive = line2[:adaptive] inc2_adaptive = inc2[:adaptive] broad2_coherence = round(coh2; digits = 2) floor2_uniform_ideal = floor2[:uniform_ideal] floor2_uniform_prod = floor2[:uniform_prod] vortex_cancellation = round(A0coh / A0; digits = 4)

    # What the production saveat path actually feeds the acceleration spline: on a non-dense
    # (saveat) solution sol(t, Val{1}) is only the local linear-segment slope. Quantify the
    # knot-value deviation from the dense-truth derivative at the same times (electron 1).
    kts = sol_unif.u[1].t
    aknot_prod = [sol_unif.u[1](t, Val{1})[ai] for t in kts]
    aknot_true = [sols[1](t, Val{1})[ai] for t in kts]
    aknot_rel = sqrt(sum(abs2, aknot_prod .- aknot_true) / max(sum(abs2, aknot_true), eps()))
    @info "production saveat a_itp knot deviation (electron 1)" aknot_rms_rel = aknot_rel

    # Manifest dicts (built early: the report export embeds the same metrics verbatim).
    run_id = string(uuid4())
    config = Dict(
        "kind" => "spline_error", "a0" => a₀, "initial_phase" => φ₀, "N" => N,
        "component" => ERR_COMPONENT, "err_spp" => ERR_SPP, "err_saveat" => ERR_SAVEAT,
        "zoom_periods" => ERR_ZOOM, "reltol" => reltol, "abstol" => abstol,
        "knots_per_period_adaptive" => kpp,
        "phys_h2h1_acc" => phys2,
        "floor2_adaptive" => floor2[:adaptive],
        "floor2_uniform_ideal" => floor2[:uniform_ideal],
        "floor2_uniform_prod" => floor2[:uniform_prod],
        "broad2_adaptive" => broad2[:adaptive],
        "broad2_uniform_ideal" => broad2[:uniform_ideal],
        "broad2_uniform_prod" => broad2[:uniform_prod],
        "line2_adaptive" => line2[:adaptive],
        "line2_uniform_ideal" => line2[:uniform_ideal],
        "line2_uniform_prod" => line2[:uniform_prod],
        "inc2_adaptive" => inc2[:adaptive],
        "broad2_coherence_adaptive" => coh2,
        "a_peak_rms" => A0,
        "a_peak_coherent" => A0coh,
        "aknot_rms_rel_prod" => aknot_rel,
        "source_campaign" => basename(rstrip(SOURCE_CAMPAIGN, '/')), "source_run" => parent_run_id,
    )
    laser_out = Dict(
        "wavelength" => λ, "a0" => a₀, "w0" => w₀, "m" => m_az, "p" => p_rad,
        "pol" => string(pol), "profile" => string(profile),
        "temporal_width" => τ0, "focus_position" => z_focus, "phi0" => φ₀,
    )
    setup_out = Dict("Rmax" => Rmax, "τi" => τi, "τf" => τf)

    # Zoom window (ERR_ZOOM periods around the envelope peak) + plotting helpers.
    ipk = argmax(abs.(ā))
    zoom = searchsortedfirst(ts, ts[ipk] - ERR_ZOOM * T / 2):searchsortedlast(ts, ts[ipk] + ERR_ZOOM * T / 2)
    tsT = ts ./ T
    pcT = (τi .+ ((1:np_tot) .- 0.5) .* T) ./ T   # per-period envelope bin centres
    r0w = r0 ./ w₀
    crange = (0.0, Rmax / w₀)
    cg = cgrad(:viridis)
    tracecol = i -> cg[clamp((r0w[i] - crange[1]) / (crange[2] - crange[1]), 0.0, 1.0)]
    traces = Dict(:adaptive => [(i, E_adap[:, i]) for i in trace_list],
        :uniform_ideal => [(i, E_ideal_tr[i]) for i in trace_list])

    philab = phi_label(φ₀)
    pstr = @sprintf("a₀=%.0e, φ₀=%s, N=%d, δa^%s", a₀, philab, N, ERR_COMPONENT)
    base = @sprintf("a0_%.0e_phi%.4f", a₀, φ₀)
    id8 = first(string(uuid4()), 8)
    mkpath(OUTDIR)

    # Report-payload export (+ the zoomcase figure below): full-res window around the peak.
    exportfile = nothing
    if ERR_EXPORT != ""
        wexp = searchsortedfirst(ts, ts[ipk] - ERR_EXPORT_PERIODS * T / 2):searchsortedlast(ts, ts[ipk] + ERR_EXPORT_PERIODS * T / 2)
        μ2 = Dict(v => bandpass(μ[v], orders, 2.0) for v in VARIANTS)
        ā2 = bandpass(ā, orders, 2.0)
        fexp = ERR_EXPORT == "1" ?
            joinpath(OUTDIR, "splerr_report_$(ERR_COMPONENT)_$(base)_$(id8).json") :
            isabspath(ERR_EXPORT) ? ERR_EXPORT : joinpath(OUTDIR, ERR_EXPORT)
        write_splerr_export(fexp; component = ERR_COMPONENT, config, laser = laser_out,
            setup = setup_out, run_id, T, τi, spp = ERR_SPP, np_tot, orders, N, arms, μ, rms,
            μ2, ā2, Sphys, Scoh, Sinc, Sω, A0, export_ids, r0, Rmax, w₀, A_tr, E_adap,
            E_prod_tr, E_ideal_tr, sols, saveat, ts, wexp)
        exportfile = basename(fexp)
        println("export → $fexp")
    end

    figs = Any[
        ("time", splerr_time_figure(; tsT, zoom, pcT, spp = ERR_SPP, A0, arms, μ, rms, traces,
            tracecol, crange, N,
            title = "Trajectory-spline error vs dense Vern9 — " * pstr)),
        ("heat", splerr_heat_figure(; tsT, zoom, E = E_adap, A0, N,
            title = "Per-electron spline error, adaptive knots — " * pstr)),
        ("spec", splerr_spec_figure(; orders, Sphys, Scoh, Sinc, Sω, N, floor2, phys2,
            title = "Spline-error spectra: coherent survival at 2ω — " * pstr)),
        ("knots", splerr_knot_figure(; sols, idxs = trace_list, T,
            zoomT = (tsT[first(zoom)], tsT[last(zoom)]), tracecol, crange,
            title = "Adaptive knot spacing — " * pstr)),
    ]
    ERR_EXPORT == "" || push!(figs,
        ("zoomcase", splerr_zoomcase_figure(; ts, T, ipk, ids = export_ids, r0w, A0,
            A_tr, E_adap, E_prod_tr, E_ideal_tr, sols,
            title = "Individual splines under the microscope — " * pstr)))
    plots = String[]
    for (tag, fig) in figs
        fname = "splerr_$(tag)_$(base)_$(id8).png"
        save(joinpath(OUTDIR, fname), fig)
        push!(plots, fname)
        println("saved → $(joinpath(OUTDIR, fname))")
    end

    write_run_manifest(OUTDIR; run_id, script = basename(PROGRAM_FILE),
        config, laser = laser_out, setup = setup_out, datafile = exportfile, plots)
    println("manifest → $(joinpath(OUTDIR, "run_$(run_id).toml"))")
    return run_id
end

# ── Drive over the source campaign ──
isdir(SOURCE_CAMPAIGN) || error("EDM_SOURCE_CAMPAIGN is not a directory: $SOURCE_CAMPAIGN")
manifests = sort([joinpath(SOURCE_CAMPAIGN, f) for f in readdir(SOURCE_CAMPAIGN)
                  if startswith(f, "run_") && endswith(f, ".toml") && contains(f, SOURCE_RUN)])
isempty(manifests) && error("no matching run_*.toml in $SOURCE_CAMPAIGN (EDM_SOURCE_RUN=\"$SOURCE_RUN\")")
if ANALYSIS == "spline-error"
    @info "spline-error analysis" SOURCE_CAMPAIGN SOURCE_RUN OUTDIR runs = length(manifests) PHASE_SPEC ERR_SPP ERR_SAVEAT ERR_COMPONENT ERR_N_TRACE ERR_ZOOM
else
    @info "trajectory analysis" SOURCE_CAMPAIGN SOURCE_RUN OUTDIR runs = length(manifests) PHASE_SPEC COORDS N_PLOT_TRAJ NPATH
end
mkpath(OUTDIR)
for mfile in manifests
    ANALYSIS == "spline-error" ? analyze_spline_error(mfile) : analyze_run(mfile)
end
println("done — $(length(manifests)) run(s) → $OUTDIR")
