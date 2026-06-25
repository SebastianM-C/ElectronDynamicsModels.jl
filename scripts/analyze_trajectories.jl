# Electron-trajectory analysis for the φ₀=−π/2 "cm" campaign.
#
# Same laser + sunflower electron distribution as scripts/thomson_scattering.jl, but instead
# of accumulating the radiated field this re-solves the (cheap) trajectory ensemble on CPU and
# visualizes the worldlines. Parameters are read from a source campaign's run_*.toml manifests
# (e.g. field_campaign_cm_899976, the a₀ scan) with the initial phase OVERRIDDEN to −π/2 — the
# φ₀=−π/2 runs were produced elsewhere and aren't stored, but trajectories are cheap to recompute,
# so only the run parameters are needed, not any saved field/trajectory data.
#
# Per source run (= per a₀) it writes TWO PNGs — the worldlines projected onto the zx and zy
# planes (axes in units of w₀), each with marginal histograms of the FINAL electron positions
# (at τf) — plus a run_<uuid>.toml manifest, into the runs dir.
#
# Worldlines are sampled with the solution's DENSE interpolation via symbolic-expression indexing
# — `sol(ts; idxs = sys.x[4] / w₀)` — the same `idxs` machinery SciMLBase's Makie plot recipe uses
# internally (we call it directly so each line can be coloured by its initial radius r₀/w₀, which
# the recipe's label-only PlotSpecs can't carry). The w₀ scaling lives inside the symbolic getter.
#
# Coordinate mode (EDM_COORDS): the transverse motion here is a real ~0.008·w₀ ponderomotive
# displacement — larger than the ~0.001·w₀ forward z-drift — but it's only ~0.25% of the ±3.25·w₀
# initial disk, so on ABSOLUTE transverse axes it looks frozen. "displacement" plots (x−x₀, y−y₀)
# vs z, centring every electron, so that motion is visible; "absolute" keeps beam-frame positions.
#
# ENV knobs (all optional):
#   EDM_SOURCE_CAMPAIGN  dir of run_*.toml to take parameters from
#                        (default /storage/pool/smc/thomson_runs/field_campaign_cm_899976)
#   EDM_INITIAL_PHASE    φ₀ override applied to every run (default -π/2); "manifest" ⇒ use each
#                        manifest's own [config].initial_phase
#   EDM_COORDS           "absolute" (default) or "displacement" (plot x−x₀, y−y₀ vs z)
#   EDM_OUTDIR           output dir (default <pkg>/runs/field_campaign_cm_phi0)
#   EDM_N                cap electrons per run (default = manifest N; the cap takes the inner
#                        sunflower disk, so use the full N for production figures)
#   EDM_N_PLOT_TRAJ      worldlines actually drawn (default 800; histograms always use all N)
#   EDM_NPATH            dense τ samples per drawn worldline (default 10000 displacement / 1500 absolute)
#
#   julia --project=scripts scripts/analyze_trajectories.jl

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner   # explicit Vern9 — pure ODE, no implicit-init solver needed
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using Printf
using CairoMakie
using TOML
using UUIDs

include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts
include(joinpath(@__DIR__, "manifest.jl"))      # RunManifests: write_run_manifest

# Atomic units
const c = 137.03599908330932

# ── Configuration (ENV-overridable) ──
const SOURCE_CAMPAIGN = get(ENV, "EDM_SOURCE_CAMPAIGN", "/storage/pool/smc/thomson_runs/field_campaign_cm_899976")
const PHASE_SPEC = get(ENV, "EDM_INITIAL_PHASE", string(-π / 2))   # numeric override, or "manifest"
const COORDS = get(ENV, "EDM_COORDS", "absolute")                  # "absolute" | "displacement"
const OUTDIR = get(ENV, "EDM_OUTDIR", joinpath(pkgdir(ElectronDynamicsModels), "runs", "field_campaign_cm_phi0"))
const N_CAP = haskey(ENV, "EDM_N") ? parse(Int, ENV["EDM_N"]) : typemax(Int)
const N_PLOT_TRAJ = parse(Int, get(ENV, "EDM_N_PLOT_TRAJ", "800"))
# Dense samples/worldline. Displacement zooms in on the (circular, circular_minus) quiver, which
# needs ~26 pts/optical-cycle to render smoothly; absolute hides the quiver under the disk so 1500
# suffices. Default tracks COORDS; override with EDM_NPATH.
const NPATH = parse(Int, get(ENV, "EDM_NPATH", COORDS == "displacement" ? "10000" : "1500"))
COORDS in ("absolute", "displacement") || error("EDM_COORDS must be \"absolute\" or \"displacement\", got \"$COORDS\"")

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

# ── Analyze one source run: reconstruct, re-solve, plot zx + zy, write manifest ──
function analyze_run(mfile)
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

    @info "run" file = basename(mfile) a₀ φ₀ N COORDS parent_run_id

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

    # Dense solve (no saveat) so the worldlines can be sampled with the continuous interpolation;
    # only ~1.3k steps/trajectory ⇒ <1 GB for 10⁴ electrons.
    ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
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

# ── Drive over the source campaign ──
isdir(SOURCE_CAMPAIGN) || error("EDM_SOURCE_CAMPAIGN is not a directory: $SOURCE_CAMPAIGN")
manifests = sort([joinpath(SOURCE_CAMPAIGN, f) for f in readdir(SOURCE_CAMPAIGN)
                  if startswith(f, "run_") && endswith(f, ".toml")])
isempty(manifests) && error("no run_*.toml found in $SOURCE_CAMPAIGN")
@info "trajectory analysis" SOURCE_CAMPAIGN OUTDIR runs = length(manifests) PHASE_SPEC COORDS N_PLOT_TRAJ NPATH
mkpath(OUTDIR)
for mfile in manifests
    analyze_run(mfile)
end
println("done — $(length(manifests)) run(s) → $OUTDIR")
