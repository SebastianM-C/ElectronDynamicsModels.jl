# scripts/plot_final_positions.jl — where the pulse leaves the electrons: IC disk under the
# LG intensity + final transverse radius products.
#
# Usage: julia --project=scripts scripts/plot_final_positions.jl <run_manifest.toml>
#
# Reconstructs the run from its manifest (thomson AND inverse manifests: `scattering=inverse`
# switches to the reversed laser k_direction=[0,0,-1] and the boosted u⁰=(γc,0,0,γβc) with the
# meet-at-origin start z=γβc·τi — mirrors inverse_thomson_scattering.jl) and re-solves ONLY the
# trajectory endpoints (save_everystep=false): no fields, no splines — minutes on a CPU.
#
# Products (chips of the parent run, plot_pixel_traces.jl sidecar pattern):
#   ic_lg_<tag>.png      the start disk over the closed-form LG intensity |u_rel|² heatmap
#                        (p=2, |m|=2 only — other modes get scatter-only, no backdrop)
#   finalpos_<tag>.png   final ρ = √(x²+y²) per starting point (disk colored by ρ_f/w₀) +
#                        histogram of ρ_f/w₀ with the fraction beyond EDM_RHO_OUT — the
#                        ponderomotive-expulsion estimate ("are they pushed out of the
#                        interaction region?")
#   finalpos_<tag>.jls   cache: xμ0, xμf, ρ0, ρf (re-render without an EDM solve)
#
# ENV knobs:
#   EDM_RHO_OUT   expulsion threshold in w₀ units (default 3.25)
#   EDM_OUTDIR    output dir, default = the manifest's directory

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using Printf
using CairoMakie
using Serialization
using TOML

const c = 137.03599908330932   # speed of light, atomic units (repo convention)

length(ARGS) == 1 || error("usage: plot_final_positions.jl <run_manifest.toml>")
const MFILE = abspath(ARGS[1])
isfile(MFILE) || error("no manifest at $MFILE")

const RHO_OUT = parse(Float64, get(ENV, "EDM_RHO_OUT", "3.25"))
const OUTDIR = get(ENV, "EDM_OUTDIR", dirname(MFILE))

# ── Reconstruct the run (mirrors plot_pixel_traces.jl / analyze_trajectories.jl) ──
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
reltol = Float64(get(cfg, "reltol", 1.0e-12))
abstol = Float64(cfg["abstol"])
p_radial = Int(laser_p["p"])
m_azim = Int(laser_p["m"])
γ = Float64(get(cfg, "gamma", 1.0))
inverse = get(cfg, "scattering", "") == "inverse"
dtmax = Float64(get(cfg, "dtmax", Inf))
ω = 2π * c / λ
RUN_TAG = m["provenance"]["run_id"]
idtag = first(RUN_TAG, 8)

@info "final positions" MFILE RUN_TAG a₀ γ inverse N RHO_OUT

@named world = Worldline(:τ, :atomic)
laser_kw = inverse ? (; k_direction = [0, 0, -1]) : (;)
@named laser = LaguerreGaussLaser(;
    wavelength = λ, a0 = a₀, beam_waist = w₀,
    radial_index = p_radial, azimuthal_index = m_azim,
    world, temporal_profile = Symbol(laser_p["profile"]), temporal_width = τ0,
    focus_position = Float64(laser_p["focus_position"]), polarization = pol,
    initial_phase = φ₀, laser_kw...,
)
elec = if get(cfg, "system", "classical") == "ll"
    @named elec = LandauLifshitzElectron(; laser)
else
    @named elec = ClassicalElectron(; laser)
end
sys = mtkcompile(elec)

# Boost + meet-at-origin timing (force-free flight: every electron crosses z=0 at τ=0 = t=0).
u⁰_t = γ * c
u³_z = inverse ? c * sqrt(γ^2 - 1) : 0.0
u⁰ = SVector{4}(u⁰_t, 0.0, 0.0, u³_z)

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, [sys.x => [u⁰_t * τi, 0.0, 0.0, u³_z * τi], sys.u => collect(u⁰)], (τi, τf);
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
xμ0 = [[u⁰_t * τi, r..., u³_z * τi] for r in R₀]   # unbunched (bunch_dz ≡ 0 checked below)
Int(get(cfg, "bunch_nb", 0)) == 0 ||
    @warn "bunched run: reconstruction ignores bunch_dz (≤0.15λ longitudinal offsets)"

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
function prob_func(prob, ctx)
    u0, p = set_x(prob, SVector{8}(SVector{4}(xμ0[ctx.sim_id]...)..., u⁰...))
    return remake(prob; u0, p)
end
dtmax_kw = isfinite(dtmax) ? (; dtmax) : (;)
t_solve = @elapsed sol = solve(
    EnsembleProblem(prob; prob_func, safetycopy = false), Vern9(), EnsembleThreads();
    reltol, abstol, trajectories = N, save_everystep = false, save_start = false, dtmax_kw...
)
@info "endpoints solved" t_solve N

xμf = [s[sys.x][end] for s in sol.u]               # final 4-position (layout-agnostic indexing)
ρ0 = [hypot(r[1], r[2]) for r in R₀] ./ w₀
ρf = [hypot(x[2], x[3]) for x in xμf] ./ w₀
pct_out = 100 * count(>(RHO_OUT), ρf) / N
Δρ = ρf .- ρ0
maxΔρ = maximum(abs, Δρ)
@info "expulsion" pct_out RHO_OUT maximum(ρf) maxΔρ

# Closed-form LG transverse intensity |u_rel|² — p = 2, |m| = 2 only (the production mode;
# same expression as inverse_thomson_scattering.jl's u_rel2). ARGS IN w₀ UNITS: σ = ρ²/w₀²
# is just x²+y² here — the plotting grid below is already normalized.
u_rel2 = (p_radial == 2 && abs(m_azim) == 2) ?
    (x, y) -> (σ = x^2 + y^2; (√12 * 2σ * (1 - 4σ / 3 + σ^2 / 3) * exp(-σ))^2) :
    nothing

# ── PNG 1: start disk over the LG intensity ──
x0w, y0w = [r[1] for r in R₀] ./ w₀, [r[2] for r in R₀] ./ w₀
ext = 1.1 * max(Rmax / w₀, RHO_OUT)
fig1 = Figure(size = (640, 560))
ax = Axis(fig1[1, 1]; title = "start disk over the LG mode intensity",
    xlabel = "x / w₀", ylabel = "y / w₀", aspect = 1)
if u_rel2 !== nothing
    gr = range(-ext, ext, length = 401)
    hm = heatmap!(ax, gr, gr, [u_rel2(xx, yy) for xx in gr, yy in gr]; colormap = :inferno)
    Colorbar(fig1[1, 2], hm; label = "|u_rel|²  (LG p=$(p_radial), m=$(m_azim))")
end
scatter!(ax, x0w, y0w; color = (:white, 0.8), strokecolor = :black, strokewidth = 0.3,
    markersize = 4)
Label(fig1[0, :], @sprintf("initial conditions — %s  (N=%d, a₀=%g, γ−1=%.3g)",
    idtag, N, a₀, γ - 1), fontsize = 16, font = :bold)
png1 = joinpath(OUTDIR, "ic_lg_$(RUN_TAG).png")
save(png1, fig1)
println("saved → $png1")

# ── PNG 2: final radius per starting point + histogram with the expulsion fraction ──
fig2 = Figure(size = (1150, 520))
ax1 = Axis(fig2[1, 1]; title = "radial displacement Δρ by starting point",
    xlabel = "x₀ / w₀", ylabel = "y₀ / w₀", aspect = 1)
crange = maxΔρ > 0 ? (-maxΔρ, maxΔρ) : (-1e-12, 1e-12)
sc = scatter!(ax1, x0w, y0w; color = Δρ, colormap = :RdBu, colorrange = crange,
    markersize = 5)
Colorbar(fig2[1, 2], sc; label = "Δρ = (ρ_f − ρ₀) / w₀")
ax2 = Axis(fig2[1, 3]; title = @sprintf("final radii — %.1f%% beyond %.2f w₀ (max |Δρ| = %.2g w₀)",
        pct_out, RHO_OUT, maxΔρ),
    xlabel = "ρ_final / w₀", ylabel = "electrons")
hist!(ax2, ρf; bins = 60, color = (:steelblue, 0.8))
vlines!(ax2, [RHO_OUT]; color = :crimson, linestyle = :dash, linewidth = 1.5)
text!(ax2, 0.98, 0.95; text = @sprintf("%.1f%% out", pct_out), align = (:right, :top),
    space = :relative, color = :crimson, fontsize = 14)
Label(fig2[0, :], @sprintf(
        "ponderomotive expulsion — %s  (a₀=%g, γ−1=%.3g; max ρ_f = %.2f w₀, start disk ≤ %.2f w₀)",
        idtag, a₀, γ - 1, maximum(ρf), Rmax / w₀), fontsize = 16, font = :bold)
png2 = joinpath(OUTDIR, "finalpos_$(RUN_TAG).png")
save(png2, fig2)
println("saved → $png2")

jlsfile = joinpath(OUTDIR, "finalpos_$(RUN_TAG).jls")
serialize(jlsfile, (; xμ0 = permutedims(reduce(hcat, xμ0)), xμf = permutedims(reduce(hcat, xμf)),
    ρ0, ρf, rho_out = RHO_OUT, pct_out, N, a₀, γ, λ, w₀, Rmax, run_id = RUN_TAG))
println("serialized → $jlsfile")

# ── Sidecars (harmonic_products derived_*.toml schema → dashboard chips) ──
repo_commit = try
    readchomp(`git -C $(pkgdir(ElectronDynamicsModels)) rev-parse HEAD`)
catch
    "unknown"
end
for (kind, label, png, pp, desc) in (
        ("ic_lg", "start disk over LG intensity", basename(png1),
            Dict{String, Any}("N" => N, "gamma_eps" => γ - 1),
            "The as-run sunflower start disk drawn over the closed-form LG p=$(p_radial), " *
            "m=$(m_azim) transverse intensity |u_rel|² — which electrons sit on the rings " *
            "that actually drive them."),
        ("finalpos", "final radii + expulsion fraction", basename(png2),
            Dict{String, Any}("N" => N, "gamma_eps" => γ - 1, "rho_out_w0" => RHO_OUT,
                "pct_out" => round(pct_out; sigdigits = 3),
                "max_rho_f_w0" => round(maximum(ρf); sigdigits = 4),
                "max_abs_drho_w0" => round(maxΔρ; sigdigits = 3)),
            "Endpoint-only trajectory re-solve: final transverse radius ρ_f = √(x²+y²) per " *
            "starting point, and the ρ_f/w₀ histogram with the fraction beyond " *
            "$(RHO_OUT) w₀ — the ponderomotive-expulsion estimate for whether electrons " *
            "leave the interaction region during the pulse."),
    )
    sidecar = Dict(
        "schema_version" => 1,
        "derived" => Dict(
            "depends_on" => [RUN_TAG], "kind" => kind, "label" => label,
            "plot" => png, "source" => basename(MFILE), "description" => desc,
        ),
        "plot_params" => pp,
        "provenance" => Dict(
            "host" => readchomp(`hostname`), "repo_commit" => repo_commit,
            "script" => "plot_final_positions.jl",
            "timestamp" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())),
        ),
        "setup" => Dict("field" => "none"),
    )
    scfile = joinpath(OUTDIR, "derived_$(kind)_$(idtag).toml")
    open(scfile, "w") do io
        TOML.print(io, sidecar)
    end
    println("sidecar → $scfile")
end
