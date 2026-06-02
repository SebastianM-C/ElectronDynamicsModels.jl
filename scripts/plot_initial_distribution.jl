# Setup diagnostic: how the initial electron distribution sits in the laser field.
# Evaluates the LaguerreGauss field (Eˣ, Eʸ, Eᶻ, |E|) over the transverse plane at
# the electrons' starting z (= focus, z=0) via FieldEvaluator (no ODE solve), and
# overlays the sunflower-distributed initial electron positions. Shows whether the
# electron disk samples the LG ring/vortex structure. Same laser + distribution as
# scripts/thomson_scattering.jl.
#
# Env: EDM_N (electrons to scatter, default 2000), EDM_A0 (default 0.1),
#      EDM_T (eval time in a.u., default 0 = focus), EDM_EXTENT (grid half-width in
#      units of Rmax, default 1.15), EDM_NGRID (field grid pts/side, default 220).
#
#   julia +release --project=scripts scripts/plot_initial_distribution.jl

using ElectronDynamicsModels
using ElectronDynamicsModels: FieldEvaluator
using ModelingToolkit
using StaticArrays
using LinearAlgebra
using Printf
using CairoMakie
using UUIDs

const c = 137.03599908330932
const ω = 0.057
const λ = 2π * c / ω
const w₀ = 75λ
const Rmax = 3.25w₀

const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))
const NSCAT = parse(Int, get(ENV, "EDM_N", "2000"))
const T_EVAL = parse(Float64, get(ENV, "EDM_T", "0.0"))
const EXTENT = parse(Float64, get(ENV, "EDM_EXTENT", "1.15")) * Rmax
const NGRID = parse(Int, get(ENV, "EDM_NGRID", "220"))
const p_radial, m_azimuthal = 2, -2
const OUTDIR = get(ENV, "EDM_OUTDIR", joinpath(pkgdir(ElectronDynamicsModels), "runs"))
const RUN_TAG = string(uuid4())

# Sunflower distribution — identical to thomson_scattering.jl.
const ϕ = (1 + √5) / 2
radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
function sunflower(n, α)
    pts = Vector{SVector{2, Float64}}(undef, n)
    stride = 2π / ϕ^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        θ = k * stride
        pts[k] = SVector(r * cos(θ), r * sin(θ))
    end
    return pts
end

# ── Laser + field evaluator (same params as thomson_scattering.jl) ──
@named world = Worldline(:τ, :atomic)
@named laser = LaguerreGaussLaser(;
    wavelength = λ, a0 = A0, beam_waist = w₀,
    radial_index = p_radial, azimuthal_index = m_azimuthal,
    world, temporal_profile = :gaussian, temporal_width = 150 / ω,
    focus_position = 0.0, polarization = :circular, initial_phase = 0.0,
)
fe = FieldEvaluator(laser)

# ── Sample the transverse field at z = 0, t = T_EVAL ──
xs = LinRange(-EXTENT, EXTENT, NGRID)
ys = LinRange(-EXTENT, EXTENT, NGRID)
Ex = Matrix{Float64}(undef, NGRID, NGRID)
Ey = similar(Ex); Ez = similar(Ex); Emag = similar(Ex)
for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
    E = fe(SVector(T_EVAL, x, y, 0.0)).E
    Ex[i, j] = real(E[1]); Ey[i, j] = real(E[2]); Ez[i, j] = real(E[3])
    Emag[i, j] = sqrt(real(E[1])^2 + real(E[2])^2 + real(E[3])^2)
end

# Electron initial transverse positions (sunflower scaled to Rmax).
elec = (Rmax,) .* sunflower(NSCAT, 2)
ex = [p[1] for p in elec]; ey = [p[2] for p in elec]

# ── Figure: Eˣ, Eʸ, Eᶻ (diverging) + |E| (sequential), electrons overlaid ──
fig = Figure()
Label(fig[0, :], @sprintf("Initial electron distribution over the laser field  (m=%d, p=%d, circular, a₀=%.0e, z=0, t=%.3g)",
        m_azimuthal, p_radial, A0, T_EVAL), fontsize = 15, font = :bold)
panels = (("Eˣ", Ex, :seismic), ("Eʸ", Ey, :seismic), ("Eᶻ", Ez, :seismic), ("|E|", Emag, :viridis))
for (k, (name, data, cmap)) in enumerate(panels)
    row, col = (k - 1) ÷ 2 + 1, (k - 1) % 2 + 1
    gl = fig[row, col] = GridLayout()
    cr = maximum(abs, data)
    # explicit width/height (not aspect) so each panel + colorbar sizes equally —
    # mirrors thomson_scattering_A.jl; aspect=1 alone collapses the right column.
    ax = Axis(gl[1, 1]; width = 330, height = 330, xlabel = "x [a.u.]", ylabel = "y [a.u.]", title = name)
    hm = if cmap == :viridis
        heatmap!(ax, xs, ys, data; colormap = cmap, colorrange = (0, cr))
    else
        heatmap!(ax, xs, ys, data; colormap = cmap, colorrange = cr > 0 ? (-cr, cr) : (-1.0, 1.0))
    end
    scatter!(ax, ex, ey; color = (:white, 0.5), markersize = 2.5, strokewidth = 0)
    Colorbar(gl[1, 2], hm, height = 330, width = 12)
end
resize_to_layout!(fig)
mkpath(OUTDIR)
out = joinpath(OUTDIR, "initial_distribution_$(RUN_TAG).png")
save(out, fig)
println("electrons: $NSCAT (Rmax = $(round(Rmax; sigdigits=4)) a.u. = $(round(Rmax/w₀; digits=2))·w₀), grid ±$(round(EXTENT/w₀; digits=2))·w₀")
println("saved → $out")

# Provenance manifest for the dashboard — a standalone analysis node (setup viz,
# no parent run). write_run_manifest records repo_commit + script, so the dashboard
# links the file that produced this plot.
include(joinpath(@__DIR__, "manifest.jl"))
write_run_manifest(OUTDIR; run_id = RUN_TAG, script = basename(PROGRAM_FILE),
    config = Dict("a0" => A0, "t_eval" => T_EVAL, "n_electrons" => NSCAT,
        "extent_over_w0" => EXTENT / w₀, "kind" => "initial_distribution"),
    laser = Dict("wavelength" => λ, "w0" => w₀, "m" => m_azimuthal, "p" => p_radial,
        "pol" => "circular", "profile" => "gaussian"),
    plots = [basename(out)])
println("manifest → $(joinpath(OUTDIR, "run_$(RUN_TAG).toml"))")
