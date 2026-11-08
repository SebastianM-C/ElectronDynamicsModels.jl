# Shared physics setup for the Thomson-scattering animation: hero parameters,
# trajectory solve, scene-coordinate mapping, analytic field evaluator, grids,
# and the canonical animation time window. Included by thomson_animation.jl
# (rendering) and precompute_radiation.jl (GPU far-field cube) so both always
# agree on parameters and frame times.

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using OrdinaryDiffEqNonlinearSolve
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra

# Atomic units
const c = 137.03599908330932

# ── Hero parameters ──
ω = 0.057
T0 = 2π / ω              # laser period
λ = 2π * c / ω
w₀ = 4λ                  # stylized waist: helix pitch (λ) visible against the donut (w₀)
a₀ = 1.0                 # ponderomotive kick ∝ a0² — keeps the scattering visible
# EDM_ANIM_MODE (changing it invalidates every cube): hero = m 1, p 0, linear (single-start helix; signed E-components
# carry the helical wavefronts), τ 2.5 T0; thesis = the thesis runs' mode, m −2, p 2, circular_minus, φ₀ −π/2, τ 3 T0
# (as setup_inverse.jl's thesis mode). EDM_ANIM_RMAX (disk radius, w₀; default 1.2, the thesis runs use 3.25),
# EDM_ANIM_N (electrons, default 800) and EDM_ANIM_HALF_W (transverse half-width of the box, w₀; default 2) widen the
# scene for the thesis mode, whose outer ring peaks at √5 w₀.
const ANIM_MODE = lowercase(get(ENV, "EDM_ANIM_MODE", "hero"))
const MODES = Dict(
    "hero" => (; m = 1, p = 0, pol = :linear, τ = 2.5T0, ϕ₀ = 0.0),
    "thesis" => (; m = -2, p = 2, pol = :circular_minus, τ = 3.0T0, ϕ₀ = -π / 2),
)
haskey(MODES, ANIM_MODE) || error("EDM_ANIM_MODE must be hero or thesis, got $(repr(ANIM_MODE))")
τ = MODES[ANIM_MODE].τ
p_radial = MODES[ANIM_MODE].p
m_azimuthal = MODES[ANIM_MODE].m
pol = MODES[ANIM_MODE].pol
ϕ₀ = MODES[ANIM_MODE].ϕ₀

@named world = Worldline(:τ, :atomic)
@named laser = LaguerreGaussLaser(;
    wavelength = λ,
    a0 = a₀,
    beam_waist = w₀,
    radial_index = p_radial,
    azimuthal_index = m_azimuthal,
    world,
    temporal_profile = :gaussian,
    temporal_width = τ,
    focus_position = 0.0,
    polarization = pol,
    initial_phase = ϕ₀,
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

# ── Animation time window (lab time) ──
# Asymmetric on purpose: the radiation shell moves at c — the same speed as the
# pulse — and clears the box by ~20T0, but the scattered electrons only drift at
# a few % of c, so the long tail after the interaction is where their motion
# (and the departing flash) reads. The radiation cube is sampled at exactly
# these frame times, so changing the window means re-running the precompute.
t_start = -12T0
t_end = 36T0
n_frames = 480
frame_times = LinRange(t_start, t_end, n_frames)

# Playback window: what the slider and animate() actually show. Decoupled from
# frame_times (the DATA clock — changing that invalidates every precomputed
# product), so it can be trimmed freely; with the detector at +16λ the show is
# over by ~26T0 and the last stretch of the data window is dead air.
t_play_end = 28T0
n_play_frames = 400
play_times = LinRange(t_start, t_play_end, n_play_frames)

# ── Electron cloud: sunflower disk at the focal plane, at rest ──

N = parse(Int, get(ENV, "EDM_ANIM_N", "800"))
Rmax = parse(Float64, get(ENV, "EDM_ANIM_RMAX", "1.2")) * w₀
R₀ = Rmax * sunflower(N, 2)

# ── Trajectory solve ──
# Proper-time span covers the lab-time window with margin (τ ≈ t at these γ);
# outside the pulse the electrons are at rest / in uniform drift, where the
# CubicSpline Extension extrapolation used by the retarded-time solve is exact.
τi = t_start - 4T0
τf = t_end + 4T0
tspan = (τi, τf)

x⁰ = [τi * c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]
u0 = [sys.x => x⁰, sys.u => u⁰]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
)

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, ctx)
    r = R₀[ctx.sim_id]
    x_new = SVector{4}(τi * c, r[1], r[2], 0.0)
    u_new = SVector{4}(c, 0.0, 0.0, 0.0)
    u0, p = set_x(prob, SVector{8}(x_new..., u_new...))
    return remake(prob; u0, p)
end

ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
@info "solving $N trajectories"
solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    reltol = 1e-12, abstol = 1e-12, trajectories = N,
    saveat = τi:(T0 / 16):τf,
)
trajs = trajectory_interpolants(solution)

# ── Scene coordinates ──
# Makie's Camera3D orbits around world z (`fixed_axis = true`, hardcoded), so
# interactive rotation only feels right with z as screen-vertical. We therefore
# permute PHYSICS axes into SCENE axes at render time:
#   scene X = physics z (propagation, horizontal, pulse travels −X → +X)
#   scene Y = physics x
#   scene Z = physics y (screen-vertical, orbit axis)
# SVector instead of Point3f: Makie converts StaticVectors natively, and this
# keeps setup.jl loadable in the scripts env (no GeometryBasics there).
scene_point(xμ) = SVector{3, Float32}(xμ[4], xμ[2], xμ[3])

# ── Lab-time sampling ──
# Trajectories are parameterized by proper time; invert x⁰(τ)/c = t by bisection
# (x⁰ is strictly increasing in τ).
function position_at_labtime(traj, t)
    lo, hi = τi, τf
    for _ in 1:48
        mid = 0.5 * (lo + hi)
        xμ, _ = traj(mid)
        if xμ[1] / c < t
            lo = mid
        else
            hi = mid
        end
    end
    xμ, _ = traj(0.5 * (lo + hi))
    return scene_point(xμ)
end

electron_positions(trajs, t) = [position_at_labtime(traj, t) for traj in trajs]

# ── Spatial box (shared by the pulse sampling and the radiation cube) ──
fe = FieldEvaluator(laser)

nx, ny, nz = 128, 128, 384
const HALF_W_BOX = parse(Float64, get(ENV, "EDM_ANIM_HALF_W", "2"))
xs = LinRange(-HALF_W_BOX * w₀, HALF_W_BOX * w₀, nx)
ys = LinRange(-HALF_W_BOX * w₀, HALF_W_BOX * w₀, ny)
zs = LinRange(-12λ, 12λ, nz)   # extended along propagation so the flight reads
