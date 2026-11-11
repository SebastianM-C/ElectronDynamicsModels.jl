# Shared physics setup for the INVERSE Thomson-scattering animation: the counterpart of
# setup.jl (same hero waist, same 800-electron sunflower disk, same scene-axis permutation),
# but the disk flies in along +z at βc and the laser counter-propagates along −z
# (k_direction = [0,0,-1], the convention of scripts/inverse_thomson_scattering.jl).
# Both cross the focus (z = 0) at lab time t = 0. Included by precompute_radiation.jl when
# EDM_ANIM_GEOMETRY=inverse, and by the renderers, so both agree on parameters and frame
# times.
#
# ENV knobs (one file covers every variant; changing any of them invalidates a cube):
#   EDM_ANIM_GAMMA    electron Lorentz factor                    (default 1.5)
#   EDM_ANIM_A0       laser a0                                   (default 1.0)
#   EDM_ANIM_MODE     hero | thesis                              (default hero)
#                       hero:   m = 1,  p = 0, linear,          τ = 2.5 T0 (as setup.jl)
#                       thesis: m = −2, p = 2, circular_minus,  τ = 3 T0   (as pull-out #2)
#   EDM_ANIM_N        electrons                                  (default 800)
#   EDM_ANIM_T_START / EDM_ANIM_T_END   data window, in T0       (default −12 / 24)
#   EDM_ANIM_NFRAMES  frames on the data clock                   (default 480)
#   EDM_ANIM_SAVEAT   trajectory knots per proper-time carrier period (default 16)
#
# Physics notes (a0 = 1, γ = 1.5): the light-front momentum h = γ(1+β) = 2.62 is conserved in
# the wave, so p_z drops from γβ = 1.12 to (h² − 1 − a²)/2h = 0.93 m c at the peak: a visible
# slowdown, no reflection (that needs a > √(h² − 1) = 2.4). The on-axis forward line is
# ω_s/ω = (1+β)/(1−β) = 6.85 at γ = 1.5 (13.9 at γ = 2), redshifted by ~1 + a0²/2 (linear polarization;
# 1 + a0² for circular).

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

envf(k, d) = parse(Float64, get(ENV, k, string(d)))

# ── Hero parameters ──
ω = 0.057
T0 = 2π / ω
λ = 2π * c / ω
w₀ = 4λ
const ANIM_MODE = lowercase(get(ENV, "EDM_ANIM_MODE", "hero"))
const MODES = Dict(
    "hero" => (; m = 1, p = 0, pol = :linear, τ = 2.5T0, ϕ₀ = 0.0),
    "thesis" => (; m = -2, p = 2, pol = :circular_minus, τ = 3.0T0, ϕ₀ = -π / 2),
)
haskey(MODES, ANIM_MODE) || error("EDM_ANIM_MODE must be hero or thesis, got $(repr(ANIM_MODE))")
md = MODES[ANIM_MODE]
τ = md.τ
a₀ = envf("EDM_ANIM_A0", 1.0)
p_radial = md.p
m_azimuthal = md.m
pol = md.pol
ϕ₀ = md.ϕ₀

const GAMMA = envf("EDM_ANIM_GAMMA", 1.5)
GAMMA > 1 || error("EDM_ANIM_GAMMA must be > 1 (the electrons move), got $GAMMA")
const β = sqrt(1 - 1 / GAMMA^2)
const u⁰_t = GAMMA * c              # γc
const u³_z = c * sqrt(GAMMA^2 - 1)  # +z: γβc (√(γ²−1) avoids the 1 − 1/γ² cancellation)
const DOPPLER = (1 + β) / (1 - β)   # on-axis forward ω_s/ω (linear limit)

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
    k_direction = [0, 0, -1],
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

# ── Animation time window (lab time) ──
# At t_start = −12 T0 the pulse peak is at z = +12λ (it arrives through the detector side)
# and the disk at −12βλ; the forward flash reaches the +16λ detector at t ≈ 16 T0 and the
# disk at 16/β T0 (21.5 T0 at γ = 1.5), so −12…24 T0 holds the whole story.
t_start = envf("EDM_ANIM_T_START", -12) * T0
t_end = envf("EDM_ANIM_T_END", 24) * T0
n_frames = parse(Int, get(ENV, "EDM_ANIM_NFRAMES", "480"))
frame_times = LinRange(t_start, t_end, n_frames)
t_play_end = t_end
n_play_frames = n_frames
play_times = frame_times

# ── Electron cloud: sunflower disk, crossing z = 0 at t = 0 ──
N = parse(Int, get(ENV, "EDM_ANIM_N", "800"))
Rmax = 1.2w₀
R₀ = Rmax * sunflower(N, 2)

# ── Trajectory solve ──
# Free flight before the pulse: t = γτ, so τ = 0 ↔ t = 0 at the focus. The span starts early
# enough to cover the retarded times of every box point (the box is ~30λ long, so emission
# observed at t_start left up to ~30 T0 earlier) and ends late enough for the slowed-down
# electrons: dt/dτ = γ(τ) ≥ 1, so τf = t_end + margin always reaches t_end.
const BOX_LAMBDA = 30.0
τi = (t_start - (BOX_LAMBDA + 4) * T0) / GAMMA
τf = t_end + 4T0
tspan = (τi, τf)
# The electron meets the carrier at ω γ(1+β) per unit proper time: uniform knots per THAT
# period (the inverse script's floor-study fix), and dtmax = τ/(2γ) so the adaptive solver
# cannot leap over the interaction.
const SAVEAT_KNOTS = envf("EDM_ANIM_SAVEAT", 16)
const DTMAX = τ / (2 * GAMMA)
saveat = collect(τi:(T0 / (GAMMA * (1 + β)) / SAVEAT_KNOTS):τf)

x⁰ = [u⁰_t * τi, 0.0, 0.0, u³_z * τi]
u⁰ = [u⁰_t, 0.0, 0.0, u³_z]
u0 = [sys.x => x⁰, sys.u => u⁰]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
)

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, ctx)
    r = R₀[ctx.sim_id]
    x_new = SVector{4}(u⁰_t * τi, r[1], r[2], u³_z * τi)
    u_new = SVector{4}(u⁰_t, 0.0, 0.0, u³_z)
    u0, p = set_x(prob, SVector{8}(x_new..., u_new...))
    return remake(prob; u0, p)
end

ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
@info "solving $N inverse trajectories" GAMMA β a₀ ANIM_MODE DOPPLER n_frames
solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    reltol = 1e-12, abstol = 1e-12, dtmax = DTMAX, trajectories = N, saveat,
)
trajs = trajectory_interpolants(solution)

# ── Scene coordinates (same permutation as setup.jl) ──
#   scene X = physics z (propagation), scene Y = physics x, scene Z = physics y
scene_point(xμ) = SVector{3, Float32}(xμ[4], xμ[2], xμ[3])

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

# ── Spatial box (pulse sampling and the radiation cube's default slice range) ──
# The detector side (+z) is where the flash goes, so the box runs −12λ … +16λ.
fe = FieldEvaluator(laser)

nx, ny, nz = 128, 128, 448
xs = LinRange(-2w₀, 2w₀, nx)
ys = LinRange(-2w₀, 2w₀, ny)
zs = LinRange(-12λ, 16λ, nz)
