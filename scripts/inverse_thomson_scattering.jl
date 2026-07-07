# Inverse-Thomson (Compton-backscatter) field run: same machinery as thomson_scattering.jl
# (LaguerreGauss circular beam, sunflower electrons, ENV-driven, GPUKernelRK4, split
# Liénard–Wiechert E/B via `accumulate_field`) but the electrons are RELATIVISTIC and
# COUNTER-PROPAGATE against the pulse instead of starting at rest:
#
#   • electrons move in +z with Lorentz factor γ (EDM_GAMMA): u = [γc, 0, 0, +c√(γ²−1)];
#   • the laser is reversed to propagate in −z (k_direction = [0,0,-1]) so it hits the
#     electrons head-on;
#   • the initial positions/time are chosen so every electron crosses the focus (z=0)
#     at t=0, exactly when the pulse peak is there — they "meet at the origin";
#   • the forward-beamed, Doppler-upshifted radiation is still observed on the +Z screen
#     (the electrons beam into +z, the direction of their motion).
#
# The rest-electron thomson_scattering.jl remains the standard-Thomson reference; this is
# its inverse/backscatter sibling and shares the same reduction/plotting (harmonic_products.jl).
#
# NOTE on frequency resolution + ALIASING (read before trusting any harmonic map): head-on backscatter
# upshifts the on-axis line to ω_s = (1+β)/(1−β)·ω = γ²(1+β)²·ω ≈ 4γ²ω — exactly ≈398ω at γ=10. That is
# FAR above the SPP=16 Nyquist (8ω), so the default grid CANNOT resolve it. Worse than "unresolved": with
# 16 samples/period a 398ω wave advances 24.875 cycles/sample, so it folds (398 = 25·16 − 2) and ALIASES
# onto ≈2ω — the whole backscatter signal masquerades as a clean 2ω harmonic map. So do NOT read the 2ω
# map from a base-sampling inverse run as a physical 2ω harmonic; it is the aliased 398ω beam pattern.
# To resolve ω_s needs SPP ≳ 2·398 (Nyquist); since the scattered burst is short (~τ/ω_s, ~24 cycles) the
# feasible route is a NARROW high-rate window centred on the arrival (x⁰≈Z), not a 100× bigger cube —
# deferred together with the ±8τ tspan / x⁰_start window (the interaction is Doppler-shortened).
#
# NOTE on solver tolerances: the difficulty is NUMERICAL, not a strong-field regime. a0 = eE/(mcω) is
# Lorentz-invariant (still 0.1 here — in the rest frame E and ω both boost by ~2γ, so the electron quiver
# stays weakly nonlinear and the transverse kicks are only ~0.1c). But in PROPER TIME the carrier
# oscillates at ~2γω, the interaction window is Doppler-shortened to ~τ/γ, and the 4-state is γ× larger —
# so at the rest-electron tolerances the adaptive Vern9 LEAPS OVER the interaction near τ=0 and silently
# returns garbage (u·u — an exact invariant of the Lorentz force — violated ~50×). This script tightens
# the defaults to reltol 1e-13 / abstol 1e-11 AND caps dtmax = τ/(2γ) as an anti-skip net; validated
# (CPU) to conserve u·u to ~1e-13 across the sunflower disk at γ=10, meet-at-origin holding to ~1e-6.
#
# ENV knobs (defaults = full production): EDM_GPU_BACKEND (rocm|cuda), EDM_A0, EDM_GAMMA,
# EDM_INITIAL_PHASE, EDM_POL (linear|circular[_plus]|circular_minus), EDM_NX, EDM_N,
# EDM_NSAMPLES, EDM_SPP, EDM_NSUBSTEPS, EDM_SYNC_PER_ELECTRON, EDM_OUTDIR.
# Backscatter-spectrum knobs: EDM_WINDOW (full|narrow), EDM_SCREEN_HW (screen half-width in w₀,
# default 25), EDM_HARMONICS (comma-sep n; default ≈4γ² in :narrow). Guards (fail fast, BEFORE the
# ensemble solve): requested harmonics must clear Nyquist (any mode); the :full window must reach
# the burst; the field cube must fit device memory (EDM_SKIP_MEMCHECK=1 overrides).
# Writes the field .jls + per-harmonic PNGs + run_<uuid>.toml manifest.
#
# To resolve the ≈4γ²ω backscatter line (example, on-axis, ~0.7× the default VRAM):
#   EDM_WINDOW=narrow EDM_SCREEN_HW=5 EDM_SPP=2048 EDM_OUTDIR=runs/inv julia --project=scripts \
#       scripts/inverse_thomson_scattering.jl

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using OrdinaryDiffEqNonlinearSolve
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using FFTW
using CairoMakie
using AcceleratedKernels
using Serialization
using Printf
using UUIDs
using RunManifests

include(joinpath(@__DIR__, "harmonic_products.jl"))   # write_harmonic_products (shared with the recovery path)
include(joinpath(@__DIR__, "gpu_telemetry.jl"))   # with_gpu_sampler + gpu_manifest_section → the manifest [gpu] section

# GPU backend selected via ENV: "rocm" (workstation default) or "cuda" (issaf H200).
const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "rocm"))
if GPU_BACKEND == "cuda"
    using CUDA
    const gpu_backend = CUDA.CUDABackend()
elseif GPU_BACKEND == "rocm"
    using AMDGPU
    const gpu_backend = AMDGPU.ROCBackend()
else
    error("EDM_GPU_BACKEND must be \"cuda\" or \"rocm\", got $(repr(GPU_BACKEND))")
end

# Atomic units
const c = 137.03599908330932

# ── Run configuration (ENV-overridable; defaults reproduce the full production run) ──
const ϕ₀ = parse(Float64, get(ENV, "EDM_INITIAL_PHASE", "0.0"))
const OUTDIR = get(ENV, "EDM_OUTDIR", ".")
const NX = parse(Int, get(ENV, "EDM_NX", "400"))
const NELEC = parse(Int, get(ENV, "EDM_N", "10000"))
const NSAMPLES = parse(Int, get(ENV, "EDM_NSAMPLES", "8000"))
const SPP = parse(Int, get(ENV, "EDM_SPP", "16"))
const NSUBSTEPS = parse(Int, get(ENV, "EDM_NSUBSTEPS", "1"))
const RELTOL = parse(Float64, get(ENV, "EDM_RELTOL", "1e-13"))   # ODE-solve rel tolerance (Vern9);
#   tighter than the rest-electron default — the boosted, γ×-larger state + fast 2γω oscillation let a
#   looser reltol leap over the interaction near τ=0 (u·u blows up). See DTMAX for the anti-skip net.
const ABSTOL_ENV = get(ENV, "EDM_ABSTOL", "")                    # "" ⇒ 1e-11 (see ABSTOL); else this Float64
const INTERP_SAVEAT = get(ENV, "EDM_INTERP_SAVEAT", "4")         # trajectory-spline knots per PROPER-TIME carrier period;
#   uniform saveat = T/(γ(1+β))/knots is ALWAYS applied (the small-a0 floor study's production fix:
#   the CubicSpline needs dense UNIFORM knots — adaptive Vern9 output is sparse at small a0 and its
#   non-uniformity is the h2/h1 floor source). The reference period is the Doppler-shifted carrier
#   the electron actually sees (~2γω in τ), NOT the lab period T — see the note at the ensemble solve.
const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))
const GAMMA = parse(Float64, get(ENV, "EDM_GAMMA", "10.0"))      # electron Lorentz factor (counter-propagating, +z)
GAMMA >= 1.0 || error("EDM_GAMMA must be ≥ 1, got $GAMMA")
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))
const FIELD_MODE = Symbol(get(ENV, "EDM_FIELD_MODE", "split"))   # :split → (E,B,E_far,B_far) | :total → (E,B) only (halves VRAM/output)
FIELD_MODE in (:split, :total) || error("EDM_FIELD_MODE must be \"split\" or \"total\", got \"$FIELD_MODE\"")
# Observer-time window mode. :full = the wide legacy window (coarse δt=T/SPP; at SPP=16 the ≈4γ²ω
# backscatter line ALIASES onto ~2ω — see header). :narrow = recentre the sample window on the burst
# (arrival ≈ Z; width DERIVED from the Doppler compression — see the window block) and auto-size
# N_samples to just bracket it across the screen, so a HIGH EDM_SPP resolves ≈4γ²ω without a huge
# cube (VRAM ∝ N_samples). Pair :narrow with EDM_SPP≈2048.
const WINDOW = Symbol(get(ENV, "EDM_WINDOW", "full"))
WINDOW in (:full, :narrow) || error("EDM_WINDOW must be \"full\" or \"narrow\", got \"$WINDOW\"")
const SCREEN_HW = parse(Float64, get(ENV, "EDM_SCREEN_HW", "25"))   # screen half-width in w₀; shrink (e.g. 5)
#   in :narrow mode to cut the flat-screen geometric arrival spread (∝ hw²/Z) ⇒ shorter window ⇒ less VRAM.
const SKIP_POST = get(ENV, "EDM_SKIP_POSTPROCESS", "0") == "1"   # field-only: serialize cube + manifest, defer the (CPU/IO) reduction to an async step
const RUN_TAG = get(ENV, "EDM_RUN_TAG", string(uuid4()))   # launcher may pin via EDM_RUN_TAG so .jls/log/manifest share one id
mkpath(OUTDIR)
@info "Inverse-Thomson (field) run config" RUN_TAG GPU_BACKEND ϕ₀ A0 GAMMA WINDOW SCREEN_HW SYNC FIELD_MODE OUTDIR NX NELEC NSAMPLES SPP NSUBSTEPS
const T_START = time()   # wall-clock start → [timing].total in the manifest

# Laser parameters
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ
Rmax = 3.25w₀

a₀ = A0

p_radial = 2
m_azimuthal = -2
pol = Symbol(get(ENV, "EDM_POL", "circular_minus"))   # EDM_POL: linear | circular[_plus] | circular_minus (default matches the LPWA analytic trajectory's spin); recorded in [laser].pol
profile = :gaussian
z_focus = 0.0

# Counter-propagating (inverse-Thomson) 4-velocity of the electrons, moving in +z with Lorentz
# factor γ. Under the package metric (+,−,−,−) with u·u = c²: u⁰ = γc, u³ = +c√(γ²−1) = γβc.
# √(γ²−1) is used directly (avoids the 1−1/γ² cancellation at large γ).
const β = sqrt(1 - 1 / GAMMA^2)
const u⁰_t = GAMMA * c              # time component γc
const u³_z = c * sqrt(GAMMA^2 - 1)  # +z spatial component γβc (> 0)
# Anti-skip cap on the max proper-time step: the interaction's 1/e width is ~τ/γ, so half of it
# guarantees the adaptive solver lands ≥1 step inside instead of leaping over it (see the tolerance note).
const DTMAX = τ / (2 * GAMMA)

# On-axis backscatter fundamental in units of ω: ω_s/ω = (1+β)/(1−β) ≈ 4γ² (≈398 at γ=10). This is the
# harmonic the :narrow window targets. EDM_HARMONICS overrides (comma-separated n); default is the
# fundamental ±1 + its 2nd (narrow) or the legacy (1,2,3,4) (full — the ω-harmonics of the drive laser).
const N0 = round(Int, (1 + β) / (1 - β))
const HARMONICS = let h = get(ENV, "EDM_HARMONICS", "")
    isempty(h) ? (WINDOW == :narrow ? (N0 - 1, N0, N0 + 1, 2N0) : (1, 2, 3, 4)) :
        Tuple(parse.(Int, split(h, ",")))
end
# Guard the aliasing trap — in ANY window mode: `harmonic_bins` would otherwise locate a
# super-Nyquist harmonic on the nearest in-range rfft bin and the maps would be published labeled
# with the requested n while holding that bin's content (harmonic_bins itself now also errors).
2 * maximum(HARMONICS) > SPP &&
    error(
    "resolving n=$(maximum(HARMONICS))ω needs EDM_SPP ≥ 2·max(harmonic) = " *
        "$(2 * maximum(HARMONICS)) to clear Nyquist; got EDM_SPP=$SPP"
)

# Time span (proper time). Kept at ±8τ for parity with thomson_scattering.jl; the boosted
# electron's interaction is Doppler-shortened, so this can likely be tightened later.
τi = -8τ
τf = 8τ
tspan = (τi, τf)

# ── Screen geometry + observer window — sized HERE, before the (expensive) ensemble solve,
# so the coverage/memory guards below fail fast instead of after hours of integration. ──
const Z = 2.0e5λ
const samples_per_period = SPP
const δt = 2π / ω / samples_per_period
const screen_hw = SCREEN_HW * w₀
Nx = NX
Ny = NX

# Observer-time sample window (x⁰_start + N_samples), mode-dependent:
#  :full   — legacy wide window. x⁰_start = c·τi + hypot(Z, screen_edge) (plain c·τi, NOT γc·τi: the boosted
#            electron starts far behind at γc·τi but only radiates near t=0, whose light reaches the screen
#            at x⁰≈Z; c·τi is a small lead-in — γc·τi would start ~γ× too early and miss it). N = EDM_NSAMPLES.
#  :narrow — recentre on the burst: on-axis arrival ≈ Z (measured + ~0.003 T at γ=10); off-axis pixels
#            arrive LATER by the flat-screen path excess |r_o−r_e|²/(2Z), up to ~(√2·screen_hw + Rmax)²/(2Z)
#            at the corner (worst emitter). Window = lead + that spread + burst + tail; N_samples auto-sized
#            to it, so raising EDM_SPP resolves ≈4γ²ω at a fixed/short window.
# The burst length is DERIVED, not frozen at a measurement: the ~τ-long interaction is observed
# Doppler-compressed by (1−β)/(1+β) = 1/N0, so the 1%-field Gaussian width is 2√(ln 100)·cτ/N0
# (= 0.257λ at γ=10 — matching the CPU-LW measured ~0.26 T to <1%); the ×1.4 margin reproduces the
# previously hand-budgeted 0.35λ at γ=10 and keeps the window correct at low γ, where the burst is
# ∝ 1/N0 wider (≈3λ at γ=3 — a frozen 0.35λ budget would clip it).
const burst = 1.4 * 2 * sqrt(log(100)) * c * τ / N0
const corner_spread = (√2 * screen_hw + Rmax)^2 / (2Z)   # latest arrival excess (corner pixel, worst-side emitter)
const N_samples, x⁰_start = if WINDOW == :full
    NSAMPLES, c * τi + hypot(Z, screen_hw + Rmax)
else
    lead = 0.5λ; tail = 0.5λ                            # lead-in / tail (x⁰ lengths, 1λ = 1 T)
    x0 = Z - lead                                       # arrival ≈ Z ⇒ burst sits ~`lead` into the window
    ceil(Int, (lead + corner_spread + burst + tail) / (c * δt)), x0
end
# Coverage guard (:full only — :narrow covers by construction): the burst arrives at ≈Z
# (+corner_spread off-axis), but the legacy window starts a fixed ≈8cτ ≈ 191λ BEFORE Z with length
# NSAMPLES·λ/SPP — so raising EDM_SPP (the natural knob for the ≈4γ²ω line) shrinks it until it
# ends before any signal arrives, and the all-zero cube would serialize + reduce + publish silently.
WINDOW == :full && x⁰_start + (N_samples - 1) * c * δt < Z + corner_spread + burst &&
    error(
    "EDM_WINDOW=full observer window ends " *
        "$(round((Z + corner_spread + burst - x⁰_start - (N_samples - 1) * c * δt) / λ, digits = 1))λ " *
        "before the backscatter burst (arrival ≈ Z + corner spread): raise EDM_NSAMPLES, lower " *
        "EDM_SPP, or use EDM_WINDOW=narrow (auto-sized)."
)
# Memory guard: the split-LW kernel eagerly allocates nbuf × (Nx·Ny·3·N_samples) Float64 buffers per
# device (electron sharding replicates them per device), so an oversized window — e.g. :narrow at the
# default EDM_SCREEN_HW=25, whose corner_spread ≈ 21λ at SPP=2048 asks for ~700 GB — must fail HERE,
# not OOM after the ensemble solve. EDM_SKIP_MEMCHECK=1 overrides (exotic backends/memory pools).
let nbuf = FIELD_MODE == :split ? 4 : 2, bytes = nbuf * N_samples * 3 * NX * NX * 8
    @info "field cube estimate" N_samples nbuf cube_gib = round(bytes / 2^30, digits = 1)
    if get(ENV, "EDM_SKIP_MEMCHECK", "0") != "1"
        mem = try
            gpu_memory_info(gpu_backend)
        catch err
            @warn "gpu_memory_info failed — skipping the field-cube memory check" err
            nothing
        end
        mem !== nothing && bytes > 0.9 * mem.total && error(
            "estimated field cube ($(round(bytes / 2^30, digits = 1)) GiB in $nbuf buffers) exceeds " *
                "90% of device memory ($(round(mem.total / 2^30, digits = 1)) GiB): shrink " *
                "EDM_SCREEN_HW (corner_spread ∝ screen_hw²), lower EDM_SPP, or set " *
                "EDM_FIELD_MODE=total. EDM_SKIP_MEMCHECK=1 overrides."
        )
    end
end

@named world = Worldline(:τ, :atomic)

# Reversed laser: k_direction = [0,0,-1] flips the internal kz_sign so the carrier phase,
# traveling-wave envelope, Gouy/curvature and all E/B components propagate toward −z (head-on).
@named laser = LaguerreGaussLaser(;
    wavelength = λ,
    a0 = a₀,
    beam_waist = w₀,
    radial_index = p_radial,
    azimuthal_index = m_azimuthal,
    world,
    temporal_profile = profile,
    temporal_width = τ,
    focus_position = z_focus,
    polarization = pol,
    initial_phase = ϕ₀,
    k_direction = [0, 0, -1],
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

# Meet-at-origin timing: with x⁰(τ)=γc·τ and z(τ)=γβc·τ (force-free flight), every electron
# crosses z=0 at τ=0 ⇔ t=0, exactly when the −z pulse peaks at the focus. So the on-axis start
# (τ=τi) is x⁰=γc·τi (< 0, in the past) and z=γβc·τi (< 0, far behind the focus in −z).
x⁰ = [u⁰_t * τi, 0.0, 0.0, u³_z * τi]
u⁰ = [u⁰_t, 0.0, 0.0, u³_z]
u0 = [sys.x => x⁰, sys.u => u⁰]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
)
sol0 = solve(prob, Vern9(), reltol = 1.0e-15, abstol = 1.0e-12, dtmax = DTMAX)

# Sunflower distribution for electron positions
const ϕ = (1 + √5) / 2

function radius(k, n, b)
    if k > n - b
        return 1.0
    else
        return sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
    end
end

function sunflower(n, α)
    points = Vector{Vector{Float64}}()
    angle_stride = 2π / ϕ^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        θ = k * angle_stride
        push!(points, [r * cos(θ), r * sin(θ)])
    end
    return points
end

# Ensemble solve. Each electron gets the sunflower transverse offset (r) at its −z start plane
# and the same boosted 4-velocity; the shared timing makes them all reach z=0 (their transverse
# offset, at the waist) at t=0.
N = NELEC
R₀ = Rmax * sunflower(N, 2)
xμ = [[u⁰_t * τi, r..., u³_z * τi] for r in R₀]

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, ctx)
    i = ctx.sim_id
    x_new = SVector{4}(xμ[i]...)
    u_new = SVector{4}(u⁰_t, 0.0, 0.0, u³_z)
    u0, p = set_x(prob, SVector{8}(x_new..., u_new...))
    return remake(prob; u0, p)
end

# Absolute tolerance: a fixed tight 1e-11, NOT the rest-electron a₀-tuned abserr(a₀) ≈ 4e-10. The
# boosted problem's difficulty comes from the large state magnitude + fast 2γω oscillation, not the
# nominal field amplitude, so an a₀-tuned abstol is inappropriate here; 1e-11 conserves u·u to ~1e-13 at γ=10.
const ABSTOL = isempty(ABSTOL_ENV) ? 1.0e-11 : parse(Float64, ABSTOL_ENV)
# Uniform saveat so the trajectory CubicSpline gets dense UNIFORM knots (the floor-study fix,
# applied unconditionally). The knot spacing divides the PROPER-TIME carrier period T/(γ(1+β)) ≈
# T/2γ — the period the boosted electron actually quivers at — not the lab period T (the
# rest-electron sibling's convention): spacing by T would deliver γ(1+β)× fewer knots per
# oscillation than the knob promises, and since `saveat` REPLACES Vern9's dense output as the
# spline's only knots, that would alias the quiver entirely.
# RAM: knots/trajectory = 16τ·γ(1+β)·knots / T ≈ 31k at γ=10, knots=4 → ~6 MB of splines per
# trajectory, ~60 GB at N=10⁴ — fine on the cluster nodes, tight on 123 GB boxes; lower
# EDM_INTERP_SAVEAT (or EDM_N) if host RAM binds.
saveat = collect(τi:((2π / ω) / (GAMMA * (1 + β)) / parse(Float64, INTERP_SAVEAT)):τf)
ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
t_trajectories = @elapsed solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    reltol = RELTOL, abstol = ABSTOL, dtmax = DTMAX, trajectories = N, saveat
)
@info "trajectories solved" t_trajectories RELTOL ABSTOL DTMAX knots_per_period = INTERP_SAVEAT

# Radiation computation
trajs = trajectory_interpolants(solution)

# (Screen geometry + observer-window sizing and their guards live ABOVE the ensemble solve,
# so bad configurations fail before the expensive integration.)
x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)
@info "observer window" WINDOW screen_hw_w0 = SCREEN_HW N_samples window_periods = N_samples / SPP x⁰_start_rel_Z_periods = (x⁰_start - Z) / λ

screen = ObserverScreen(
    LinRange(-screen_hw, screen_hw, Nx),
    LinRange(-screen_hw, screen_hw, Ny),
    Z,
    x⁰_samples;
    c,
)

# Exact field via the split Liénard–Wiechert GPU kernel.
# Returns (; E, B, E_far, B_far), each (N_samples, 3, Nx, Ny): E, B are the total
# field (for the harmonic maps below); E_far, B_far the far (radiation) field alone.
# Multi-GPU: when >1 device is visible (e.g. SLURM --gres=gpu:h200:2) shard the electrons across
# them — linear superposition ⇒ the summed partials are exact; one device ⇒ the plain path.
ndev = gpu_device_count(gpu_backend)
# Sample GPU power/util/VRAM across the accumulate_field window on all sharded devices
# (→ manifest [gpu] stats + the gputrace TSV time series; see gpu_telemetry.jl).
gputracefile = joinpath(OUTDIR, "gputrace_$(RUN_TAG).tsv")
t_field = @elapsed begin
    fld, gpu_telem = with_gpu_sampler(gpu_backend, GPU_SAMPLE_DT;
            devices = 1:ndev, tracefile = gputracefile) do
        if ndev > 1
            @info "sharding electrons across $ndev devices"
            accumulate_field_sharded(
                trajs, screen, GPUKernelRK4(), gpu_backend;
                n_substeps = NSUBSTEPS, mode = Val(FIELD_MODE), sync_per_electron = SYNC
            )
        else
            accumulate_field(
                trajs, screen, GPUKernelRK4(), gpu_backend;
                n_substeps = NSUBSTEPS, mode = Val(FIELD_MODE), sync_per_electron = SYNC
            )
        end
    end
end
@info "field accumulated" t_field ndev

# Serialize the full split field so offline scripts can read this run directly.
# NOTE: full-res this is 4 × (N_samples·3·Nx·Ny·8) bytes ≈ 4×30.7 GB at the default
# resolution — much larger than the 4-potential .jls; size the run dir accordingly.
datafile = joinpath(OUTDIR, "field_$(Nx)_N$(N)_Ns$(N_samples)_spp$(samples_per_period)_$(RUN_TAG).jls")
serialize(datafile, fld)
println("serialized → $datafile")

# ── Harmonic maps + ∠F phase + power spectrum (reduce + serialize + plot) ──
# Shared with the standalone recovery path in harmonic_products.jl, so the reduction and
# rendering live in one place. Emits hmaps_<tag>.jls + the per-harmonic 2×3 E/B grids
# (:jet, per-panel extrema — same style as the LPWA maps), the ∠F phase grids, and the power spectrum.
if SKIP_POST
    @info "EDM_SKIP_POSTPROCESS=1 — cube serialized; harmonic maps + screen observables deferred to the async post-process"
    hprod = nothing
    plotfiles = String[]
else
    hprod = write_harmonic_products(
        fld, screen.x_grid, screen.y_grid, ω, δt;
        w₀, run_tag = RUN_TAG, outdir = OUTDIR, source_datafile = basename(datafile),
        harmonics = HARMONICS,   # :narrow ⇒ around the ≈4γ²ω backscatter line; :full ⇒ (1,2,3,4)
        title_prefix = "Inverse Thomson scattering", fileprefix = "inverse_thomson",
    )
    plotfiles = hprod.plots
end

# ── Reproducibility manifest (same schema as thomson_scattering.jl) ──
using TOML
using Dates

provenance = run_provenance(;
    run_id = RUN_TAG, gpu_backend = GPU_BACKEND, repo_dir = pkgdir(ElectronDynamicsModels),
    gpu_device = GPU_BACKEND == "cuda" ? CUDA.name(CUDA.device()) : nothing,
)

config = Dict{String, Any}(
    "initial_phase" => ϕ₀,
    "a0" => A0,
    "gamma" => GAMMA,                  # electron Lorentz factor (inverse-Thomson boost)
    "beta" => β,
    "Nx" => Nx,
    "Ny" => Ny,
    "N" => N,
    "N_samples" => N_samples,
    "samples_per_period" => samples_per_period,
    "n_substeps" => NSUBSTEPS,
    "reltol" => RELTOL,                # ODE-solve tolerances (replay); tightened for the boosted electron
    "abstol" => ABSTOL,
    "dtmax" => DTMAX,                  # anti-skip proper-time step cap = τ/(2γ) (see the solver-tolerance note)
    "interp_saveat" => INTERP_SAVEAT,  # trajectory-spline knots per proper-time carrier period (uniform, always)

    "mode" => string(FIELD_MODE),      # :split → (E,B,E_far,B_far) | :total → (E,B); mirrors lpwa.jl
    "sync_per_electron" => SYNC,       # replay input: run_spec_from_manifest reads this
    "observable" => "field",           # distinguishes this run from the 4-potential (_A) runs
    "scattering" => "inverse",         # counter-propagating boosted electrons vs. rest-electron thomson_scattering.jl
    "window" => string(WINDOW),        # :full (wide, coarse) | :narrow (burst-centred, high-SPP)
    "screen_hw_w0" => SCREEN_HW,       # EDM_SCREEN_HW knob (w₀ units; [setup].screen_hw is the derived a.u. value)
    "harmonics" => collect(HARMONICS), # harmonic bins the maps extract (≈4γ²ω for :narrow)
    "backscatter_n0" => N0,            # on-axis backscatter fundamental ω_s/ω = (1+β)/(1−β) ≈ 4γ²
)

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
    "log" => "run_$(RUN_TAG).log",   # captured by the run wrapper; travels with the run
)
if !SKIP_POST
    outputs["harmonic_maps"] = basename(hprod.hmapsfile)   # reduced maps → resolve_hmaps finds them directly
    outputs["plots"] = basename.(plotfiles)
end

laser_params = Dict{String, Any}(
    "wavelength" => prob.ps[sys.laser.λ],
    "a0" => prob.ps[sys.laser.a₀],
    "w0" => prob.ps[sys.laser.w₀],
    "p" => prob.ps[sys.laser.p],
    "m" => prob.ps[sys.laser.m],
    "pol" => string(pol),
    "profile" => string(profile),
    "temporal_width" => prob.ps[sys.laser.τ0],
    "focus_position" => prob.ps[sys.laser.z₀],
    "phi0" => prob.ps[sys.laser.ϕ₀],
    "k_direction" => "[0, 0, -1]",     # reversed propagation (−z) — not a symbolic param, recorded literally
)
setup = Dict{String, Any}(
    "τi" => τi,
    "τf" => τf,
    "Rmax" => Rmax,
    "Z" => Z,
    "screen_hw" => screen_hw,          # screen half-width (a.u.); = SCREEN_HW·w₀
    "x0_start" => x⁰_start,            # observer-time window start (a.u.); Z-relative offset ⇒ [config].window
)   # input knobs (Nx/Ny/N/N_samples/spp) live in [config]; setup is the integration window + screen geometry

# Wall-clock phase timings → [timing] (dashboard renders total/trajectories/field, in seconds).
timing = Dict{String, Any}(
    "total" => time() - T_START,
    "trajectories" => t_trajectories,
    "field" => t_field,
)
# Sharding → [sharding] (axis → partition count). Flat + generic so future axes (e.g. a Z-split
# 3D screen) slot in with no schema change. NOT in [timing] — a device count is not a duration.
sharding = Dict{String, Any}("electrons" => ndev)
# GPU telemetry → [gpu] (static device snapshot + power/util/VRAM stats over the field window).
# `nothing` (no vendor extension / telemetry error) ⇒ the section is simply omitted.
gpu = gpu_manifest_section(gpu_backend, GPU_BACKEND, Nx * Ny, ndev, gpu_telem)
extra = Dict{String, Any}("timing" => timing, "sharding" => sharding)
gpu === nothing || (extra["gpu"] = gpu)
manifestfile = write_solver_manifest(
    OUTDIR; run_id = RUN_TAG, provenance, config, laser = laser_params, setup, outputs, extra,
)
println("manifest → $manifestfile")
