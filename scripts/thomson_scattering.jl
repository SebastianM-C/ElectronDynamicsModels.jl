# Field-accumulation Thomson-scattering run: same setup as thomson_scattering_A.jl
# (LaguerreGauss circular beam, sunflower electrons, ENV-driven, GPUKernelRK4) but
# computes the exact radiated field via `accumulate_field` (the split Liénard–
# Wiechert E, B) instead of the 4-potential, and plots Re(Ẽ(ω)), Re(B̃(ω)) harmonic
# maps at the end. The `_A` script remains the 4-potential reference.
#
# ENV knobs (defaults = full production): EDM_GPU_BACKEND (rocm|cuda), EDM_A0,
# EDM_INITIAL_PHASE, EDM_NX, EDM_N, EDM_NSAMPLES, EDM_SPP, EDM_NSUBSTEPS,
# EDM_POL (linear|circular[_plus]|circular_minus),
# EDM_SYNC_PER_ELECTRON, EDM_OUTDIR. Writes the field .jls + per-harmonic PNGs +
# run_<uuid>.toml manifest.

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

include(joinpath(@__DIR__, "manifest.jl"))   # RunManifests: run_provenance, write_solver_manifest
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
const RELTOL = parse(Float64, get(ENV, "EDM_RELTOL", "1e-12"))   # ODE-solve rel tolerance (Vern9)
const ABSTOL_ENV = get(ENV, "EDM_ABSTOL", "")                    # "" ⇒ abserr(a0); else this Float64
const INTERP_SAVEAT = get(ENV, "EDM_INTERP_SAVEAT", "")          # trajectory-spline knots/laser-period;
#   "" ⇒ adaptive (Vern9 native steps; sparse at small a0 ⇒ coarse cubic spline). A number forces uniform
#   saveat = T/knots-per-period so the CubicSpline has dense knots (small-a0 2ω-floor source study).
const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))
const FIELD_MODE = Symbol(get(ENV, "EDM_FIELD_MODE", "split"))   # :split → (E,B,E_far,B_far) | :total → (E,B) only (halves VRAM/output)
FIELD_MODE in (:split, :total) || error("EDM_FIELD_MODE must be \"split\" or \"total\", got \"$FIELD_MODE\"")
const SKIP_POST = get(ENV, "EDM_SKIP_POSTPROCESS", "0") == "1"   # field-only: serialize cube + manifest, defer the (CPU/IO) reduction to an async step
const RUN_TAG = get(ENV, "EDM_RUN_TAG", string(uuid4()))   # launcher may pin via EDM_RUN_TAG so .jls/log/manifest share one id
mkpath(OUTDIR)
@info "Thomson (field) run config" RUN_TAG GPU_BACKEND ϕ₀ A0 SYNC FIELD_MODE OUTDIR NX NELEC NSAMPLES SPP NSUBSTEPS
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
pol = Symbol(get(ENV, "EDM_POL", "circular_minus"))   # EDM_POL: :linear | :circular[_plus] | :circular_minus (default matches the LPWA analytic trajectory's spin); recorded in [laser].pol
profile = :gaussian
z_focus = 0.0

@named world = Worldline(:τ, :atomic)

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
    initial_phase = ϕ₀
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)

# Time span
τi = -8τ
τf = 8τ
tspan = (τi, τf)

x⁰ = [τi * c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]
u0 = [sys.x => x⁰, sys.u => u⁰]

prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, u0, tspan, u0_constructor = SVector{8}, fully_determined = true
)
sol0 = solve(prob, Vern9(), reltol = 1.0e-15, abstol = 1.0e-12)

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

# Ensemble solve
N = NELEC
R₀ = Rmax * sunflower(N, 2)
xμ = [[τi * c, r..., 0.0] for r in R₀]

set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

function prob_func(prob, ctx)
    i = ctx.sim_id
    x_new = SVector{4}(xμ[i]...)
    u_new = SVector{4}(c, 0.0, 0.0, 0.0)
    u0, p = set_x(prob, SVector{8}(x_new..., u_new...))
    return remake(prob; u0, p)
end

function abserr(a₀)
    amp = log10(a₀)
    expo = -amp^2 / 27 + 32amp / 27 - 220 / 27
    return 10^expo
end

const ABSTOL = isempty(ABSTOL_ENV) ? abserr(a₀) : parse(Float64, ABSTOL_ENV)
# Optional uniform saveat (= T_laser / knots-per-period) so the trajectory CubicSpline gets dense knots.
# Passed ONLY when the knob is set, so the default path is byte-identical to the production solve
# (no saveat ⇒ Vern9's adaptive output). The solve always steps adaptively to RELTOL/ABSTOL regardless.
const SAVEAT_KW = isempty(INTERP_SAVEAT) ? (;) :
    (; saveat = collect(τi:((2π / ω) / parse(Float64, INTERP_SAVEAT)):τf))
ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
t_trajectories = @elapsed solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    reltol = RELTOL, abstol = ABSTOL, trajectories = N, SAVEAT_KW...
)
@info "trajectories solved" t_trajectories RELTOL ABSTOL knots_per_period = isempty(INTERP_SAVEAT) ? "adaptive" : INTERP_SAVEAT

# Radiation computation
trajs = trajectory_interpolants(solution)

# Screen parameters
const Z = 2.0e5λ
const samples_per_period = SPP
const δt = 2π / ω / samples_per_period
const N_samples = NSAMPLES
const x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)

Nx = NX
Ny = NX

x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)

screen = ObserverScreen(
    LinRange(-25w₀, 25w₀, Nx),
    LinRange(-25w₀, 25w₀, Ny),
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
        title_prefix = "Thomson scattering", fileprefix = "thomson",
    )
    plotfiles = hprod.plots
end

# ── Reproducibility manifest (same schema as thomson_scattering_A.jl) ──
using TOML
using Dates

provenance = run_provenance(;
    run_id = RUN_TAG, gpu_backend = GPU_BACKEND, repo_dir = pkgdir(ElectronDynamicsModels),
    gpu_device = GPU_BACKEND == "cuda" ? CUDA.name(CUDA.device()) : nothing,
)

config = Dict{String, Any}(
    "initial_phase" => ϕ₀,
    "a0" => A0,
    "Nx" => Nx,
    "Ny" => Ny,
    "N" => N,
    "N_samples" => N_samples,
    "samples_per_period" => samples_per_period,
    "n_substeps" => NSUBSTEPS,
    "reltol" => RELTOL,                # ODE-solve tolerances (replay + small-a0 floor study)
    "abstol" => ABSTOL,
    "interp_saveat" => isempty(INTERP_SAVEAT) ? "adaptive" : INTERP_SAVEAT,  # trajectory-spline knots/period

    "mode" => string(FIELD_MODE),      # :split → (E,B,E_far,B_far) | :total → (E,B); mirrors lpwa.jl
    "sync_per_electron" => SYNC,       # replay input: run_spec_from_manifest reads this
    "observable" => "field",          # distinguishes this run from the 4-potential (_A) runs
)

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
    "log" => "run_$(RUN_TAG).log",   # captured by the run wrapper; travels with the run
)
gpu_telem.trace === nothing || (outputs["gpu_trace"] = basename(gpu_telem.trace))
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
)
setup = Dict{String, Any}(
    "τi" => τi,
    "τf" => τf,
    "Rmax" => Rmax,
    "Z" => Z,
)   # input knobs (Nx/Ny/N/N_samples/spp) live in [config]; setup is just the integration window + screen depth

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
