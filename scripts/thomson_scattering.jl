# Field-accumulation Thomson-scattering run: same setup as thomson_scattering_A.jl
# (LaguerreGauss circular beam, sunflower electrons, ENV-driven, GPUKernelRK4) but
# computes the exact radiated field via `accumulate_field` (the split Liénard–
# Wiechert E, B) instead of the 4-potential, and plots Re(Ẽ(ω)), Re(B̃(ω)) harmonic
# maps at the end. The `_A` script remains the 4-potential reference.
#
# ENV knobs (defaults = full production): EDM_GPU_BACKEND (rocm|cuda), EDM_A0,
# EDM_INITIAL_PHASE, EDM_NX, EDM_N, EDM_NSAMPLES, EDM_SPP, EDM_NSUBSTEPS,
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
const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))
const RUN_TAG = get(ENV, "EDM_RUN_TAG", string(uuid4()))   # launcher may pin via EDM_RUN_TAG so .jls/log/manifest share one id
mkpath(OUTDIR)
@info "Thomson (field) run config" RUN_TAG GPU_BACKEND ϕ₀ A0 SYNC OUTDIR NX NELEC NSAMPLES SPP NSUBSTEPS

# Laser parameters
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ
Rmax = 3.25w₀

a₀ = A0

p_radial = 2
m_azimuthal = -2
pol = :circular
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

ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
solution = solve(
    ensemble, Vern9(), EnsembleThreads();
    reltol = 1.0e-12, abstol = abserr(a₀), trajectories = N
)

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
# Returns (; E, B, E_rad, B_rad), each (N_samples, 3, Nx, Ny): E, B are the total
# field (for the harmonic maps below); E_rad, B_rad the radiation field alone.
@time fld = accumulate_field(
    trajs, screen, GPUKernelRK4(), gpu_backend;
    n_substeps = NSUBSTEPS, sync_per_electron = SYNC
)

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
hprod = write_harmonic_products(
    fld, screen.x_grid, screen.y_grid, ω, δt;
    w₀, run_tag = RUN_TAG, outdir = OUTDIR,
    title_prefix = "Thomson scattering", fileprefix = "thomson",
)
plotfiles = hprod.plotfiles

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
    "sync_per_electron" => SYNC,       # replay input: run_spec_from_manifest reads this
    "observable" => "field",          # distinguishes this run from the 4-potential (_A) runs
)

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
    "log" => "run_$(RUN_TAG).log",   # captured by the run wrapper; travels with the run
    "harmonic_maps" => basename(hprod.hmapsfile),   # reduced maps → resolve_hmaps finds them directly
    "plots" => basename.(plotfiles),
)

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

manifestfile = write_solver_manifest(
    OUTDIR; run_id = RUN_TAG, provenance, config, laser = laser_params, setup, outputs
)
println("manifest → $manifestfile")
