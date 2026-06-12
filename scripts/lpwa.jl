using LinearAlgebra
using HypergeometricFunctions: pochhammer, _₁F₁
using SpecialFunctions
using StaticArrays
using DataInterpolations
using UUIDs
using ElectronDynamicsModels
using Serialization

using RunManifests
include(joinpath(@__DIR__, "harmonic_products.jl"))   # write_harmonic_products (shared with thomson + recovery)

const ϕ₀ = parse(Float64, get(ENV, "EDM_INITIAL_PHASE", "0.0"))
const OUTDIR = get(ENV, "EDM_OUTDIR", ".")
mkpath(OUTDIR)   # fail-fast at the top, never after the (expensive) accumulation
const NX = parse(Int, get(ENV, "EDM_NX", "400"))
const NELEC = parse(Int, get(ENV, "EDM_N", "1000"))
const NSAMPLES = parse(Int, get(ENV, "EDM_NSAMPLES", "6000"))
const SPP = parse(Int, get(ENV, "EDM_SPP", "16"))
const NSUBSTEPS = parse(Int, get(ENV, "EDM_NSUBSTEPS", "1"))
const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))
const RUN_TAG = get(ENV, "EDM_RUN_TAG", string(uuid4()))   # launcher may pin via EDM_RUN_TAG so .jls/log/manifest share one id

const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "cuda"))
if GPU_BACKEND == "cuda"
    using CUDA
    const gpu_backend = CUDA.CUDABackend()
elseif GPU_BACKEND == "rocm"
    using AMDGPU
    const gpu_backend = AMDGPU.ROCBackend()
else
    error("EDM_GPU_BACKEND must be \"cuda\" or \"rocm\", got $(repr(GPU_BACKEND))")
end

# Reproducibility guard: never run from uncommitted (tracked) code — the manifest's
# repo_commit must reproduce this run. Set EDM_ALLOW_DIRTY=1 to override for throwaway runs.
const REPO_DIR = pkgdir(ElectronDynamicsModels)
assert_committed(REPO_DIR)

a₀ = A0
c = 137.03599908330932
qme = -1.0
p = 2
m = -2
ω = 0.057
λ = c * 2π / ω
w₀ = 75λ
s = 150

A₀ = a₀ * c / qme * sqrt(pochhammer(p + 1, abs(m))) / √2
#
A(ρ) = A₀ * (√2 * ρ / w₀)^abs(m) * _₁F₁(-p, abs(m) + 1, 2 * (ρ / w₀)^2) * exp(-(ρ / w₀)^2)

function trajectory(τ, ℜ₀)
    x₀, y₀, z₀ = ℜ₀
    ρ₀ = norm(ℜ₀)
    φ = -m * atan(y₀, x₀) + ϕ₀

    k = ω / c
    u⁰ = c

    χ = k * u⁰ * τ

    Δx = inv(k) * (A(ρ₀) * qme / c) * s * exp(-(χ / s)^2) * real(im * cis(φ + χ) * dawson(s / 2 + im * χ / s))
    ẋ = -u⁰ * (A(ρ₀) * qme / c) * exp(-(χ / s)^2) * cos(φ + χ)
    Δy = inv(k) * (A(ρ₀) * qme / c) * s * exp(-(χ / s)^2) * real(cis(φ + χ) * dawson(s / 2 + im * χ / s))
    ẏ = -u⁰ * (A(ρ₀) * qme / c) * exp(-(χ / s)^2) * sin(φ + χ)
    Δz = inv(2k) * (A(ρ₀) * qme / c)^2 * s / 2 * sqrt(π / 2) * (1 + erf(sqrt(2) * χ / s))
    ż = u⁰ / 2 * (A(ρ₀) * qme / c)^2 * exp(-2 * (χ / s)^2)

    # (u⁰)² − (uˣ² + uʸ² + uᶻ²) = c²
    # u⁰ = √(c² + |u|²)
    # u⁰ − uᶻ = c   for plane wave
    cγ = c + ż
    x⁰ = c * τ + Δz

    return SVector{8}(x⁰, x₀ + Δx, y₀ + Δy, z₀ + Δz, cγ, ẋ, ẏ, ż)
end

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

N = NELEC
Rmax = 3.25w₀

R₀ = Rmax * sunflower(N, 2)
xi = [[r..., 0.0] for r in R₀]

Nτ = 10_000
τ = 150 / ω
τs = collect(range(-8τ, 8τ, length = Nτ))

using PhysicalConstants.CODATA2018: ε_0
using UnitfulAtomic

x_idxs = SA[1, 2, 3, 4]
u_idxs = SA[5, 6, 7, 8]
q_e = -1
ε₀ = austrip(ε_0)
K = q_e / (4π * ε₀ * c)

τi = -8τ
τf = 8τ

# verify trajectory
for f in (0.0, 11, 120)
    tau = f * λ / c
    δtau = 1.0e-6 * λ / c
    𝔯 = w₀ * [rand(2); 0.0]

    t2 = trajectory(tau + δtau, 𝔯)
    t1 = trajectory(tau, 𝔯)

    t′ = (t2 - t1) / δtau

    @assert all(isapprox.(t′[1:4], t1[5:8], rtol = 1.0e-4, atol = 1.0e-8))
end

function build_traj(ℜ₀)
    us = [trajectory(τₚ, ℜ₀) for τₚ in τs]
    itp = CubicSpline(us, τs; extrapolation = ExtrapolationType.Extension)
    a_itp = CubicSpline(
        [DataInterpolations.derivative(itp, τₚ)[u_idxs] for τₚ in τs], τs;
        extrapolation = ExtrapolationType.Extension
    )
    return ElectronDynamicsModels.TrajectoryInterpolant(itp, a_itp, x_idxs, u_idxs, K)
end

trajs = Vector{Any}(undef, N)
Threads.@threads for i in 1:N
    trajs[i] = build_traj(xi[i])
end
trajs = identity.(trajs)

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
# mode = Val(:total) returns just (; E, B), each (N_samples, 3, Nx, Ny) — the total
# field, summed from the far (1/R) and near (1/R²) pieces on the device.
# (Switch to Val(:split) to also recover E_far, B_far for far-field-only diagnostics.)
@time fld = accumulate_field(
    trajs, screen, GPUKernelRK4(), gpu_backend;
    mode = Val(:total),
    n_substeps = NSUBSTEPS, sync_per_electron = SYNC
)

datafile = joinpath(OUTDIR, "field_$(Nx)_N$(N)_Ns$(N_samples)_spp$(samples_per_period)_$(RUN_TAG).jls")
serialize(datafile, fld)
println("serialized → $datafile")

# ── Harmonic maps + ∠F phase + power spectrum (reduce + serialize + plot) ──
# Shared with thomson_scattering.jl + the recovery path via harmonic_products.jl, so the
# LPWA maps come from exactly the same code as the ODE-solved run they're compared against.
hprod = write_harmonic_products(
    fld, screen.x_grid, screen.y_grid, ω, δt;
    w₀, run_tag = RUN_TAG, outdir = OUTDIR,
    title_prefix = "LPWA", fileprefix = "lpwa",
)
plotfiles = hprod.plotfiles

# ── Reproducibility manifest (mirrors thomson_scattering.jl; analytic-LPWA variant,
# so model params come from the script globals rather than an MTK `prob`). The git
# capture + provenance block are shared via manifest.jl's run_provenance. ──
provenance = run_provenance(;
    run_id = RUN_TAG, gpu_backend = GPU_BACKEND, repo_dir = REPO_DIR,
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
    "sync_per_electron" => SYNC,
    "mode" => "total",
    "observable" => "field",
    "trajectory_source" => "lpwa_analytic",   # distinguishes from the ODE-solved field runs
)

# Beam/laser parameters → [laser]. The dashboard's PARAM_SPEC reads beam params
# (wavelength, w0, p, m, pol, profile, …) from [laser]; emitting them here — instead of
# the old [model] section — is what lets them line up with the ODE-solved
# thomson_scattering.jl runs in the compare view. Unlike thomson these come straight from
# the script globals (there is no MTK `prob.ps` to read).
laser_params = Dict{String, Any}(
    "wavelength" => λ,
    "w0" => w₀,
    "p" => p,
    "m" => m,
    "profile" => "gaussian",
    "pol" => "circular",
    "a0" => A0,
    "phi0" => ϕ₀,
)

# lpwa-only model bookkeeping that is NOT a shared comparison axis (so it does not belong
# in [laser]). The dashboard builder ignores [model]; these stay purely for
# reproducibility / inspection of the analytic trajectory.
model_params = Dict{String, Any}(
    "pulse_width" => s,
    "amplitude" => A₀,
    "trajectory_sample_points" => Nτ,
)

setup = Dict{String, Any}(
    "τi" => τi,
    "τf" => τf,
    "Rmax" => Rmax,
    "Z" => Z,
)   # input knobs (Nx/Ny/N/N_samples/spp) live in [config]; setup is just the integration window + screen depth

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
    "log" => "run_$(RUN_TAG).log",   # captured by the run wrapper; travels with the run
    "harmonic_maps" => basename(hprod.hmapsfile),
    "plots" => basename.(plotfiles),
)

manifestfile = write_solver_manifest(
    OUTDIR; run_id = RUN_TAG, provenance, config, laser = laser_params, setup, outputs,
    extra = Dict("model" => model_params),
)
println("manifest → $manifestfile")
