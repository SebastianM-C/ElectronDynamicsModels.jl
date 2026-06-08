using LinearAlgebra
using HypergeometricFunctions: pochhammer, _₁F₁
using SpecialFunctions
using StaticArrays
using DataInterpolations
using UUIDs
using ElectronDynamicsModels
using Serialization
using FFTW
using CairoMakie
using Printf

include(joinpath(@__DIR__, "manifest.jl"))   # shared reproducibility / provenance helpers

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
const RUN_TAG = string(uuid4())

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
    φ = m * atan(y₀, x₀) + ϕ₀

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

function build_traj(ℜ₀)
    us = [SVector{8}(trajectory(τₚ, ℜ₀)) for τₚ in τs]
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

# ── Harmonic maps of the total field: Re(Ẽ), Re(B̃) at 1ω₁, 2ω₁ ──
# rfft one component at a time (memory-conscious), mirroring thomson_scattering.jl so
# the LPWA maps are directly comparable to the ODE-solved run. fld holds only (; E, B)
# under Val(:total), which is exactly what these maps consume.
const complabels = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")
const harmonics = (1, 2)

freqs = rfftfreq(N_samples, 1 / δt)                  # kept for the per-harmonic title
hbins = harmonic_bins(N_samples, δt, ω, harmonics)
# fields_h[k, c, :, :]: harmonic k, component c = (Eˣ,Eʸ,Eᶻ,Bˣ,Bʸ,Bᶻ) — E in 1:3, B in 4:6.
fields_h = harmonic_maps(fld, hbins)

# reduced harmonic maps — re-plottable without the raw cube
hmapsfile = joinpath(OUTDIR, "hmaps_$(RUN_TAG).jls")
serialize(
    hmapsfile,
    (; fields_h, harmonics, ffund = [freqs[b] / (ω / 2π) for b in hbins],
        x_grid = collect(screen.x_grid), y_grid = collect(screen.y_grid), w₀),
)
println("serialized harmonic maps → $hmapsfile")

# One figure per harmonic — the unified 2×3 E/B grid (defaults: :jet, per-panel extrema),
# axes in units of w₀. Rendering lives in EDMPlotsExt, active via `using CairoMakie`.
function plot_harmonic(k, n)
    title = @sprintf("LPWA (field) — %dω₁ (%.3f× fundamental)", n, freqs[hbins[k]] / (ω / 2π))
    out = joinpath(OUTDIR, @sprintf("lpwa_field_h%d_%s.png", n, RUN_TAG))
    plot_harmonic_grid(
        fields_h[k, :, :, :], screen.x_grid, screen.y_grid;
        w₀, labels = complabels, title, outfile = out,
    )
    println("saved → $out")
    return out
end

plotfiles = [plot_harmonic(k, n) for (k, n) in enumerate(harmonics)]

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

# Physics parameters of the analytic LPWA model. Unlike thomson_scattering.jl these come
# straight from the script globals (there is no MTK `prob.ps` to read). Record exactly
# what's needed to reconstruct the trajectory + screen.
model_params = Dict{String, Any}(
    "wavelength" => λ,
    "a0" => A0,
    "w0" => w₀,
    "p" => p,
    "m" => m,
    "pulse_width" => s,
    "phi0" => ϕ₀,
    "amplitude" => A₀,
    "trajectory_sample_points" => Nτ,
)

setup = Dict{String, Any}(
    "τi" => τi,
    "τf" => τf,
    "Rmax" => Rmax,
    "N" => N,
    "Z" => Z,
    "samples_per_period" => samples_per_period,
    "N_samples" => N_samples,
    "Nx" => Nx,
    "Ny" => Ny,
)

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
    "harmonic_maps" => basename(hmapsfile),
    "plots" => basename.(plotfiles),
)

manifest = Dict{String, Any}(
    "provenance" => provenance,
    "config" => config,
    "model" => model_params,
    "setup" => setup,
    "outputs" => outputs,
)

manifestfile = joinpath(OUTDIR, "run_$(RUN_TAG).toml")
open(io -> TOML.print(io, manifest; sorted = true), manifestfile, "w")
println("manifest → $manifestfile")
