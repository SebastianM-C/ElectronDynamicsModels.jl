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
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "true"))
const RUN_TAG = string(uuid4())
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

# ── Harmonic maps of the total field: Re(Ẽ), Re(B̃) at 1ω₁, 2ω₁ ──
const complabels = ("Eˣ", "Eʸ", "Eᶻ", "Bˣ", "Bʸ", "Bᶻ")
const harmonics = (1, 2)

freqs = rfftfreq(N_samples, 1 / δt)              # kept for the per-harmonic title
hbins = harmonic_bins(N_samples, δt, ω, harmonics)
# fields_h[k, c, :, :]: harmonic k, component c = (Eˣ,Eʸ,Eᶻ,Bˣ,Bʸ,Bᶻ) — E in 1:3, B in 4:6.
fields_h = harmonic_maps(fld, hbins)

# One figure per harmonic — the unified 2×3 E/B grid (diverging :seismic, symmetric range),
# raw a.u. axes. Rendering lives in EDMPlotsExt, active via `using CairoMakie`.
function plot_harmonic(k, n)
    title = @sprintf("Thomson scattering (field) — %dω₁ (%.3f× fundamental)", n, freqs[hbins[k]] / (ω / 2π))
    out = joinpath(OUTDIR, @sprintf("thomson_field_h%d_%s.png", n, RUN_TAG))
    plot_harmonic_grid(
        fields_h[k, :, :, :], screen.x_grid, screen.y_grid;
        w₀, labels = complabels, colormap = :seismic, colorrange = symmetric_colorrange, title, outfile = out,
    )
    println("saved → $out")
    return out
end

plotfiles = [plot_harmonic(k, n) for (k, n) in enumerate(harmonics)]

# field-component power spectra, every run (shows which components carry harmonic structure)
psfile = joinpath(OUTDIR, "powspec_$(RUN_TAG).png")
plot_power_spectrum(freqs, power_spectrum(fld); ω, labels = complabels, title = "Thomson — field power spectra", outfile = psfile)
println("saved → $psfile")
push!(plotfiles, psfile)

# phase maps ∠F per component at each harmonic (x/w₀, y/w₀), every run
function plot_phase(k, n)
    out = joinpath(OUTDIR, @sprintf("thomson_phase_h%d_%s.png", n, RUN_TAG))
    plot_phase_grid(
        fields_h[k, :, :, :], screen.x_grid, screen.y_grid;
        w₀, labels = complabels, title = @sprintf("Thomson (field) — ∠F at %dω₁", n), outfile = out,
    )
    println("saved → $out")
    return out
end
append!(plotfiles, [plot_phase(k, n) for (k, n) in enumerate(harmonics)])

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
