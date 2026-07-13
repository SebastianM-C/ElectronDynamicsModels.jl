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
# The `using` for the unused backend never executes, so each platform only needs its own.
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
const NX = parse(Int, get(ENV, "EDM_NX", "400"))        # screen pixels per side
const NELEC = parse(Int, get(ENV, "EDM_N", "10000"))       # ensemble size
const NSAMPLES = parse(Int, get(ENV, "EDM_NSAMPLES", "8000")) # observer-time samples
const SPP = parse(Int, get(ENV, "EDM_SPP", "16"))        # samples per optical period
const NSUBSTEPS = parse(Int, get(ENV, "EDM_NSUBSTEPS", "1"))   # RK4 substeps per sample
const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))    # normalized vector potential a₀
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))  # false = overlap uploads
const RUN_TAG = get(ENV, "EDM_RUN_TAG", string(uuid4()))   # launcher may pin via EDM_RUN_TAG; shared by .jls/log/manifest
mkpath(OUTDIR)
@info "Thomson run config" RUN_TAG GPU_BACKEND ϕ₀ A0 SYNC OUTDIR NX NELEC NSAMPLES SPP NSUBSTEPS

# Laser parameters
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ
Rmax = 3.25w₀

a₀ = A0

# Laguerre-Gauss mode / pulse parameters (named so the run manifest records
# exactly what was used, with no drift between the call and the manifest).
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

# Single electron solve (for parameter access)
x⁰ = [τi * c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]

u0 = [
    sys.x => x⁰,
    sys.u => u⁰,
]

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
const samples_per_period = SPP         # Nyquist = 8× fundamental at SPP=16
const δt = 2π / ω / samples_per_period
const N_samples = NSAMPLES             # ≈500λ window at the default 8000
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

# @time A_cpu = accumulate_potential(trajs, screen, Tsit5());

# GPU (fully-on-GPU GPUKernelRK4 path)
@time A_rk4 = accumulate_potential(
    trajs, screen, GPUKernelRK4(), gpu_backend;
    n_substeps = NSUBSTEPS, sync_per_electron = SYNC
);

# @info norm(A_cpu - A_rk4) / norm(A_cpu)

A_s = A_rk4

# Serialize the raw 4-potential — the run's archived cube.
datafile = joinpath(OUTDIR, "A_rk4_$(Nx)_N$(N)_Ns$(N_samples)_spp$(samples_per_period)_$(RUN_TAG).jls")
serialize(datafile, A_s)
println("serialized → $datafile")

# ── Harmonic maps: first two harmonics of ω₁ for the four 4-potential components ──
const complabels = ("A⁰", "Aˣ", "Aʸ", "Aᶻ")
const harmonics = (1, 2)

freqs = rfftfreq(N_samples, 1 / δt)              # kept for the per-harmonic title
hbins = harmonic_bins(N_samples, δt, ω, harmonics)
fields = harmonic_maps(A_s, hbins)               # (length(harmonics), 4, Nx, Ny): A⁰ Aˣ Aʸ Aᶻ

# One figure per harmonic — the unified 2×2 grid over the four potential components
# (:seismic, symmetric range). Rendering lives in EDMMakieExt (via any Makie backend).
function plot_harmonic(k, n)
    title = @sprintf("Thomson scattering — %dω₁ (%.3f× fundamental)", n, freqs[hbins[k]] / (ω / 2π))
    out = joinpath(OUTDIR, @sprintf("thomson_scattering_h%d_%s.png", n, RUN_TAG))
    plot_harmonic_grid(
        fields[k, :, :, :], screen.x_grid, screen.y_grid;
        w₀, labels = complabels, colormap = :seismic, colorrange = symmetric_colorrange,
        ncols = 2, panelsize = 340, title, outfile = out,
    )
    println("saved → $out")
    return out
end

plotfiles = [plot_harmonic(k, n) for (k, n) in enumerate(harmonics)]

# 4-potential power spectra, every run (shows which components carry harmonic structure)
psfile = joinpath(OUTDIR, "powspec_$(RUN_TAG).png")
plot_power_spectrum(freqs, power_spectrum(A_s); ω, labels = complabels, title = "Thomson — 4-potential power spectra", outfile = psfile)
println("saved → $psfile")
push!(plotfiles, psfile)

# phase maps ∠A per component at each harmonic (x/w₀, y/w₀), every run
function plot_phase(k, n)
    out = joinpath(OUTDIR, @sprintf("thomson_phase_h%d_%s.png", n, RUN_TAG))
    plot_phase_grid(
        fields[k, :, :, :], screen.x_grid, screen.y_grid;
        w₀, labels = complabels, ncols = 2, panelsize = 340,
        title = @sprintf("Thomson — ∠A at %dω₁", n), outfile = out,
    )
    println("saved → $out")
    return out
end
append!(plotfiles, [plot_phase(k, n) for (k, n) in enumerate(harmonics)])

# ── Reproducibility manifest ──
# Drop a TOML next to the outputs capturing provenance (repo commit, script,
# host, GPU) and the physical setup, so any result can be traced and rerun.
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
    "observable" => "potential",       # 4-potential Aᵘ run (cf. "field" in thomson_scattering.jl)
)

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
    "log" => "run_$(RUN_TAG).log",   # captured by the run wrapper; travels with the run
    "plots" => basename.(plotfiles),
)

# Laser parameters read back from the compiled system (post-initialization
# values), plus the simulation setup, for full reproducibility.
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
