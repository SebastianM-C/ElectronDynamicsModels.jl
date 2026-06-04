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
const SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "true"))  # false = overlap uploads
const RUN_TAG = string(uuid4())   # unique per run; params live in the TOML manifest
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

# Serialize the raw 4-potential so the offline scripts (plot_harmonics.jl,
# plot_power_spectrum.jl) can read this run directly.
datafile = joinpath(OUTDIR, "A_rk4_$(Nx)_N$(N)_Ns$(N_samples)_spp$(samples_per_period)_$(RUN_TAG).jls")
serialize(datafile, A_s)
println("serialized → $datafile")

# ── Harmonic maps ──
# Extract the first two harmonics of ω₁ for all four 4-potential components.
# rfft one component at a time (the full complex spectrum is 4× the raw array),
# mirroring plot_harmonics.jl's memory-conscious slicing.
const complabels = ("A⁰", "Aˣ", "Aʸ", "Aᶻ")
const harmonics = (1, 2)

freqs = rfftfreq(N_samples, 1 / δt)
harmonic_bins = [findmin(x -> abs(x - n * ω / 2π), freqs)[2] for n in harmonics]

fields = Array{ComplexF64, 4}(undef, length(harmonics), 4, Nx, Ny)
for μ in 1:4
    A_ω_c = rfft(A_s[:, μ, :, :], 1)
    for (k, idx) in enumerate(harmonic_bins)
        fields[k, μ, :, :] = A_ω_c[idx, :, :]
    end
    A_ω_c = nothing
    GC.gc()
end

# One figure per harmonic, 2×2 over the four components; each panel scaled to its
# own peak since component amplitudes differ by orders.
function plot_harmonic(k, n)
    idx = harmonic_bins[k]
    fig = Figure()
    Label(
        fig[0, :], @sprintf(
            "Thomson scattering — %dω₁ (%.3f× fundamental)",
            n, freqs[idx] / (ω / 2π)
        ), fontsize = 16, font = :bold
    )
    for μ in 1:4
        field = real.(fields[k, μ, :, :])
        cr = maximum(abs, field)
        gl = fig[cld(μ, 2), (μ - 1) % 2 + 1] = GridLayout()
        ax = Axis(
            gl[1, 1], width = 340, height = 340, xlabel = "x", ylabel = "y",
            title = @sprintf("%s  (peak %.2e)", complabels[μ], cr)
        )
        hm = heatmap!(
            ax, collect(screen.x_grid), collect(screen.y_grid), field,
            colorrange = (-cr, cr), colormap = :seismic
        )
        Colorbar(gl[1, 2], hm, width = 12, height = 340)
    end
    resize_to_layout!(fig)
    out = joinpath(OUTDIR, @sprintf("thomson_scattering_h%d_%s.png", n, RUN_TAG))
    save(out, fig)
    println("saved → $out")
    return fig
end

plotfiles = String[]
for (k, n) in enumerate(harmonics)
    plot_harmonic(k, n)
    push!(plotfiles, joinpath(OUTDIR, @sprintf("thomson_scattering_h%d_%s.png", n, RUN_TAG)))
end

# ── Reproducibility manifest ──
# Drop a TOML next to the outputs capturing provenance (repo commit, script,
# host, GPU) and the physical setup, so any result can be traced and rerun.
using TOML
using Dates

const _edm_dir = pkgdir(ElectronDynamicsModels)
_git(args...) = try
    readchomp(Cmd(["git", "-C", string(_edm_dir), args...]))
catch
    "unknown"
end
repo_status = _git("status", "--porcelain")

provenance = Dict{String, Any}(
    "run_id" => RUN_TAG,
    "repo_commit" => _git("rev-parse", "HEAD"),
    "repo_dirty" => !(repo_status == "" || repo_status == "unknown"),
    "edm_pkgdir" => string(_edm_dir),
    "script" => abspath(PROGRAM_FILE),
    "host" => gethostname(),
    "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
    "gpu_backend" => GPU_BACKEND,
    "julia_version" => string(VERSION),
    "timestamp" => string(now()),
)
if GPU_BACKEND == "cuda"
    provenance["gpu_device"] = string(CUDA.name(CUDA.device()))
end

config = Dict{String, Any}(
    "initial_phase" => ϕ₀,
    "a0" => A0,
    "Nx" => Nx,
    "Ny" => Ny,
    "N" => N,
    "N_samples" => N_samples,
    "samples_per_period" => samples_per_period,
    "n_substeps" => NSUBSTEPS,
)

outputs = Dict{String, Any}(
    "datafile" => basename(datafile),
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
    "N" => N,
    "Z" => Z,
    "samples_per_period" => samples_per_period,
    "N_samples" => N_samples,
    "Nx" => Nx,
    "Ny" => Ny,
)

manifest = Dict{String, Any}(
    "provenance" => provenance,
    "config" => config,
    "laser" => laser_params,
    "setup" => setup,
    "outputs" => outputs
)

manifestfile = joinpath(OUTDIR, "run_$(RUN_TAG).toml")
open(io -> TOML.print(io, manifest; sorted = true), manifestfile, "w")
println("manifest → $manifestfile")
