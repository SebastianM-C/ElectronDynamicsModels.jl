# Occupancy + telemetry bench for accumulate_field at PRODUCTION screen size. Occupancy is set by
# the per-launch thread count (Nx·Ny), not the electron count — so a handful of electrons (EDM_N,
# default 20) measures the prod-size kernel cheaply. Reports the STATIC thread-fill occupancy, plus
# power / utilization sampled — on a SEPARATE OS THREAD — DURING the accumulate_field run.
#
#   EDM_GPU_BACKEND=cuda EDM_NX=400 EDM_NSAMPLES=6000 EDM_N=20 \
#     julia +release -t2 --startup=no --project=scripts scripts/occupancy_bench.jl
#
# Needs ≥2 Julia threads (-t2): the sampler runs on a spawned thread while the main thread drives the
# launch-bound accumulate_field loop (which never yields, so a cooperative @async sampler would
# starve). NVML/sysfs reads are host-side + thread-safe and GPU work is async, so they overlap.
# Setup mirrors thomson_scattering.jl (LaguerreGauss circular beam, sunflower electrons); the bench
# does NOT serialize the (prod-size, ~46–92 GB) field cube — it only times the kernel + telemetry.

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqNonlinearSolve
using SciMLBase, StaticArrays, SymbolicIndexingInterface
using Statistics: mean
using Printf

const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "cuda"))
if GPU_BACKEND == "cuda"
    using CUDA
    const backend = CUDA.CUDABackend()
elseif GPU_BACKEND == "rocm"
    using AMDGPU
    const backend = AMDGPU.ROCBackend()
else
    error("EDM_GPU_BACKEND must be \"cuda\" or \"rocm\", got $(repr(GPU_BACKEND))")
end

const c = 137.03599908330932
const NX = parse(Int, get(ENV, "EDM_NX", "400"))
const NSAMPLES = parse(Int, get(ENV, "EDM_NSAMPLES", "6000"))
const NELEC = parse(Int, get(ENV, "EDM_N", "20"))
const SPP = parse(Int, get(ENV, "EDM_SPP", "16"))
const A0 = parse(Float64, get(ENV, "EDM_A0", "0.1"))
const NSUBSTEPS = parse(Int, get(ENV, "EDM_NSUBSTEPS", "1"))
const SAMPLE_DT = parse(Float64, get(ENV, "EDM_SAMPLE_DT", "0.1"))

# ── electron trajectories + observer screen (mirrors thomson_scattering.jl) ──
ω = 0.057
τ = 150 / ω
λ = 2π * c / ω
w₀ = 75λ
Rmax = 3.25w₀
@named world = Worldline(:τ, :atomic)
@named laser = LaguerreGaussLaser(;
    wavelength = λ, a0 = A0, beam_waist = w₀, radial_index = 2, azimuthal_index = -2,
    world, temporal_profile = :gaussian, temporal_width = τ, focus_position = 0.0,
    polarization = :circular_minus, initial_phase = 0.0,
)
@named elec = ClassicalElectron(; laser)
sys = mtkcompile(elec)
τi = -8τ
τf = 8τ
x⁰ = [τi * c, 0.0, 0.0, 0.0]
u⁰ = [c, 0.0, 0.0, 0.0]
prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, [sys.x => x⁰, sys.u => u⁰], (τi, τf); u0_constructor = SVector{8}, fully_determined = true)

const φ = (1 + √5) / 2
radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
function sunflower(n, α)
    pts = Vector{Vector{Float64}}()
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        θ = k * 2π / φ^2
        push!(pts, [r * cos(θ), r * sin(θ)])
    end
    return pts
end
R₀ = Rmax * sunflower(NELEC, 2)
xμ = [[τi * c, r..., 0.0] for r in R₀]
set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])
function prob_func(prob, ctx)
    x_new = SVector{4}(xμ[ctx.sim_id]...)
    u0, p = set_x(prob, SVector{8}(x_new..., c, 0.0, 0.0, 0.0))
    return remake(prob; u0, p)
end
ensemble = EnsembleProblem(prob; prob_func, safetycopy = false)
println("solving $NELEC trajectories …")
solution = solve(ensemble, Vern9(), EnsembleThreads(); reltol = 1.0e-12, abstol = 1.0e-10, trajectories = NELEC)
trajs = trajectory_interpolants(solution)

Z = 2.0e5λ
δt = 2π / ω / SPP
x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)
x⁰_samples = range(; start = x⁰_start, step = c * δt, length = NSAMPLES)
screen = ObserverScreen(LinRange(-25w₀, 25w₀, NX), LinRange(-25w₀, 25w₀, NX), Z, x⁰_samples; c)

# ── static thread-fill occupancy (no run needed; just launch size ÷ device capacity) ──
cap = gpu_sm_count(backend) * gpu_max_threads_per_sm(backend)
nthr = NX * NX
@printf("device: %s  (%d SMs × %d thr = %d capacity)\n",
    gpu_name(backend), gpu_sm_count(backend), gpu_max_threads_per_sm(backend), cap)
@printf("thread-fill occupancy @ NX=%d (%d threads/electron) = %.3f  (>1 ⇒ waves)\n",
    NX, nthr, thread_fill_occupancy(backend, nthr))

# ── telemetry sampler on a spawned OS thread (power/util DURING the run) ──
Threads.nthreads() >= 2 ||
    @warn "only $(Threads.nthreads()) Julia thread — the sampler will starve; rerun with -t2"
const running = Threads.Atomic{Bool}(true)
const samples = NTuple{4, Float64}[]   # (t, power_W, compute_util, mem_util); single writer (this task)
t0 = time()
sampler = Threads.@spawn begin
    while running[]
        u = gpu_utilization(backend)
        push!(samples, (time() - t0, gpu_power(backend), Float64(u.compute), Float64(u.memory)))
        sleep(SAMPLE_DT)
    end
end

# ── the measured run (split mode, like prod; no serialization — bench only) ──
GC.gc()
println("accumulating field ($NELEC electrons, NX=$NX, Ns=$NSAMPLES, split) …")
t_field = @elapsed fld = accumulate_field(
    trajs, screen, GPUKernelRK4(), backend; n_substeps = NSUBSTEPS, mode = Val(:split), sync_per_electron = false)
running[] = false
wait(sampler)

# ── report ──
pw = Float64[s[2] for s in samples]
cu = Float64[s[3] for s in samples]
mu = Float64[s[4] for s in samples]
@printf("\nrun: %d electrons in %.1f s  (%d telemetry samples @ %.2gs)\n", NELEC, t_field, length(samples), SAMPLE_DT)
if !isempty(pw)
    @printf("power (W):     mean %.1f   peak %.1f   min %.1f\n", mean(pw), maximum(pw), minimum(pw))
    @printf("compute util:  mean %.2f   peak %.2f\n", mean(cu), maximum(cu))
    @printf("memory util:   mean %.2f   peak %.2f\n", mean(mu), maximum(mu))
end
let m = gpu_memory_info(backend)
    @printf("VRAM: used %.1f / %.1f GB\n", m.used / 2^30, m.total / 2^30)
end
