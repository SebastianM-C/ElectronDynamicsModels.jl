# Multi-device reduce self-check on REAL GPUs (the CPU tests can only exercise the same-device branch):
# (1) _device_add! across two devices against the host sum, (2) the threaded permuted download against the
# serial one, (3) accumulate_field_sharded over devices [1, 2] against the single-device call, both reduces.
# Runs as a campaign "cell" (orchestration/campaigns/multidevice_check.sh) or directly:
#   EDM_GPU_BACKEND=cuda julia -t 4 --project=scripts scripts/multidevice_check.jl
using ElectronDynamicsModels, GPUDiagnostics, StaticArrays, DataInterpolations, LinearAlgebra, Random
const EDM = ElectronDynamicsModels
const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "cuda"))
if GPU_BACKEND == "cuda"
    using CUDA; const backend = CUDA.CUDABackend(; always_inline = true)
elseif GPU_BACKEND == "rocm"
    using AMDGPU; const backend = AMDGPU.ROCBackend()
else
    error("EDM_GPU_BACKEND must be cuda or rocm")
end
import KernelAbstractions as KA
nd = gpu_device_count(backend)
println("devices: $nd  threads: $(Threads.nthreads())")
nd >= 2 || error("multidevice_check needs ≥ 2 visible devices")
fails = 0
report(name, ok, detail) = (global fails += ok ? 0 : 1; println(rpad(name, 52), ok ? "OK   " : "FAIL ", detail))
Random.seed!(1)
# (1) device add across devices
gpu_device!(backend, 1); a_h = rand(64, 48, 3, 70); a = KA.allocate(backend, Float64, size(a_h)); copyto!(a, a_h)
gpu_device!(backend, 2); b_h = rand(64, 48, 3, 70); b = KA.allocate(backend, Float64, size(b_h)); copyto!(b, b_h)
KA.synchronize(backend)
EDM._device_add!(backend, 1, a, b)          # current device 2 → adds into device 1's array
gpu_device!(backend, 1); KA.synchronize(backend); a_back = Array(a)
d1 = maximum(abs, a_back .- (a_h .+ b_h)) / maximum(abs, a_h .+ b_h)
report("_device_add! across devices (2 → 1)", d1 == 0, "max rel diff $d1")
# same-device branch
gpu_device!(backend, 2); c = KA.allocate(backend, Float64, size(b_h)); copyto!(c, a_h); KA.synchronize(backend)
EDM._device_add!(backend, 2, c, b); d2 = maximum(abs, Array(c) .- (a_h .+ b_h)) / maximum(abs, a_h .+ b_h)
report("_device_add! same device", d2 == 0, "max rel diff $d2")
# (2) threaded permuted download
gpu_device!(backend, 1)
ser = EDM._download_permuted(a); thr = EDM._download_permuted(a; backend, dev = 1, workers = 3)
report("threaded _download_permuted == serial", ser == thr && ser == permutedims(a_back, (4, 3, 1, 2)), "")
# (3) sharded accumulate vs single device
function analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0, τspan = (0.0, 20.0), N = 600, K = 1.0)
    ts = collect(range(τspan[1], τspan[2], length = N))
    us = [SVector{8}(g * τ, A * sin(Ω * τ), 0.0, vz * τ, g, A * Ω * cos(Ω * τ), 0.0, vz) for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    as = [SVector{4}(0.0, -A * Ω^2 * sin(Ω * τ), 0.0, 0.0) for τ in ts]
    a_itp = CubicSpline(as, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4), SVector{4, Int}(5, 6, 7, 8), K)
end
trajs = [analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0), analytic_traj(; g = 1.3, A = 0.20, Ω = 2.5), analytic_traj(; g = 1.1, A = 0.30, Ω = 1.5)]
screen = ObserverScreen(LinRange(-1.0, 1.0, 24), LinRange(-1.0, 1.0, 24), 30.0, range(30.0, 50.0; length = 96); c = 1.0)
rel_l2(x, y) = norm(x .- y) / norm(y)
for (alg, kw) in ((GPUKernelNewton(), (; n_iters = 2)), (GPUKernelRK4(), (; n_substeps = 1))), mode in (Val(:total), Val(:split))
    gpu_device!(backend, 1)
    one = accumulate_field(trajs, screen, alg, backend; mode, kw...)
    for reduce in (:device, :host)
        shd = accumulate_field_sharded(trajs, screen, alg, backend; devices = [1, 2], mode, reduce, reduce_workers = 3, kw...)
        worst = maximum(rel_l2(getproperty(shd, k), getproperty(one, k)) for k in propertynames(one))
        report("sharded [1,2] $(nameof(typeof(alg))) $(mode) reduce=$reduce", worst < 1e-12, "worst rel L2 $worst")
    end
end
println(fails == 0 ? "MULTIDEVICE CHECK PASSED" : "MULTIDEVICE CHECK FAILED ($fails)")
exit(fails == 0 ? 0 : 1)
