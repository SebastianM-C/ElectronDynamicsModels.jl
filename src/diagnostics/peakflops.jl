# Attainable vector FP64 peak of a backend, MEASURED — the denominator of the manifest's
# `[flops].peak_fraction_field`. No per-architecture table: every thread runs several
# independent chains of dependent FMAs (the instruction the production kernels are made of), the
# launch is sized to fill the device, and the best of a few trials is the ceiling scalar FP64
# code can attain on that device at the clocks it actually holds. A matrix-multiply probe would
# be routed to the FP64 tensor/matrix units on datacenter parts (2× the vector rate on A100/H100/
# MI300X) — the wrong yardstick for scalar kernels — which is why the probe is an FMA chain.
#
# Latency hiding: `CHAINS` independent accumulators per thread × the resident threads per SM
# (≥ 1024 at the default launch) ≫ FP64 lanes × FMA latency (64 × ~8 on Hopper), so the vector
# pipes stay full without relying on cross-thread scheduling. The multiplier/addend derive from
# a per-thread seed so nothing folds at compile time, and the recurrence converges (|x| < 1) so
# no overflow or denormal slow paths distort the timing. The CPU backend uses BLAS `peakflops`
# instead (the SIMD gemm peak is the honest host ceiling; a per-workitem scalar chain is not).

const _PEAK_CHAINS = 8

@kernel function _fma_chain_kernel!(out, @Const(seed), n_iters)
    i = @index(Global, Linear)
    @inbounds begin
        s = seed[i]
        x = 0.999_999 - 1.0e-9 * s      # |x| < 1: the recurrence converges to y / (1 − x)
        y = 1.0e-3 + 1.0e-9 * s
        a1 = s; a2 = s + 1.0; a3 = s + 2.0; a4 = s + 3.0
        a5 = s + 4.0; a6 = s + 5.0; a7 = s + 6.0; a8 = s + 7.0
        for _ in 1:n_iters
            a1 = fma(a1, x, y); a2 = fma(a2, x, y); a3 = fma(a3, x, y); a4 = fma(a4, x, y)
            a5 = fma(a5, x, y); a6 = fma(a6, x, y); a7 = fma(a7, x, y); a8 = fma(a8, x, y)
        end
        out[i] = ((a1 + a2) + (a3 + a4)) + ((a5 + a6) + (a7 + a8))
    end
end

# Host reference of one thread's result (the test oracle: proves the kernel ran the chains it is
# credited with).
function _fma_chain_reference(s::Float64, n_iters::Integer)
    x = 0.999_999 - 1.0e-9 * s
    y = 1.0e-3 + 1.0e-9 * s
    a = ntuple(k -> s + (k - 1), _PEAK_CHAINS)
    for _ in 1:n_iters
        a = map(v -> fma(v, x, y), a)
    end
    return ((a[1] + a[2]) + (a[3] + a[4])) + ((a[5] + a[6]) + (a[7] + a[8]))
end

function _fma_chain_run!(backend, out, seed, n_iters, workgroup)
    kern = _fma_chain_kernel!(backend, workgroup)
    KernelAbstractions.synchronize(backend)
    t0 = time_ns()
    kern(out, seed, Int32(n_iters); ndrange = length(out))
    KernelAbstractions.synchronize(backend)
    return (time_ns() - t0) * 1.0e-9
end

"""
    measure_peak_fp64_flops(backend; n_threads = 2^20, workgroup = 256, trials = 5,
                            target_seconds = 0.2) -> Float64

Attainable vector FP64 peak of `backend` in FLOP/s, measured with a dependent-FMA-chain kernel
(`$(_PEAK_CHAINS)` independent chains per thread over `n_threads` threads). The chain length is
calibrated so one launch lasts about `target_seconds`, then the best of `trials` launches is
returned (best-of, like `LinearAlgebra.peakflops`, discards clock-ramp and scheduling noise).
FLOP = threads × chains × iterations × 2 (one FMA = 2 FLOP). Never routed to matrix/tensor
units: this is the ceiling for scalar FP64 code such as the field kernels. Costs about
`(trials + 2) × target_seconds` of device time.
"""
function measure_peak_fp64_flops(backend::Backend; n_threads::Integer = 2^20, workgroup::Integer = 256,
        trials::Integer = 5, target_seconds::Real = 0.2)
    n_threads > 0 && workgroup > 0 && trials > 0 && target_seconds > 0 ||
        throw(ArgumentError("measure_peak_fp64_flops: n_threads, workgroup, trials, target_seconds must be > 0"))
    seed = Adapt.adapt(backend, Float64.(0:(n_threads - 1)))
    out = Adapt.adapt(backend, zeros(Float64, n_threads))
    flop(n_iters) = 2.0 * _PEAK_CHAINS * n_iters * n_threads
    # Calibrate: a short launch (also the JIT warm-up), then size n_iters to ≈ target_seconds.
    n0 = 256
    _fma_chain_run!(backend, out, seed, n0, workgroup)                     # warm-up (compile)
    t0 = _fma_chain_run!(backend, out, seed, n0, workgroup)
    n_iters = clamp(round(Int, n0 * target_seconds / max(t0, 1.0e-6)), n0, typemax(Int32) ÷ 2)
    best = Inf
    for _ in 1:trials
        best = min(best, _fma_chain_run!(backend, out, seed, n_iters, workgroup))
    end
    # Sanity: the device computed the chains it is credited with (guards a mis-launched kernel
    # or a compiler that hoisted the loop).
    h = Array(out)
    ref = _fma_chain_reference(0.0, n_iters)
    isapprox(h[1], ref; rtol = 1.0e-9) ||
        error("measure_peak_fp64_flops: kernel result $(h[1]) ≠ host reference $ref — the probe did not run as credited")
    return flop(n_iters) / best
end
