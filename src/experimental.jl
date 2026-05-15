using ..ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen, to_gpu
import Adapt
import AcceleratedKernels as AK
import KernelAbstractions
using KernelAbstractions: Backend, @kernel, @index, @Const
using StaticArrays

# ── Unified GPU kernel: retarded-time solve + LW accumulation per electron ──
# Dispatched via `accumulate_potential(trajs, screen, GPUKernelRK4(), backend)`.
# One thread per pixel, fixed-step RK4 at the saveat resolution, immediate
# accumulation into the device output buffer — no τ_all intermediate, no
# CPU-GPU synchronization mid-flight.

"""
    GPUKernelRK4

Sentinel solver type that selects the unified GPU kernel path.
"""
struct GPUKernelRK4 end

# RHS of dτ_r/dt = 1 / (u⁰ - u⃗·n̂); allocation-free, kernel-callable.
@inline function _rt_rhs_kernel(τ, gpu_traj, r_obs)
    v = gpu_traj.itp(τ)
    x¹ = v[gpu_traj.x_idxs[2]]
    x² = v[gpu_traj.x_idxs[3]]
    x³ = v[gpu_traj.x_idxs[4]]
    u⁰ = v[gpu_traj.u_idxs[1]]
    u¹ = v[gpu_traj.u_idxs[2]]
    u² = v[gpu_traj.u_idxs[3]]
    u³ = v[gpu_traj.u_idxs[4]]
    d¹ = r_obs[1] - x¹
    d² = r_obs[2] - x²
    d³ = r_obs[3] - x³
    inv_r = inv(sqrt(d¹ * d¹ + d² * d² + d³ * d³))
    return inv(u⁰ - (u¹ * d¹ + u² * d² + u³ * d³) * inv_r)
end

# One classical RK4 step for the autonomous ODE dτ_r/dt = f(τ_r).
@inline function _rk4_step(τ, dt, gpu_traj, r_obs)
    k1 = _rt_rhs_kernel(τ, gpu_traj, r_obs)
    k2 = _rt_rhs_kernel(τ + 0.5 * dt * k1, gpu_traj, r_obs)
    k3 = _rt_rhs_kernel(τ + 0.5 * dt * k2, gpu_traj, r_obs)
    k4 = _rt_rhs_kernel(τ + dt * k3, gpu_traj, r_obs)
    return τ + dt * (k1 + 2 * (k2 + k3) + k4) * (1 / 6)
end

# Per-electron AK.foreachindex pass over (Nx × Ny) pixels.
# Implemented as a regular function (not @kernel) because KA's @kernel macro
# on Julia 1.12 emits bounds-checked `expand` calls in `@index(Global, *)`
# whose throw path is not GPU-compilable.  AK.foreachindex uses an iteration
# pattern the optimizer can prove out-of-bounds-free, so the bounds-error
# branch is dead-code-eliminated.
function _gpu_unified_one_electron!(
        A_buf, gpu_traj,
        x_grid, y_grid, z_screen,
        x⁰_first, δx⁰, N_samples, Nx, Ny,
        τi, τf, pixel_iter, backend, n_substeps
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Column-major unpacking: ix runs fastest, iy outer
        ix = ((i_lin - 1) % Nx) + 1
        iy = ((i_lin - 1) ÷ Nx) + 1
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Pixel-specific advanced-time window x⁰_i, x⁰_f
        v_i = gpu_traj.itp(τi)
        d_i¹ = r_obs[1] - v_i[gpu_traj.x_idxs[2]]
        d_i² = r_obs[2] - v_i[gpu_traj.x_idxs[3]]
        d_i³ = r_obs[3] - v_i[gpu_traj.x_idxs[4]]
        x⁰_i_px = v_i[gpu_traj.x_idxs[1]] + sqrt(d_i¹^2 + d_i²^2 + d_i³^2)

        v_f = gpu_traj.itp(τf)
        d_f¹ = r_obs[1] - v_f[gpu_traj.x_idxs[2]]
        d_f² = r_obs[2] - v_f[gpu_traj.x_idxs[3]]
        d_f³ = r_obs[3] - v_f[gpu_traj.x_idxs[4]]
        x⁰_f_px = v_f[gpu_traj.x_idxs[1]] + sqrt(d_f¹^2 + d_f²^2 + d_f³^2)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))

        if k_start > k_end
            return
        end

        # TODO(human): bridge τ from τi (the τ at observer time x⁰_i_px) up to
        # observer time x⁰_samples[k_start], and inside the slot loop advance τ
        # by δx⁰ between successive slots — using `n_substeps` RK4 sub-steps
        # rather than a single RK4 step, so that the inner step size dt =
        # δx⁰/n_substeps brings ω·dt down into RK4's accurate range.
        # The single-step version below works for ω·δx⁰ ≪ 1 but fails badly
        # for the relativistic Thomson script (ω·δx⁰ ≈ π/2).
        τ = τi
        bridge_dt = x⁰_first + (k_start - 1) * δx⁰ - x⁰_i_px
        if bridge_dt > 0
            τ = _rk4_step(τ, bridge_dt, gpu_traj, r_obs)
        end

        # March through saveat slots, accumulating at each
        for k in k_start:k_end
            τ_safe = clamp(τ, τi, τf)
            v = gpu_traj.itp(τ_safe)

            x¹ = v[gpu_traj.x_idxs[2]]
            x² = v[gpu_traj.x_idxs[3]]
            x³ = v[gpu_traj.x_idxs[4]]
            u⁰ = v[gpu_traj.u_idxs[1]]
            u¹ = v[gpu_traj.u_idxs[2]]
            u² = v[gpu_traj.u_idxs[3]]
            u³ = v[gpu_traj.u_idxs[4]]

            d¹ = r_obs[1] - x¹
            d² = r_obs[2] - x²
            d³ = r_obs[3] - x³
            r_norm = sqrt(d¹ * d¹ + d² * d² + d³ * d³)
            # m_dot(xr, uμ) with xr = (r_norm, d¹, d², d³)
            xr_dot_u = r_norm * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³)
            coeff = K / xr_dot_u

            @inbounds A_buf[k, 1, ix, iy] += coeff * u⁰
            @inbounds A_buf[k, 2, ix, iy] += coeff * u¹
            @inbounds A_buf[k, 3, ix, iy] += coeff * u²
            @inbounds A_buf[k, 4, ix, iy] += coeff * u³

            if k < k_end
                τ_next = _rk4_step(τ, δx⁰, gpu_traj, r_obs)
                τ = clamp(τ_next, τi, τf)
            end
        end
    end
    return
end

"""
    accumulate_potential(trajs, screen, ::GPUKernelRK4, backend; n_substeps = 1)

Unified GPU path: per-electron kernel launch performs retarded-time RK4
integration *and* Liénard-Wiechert accumulation in one pass.  Compared to
the two-phase AcceleratedKernels path:
  - no `τ_all` intermediate buffer (saves ~`N_samples × Nx × Ny × 8` bytes)
  - no CPU retarded-time solve, so Phase-1 wall time scales with GPU
    rather than CPU thread count
  - accumulation order eliminates the `round(Int, …)` saveat-slot mapping
    that the CPU+AK path needs (slot index `k` is the loop variable).

`n_substeps` controls how many fixed-step RK4 sub-steps are taken between
successive saveat slots (and within the bridge step).  Required when
`ω · δx⁰` lies outside RK4's accurate range — for the relativistic Thomson
script (a₀ = 10, δt = T/4) `ω · δx⁰ ≈ π/2` and `n_substeps = 1` causes
~7%/step amplitude damping; `n_substeps = 8` brings the inner step to
ω·dt ≈ 0.2 and recovers parity with the adaptive-Tsit5 reference.
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelRK4,
        backend::Backend;
        n_substeps::Int = 1,
    )
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)

    A_buf = Adapt.adapt(backend, zeros(N_samples, 4, Nx, Ny))

    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)

    # Iteration target: one element per pixel. Sentinel array; never read.
    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny))

    # Per-electron upload-and-free streaming: each trajectory's spline
    # arrays (~21 MB at N_t = 10⁵) are uploaded just before the kernel
    # launch and freed immediately after.  This keeps device memory
    # bounded by `A_buf + one trajectory + pixel_iter` regardless of
    # N_macro, so we can scale to thousands of electrons on a 16 GB card.
    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        _gpu_unified_one_electron!(
            A_buf, gpu_traj,
            screen.x_grid, screen.y_grid, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_substeps,
        )
        # Wait for the kernel to finish before releasing the trajectory's
        # device buffers — `finalize` invokes the device-array finalizer
        # (which calls `unsafe_free!` on CUDA / `unsafe_free!` on AMDGPU),
        # and freeing while the kernel still reads the buffer would
        # corrupt the read.  Per-iteration sync stalls pipelining, but
        # at ~600 ms per electron the sync cost is negligible.
        KernelAbstractions.synchronize(backend)
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    return Array(A_buf)
end

# ── Batched GPU Tsit5 path: B electrons per kernel launch ──
#
# Streaming + batched architecture for scaling in N_macro:
#   1. Pack B electron splines into flat 1D storage with per-electron offsets,
#      so length variance across electrons doesn't waste memory.
#   2. One kernel launch processes all B electrons. Each pixel-thread loops
#      over the B electrons internally — same (ix, iy) ownership ⇒ no atomics.
#   3. Outer streaming loop chunks N_macro into ⌈N_macro / B⌉ batches; per-batch
#      device memory ≈ B × spline_size, bounded regardless of N_macro.
#
# Tsit5 (5th-order) instead of RK4 (4th-order) — at ω·dt ≈ 0.2 the local error
# drops from O(2.7e-4) to O(5.4e-5), buying back the extra 3 stages and then
# some.  FSAL: k7 = f(τ_new) is reused as k1 of the next step, saving 1 RHS
# evaluation per slot (≈14% of 7 stages).

"""
    GPUKernelTsit5

Sentinel solver type that selects the batched Tsit5 GPU kernel path.
"""
struct GPUKernelTsit5 end

"""
    BatchedGPUSplines{D, V, M, VO, VK}

Flat-storage GPU representation of B trajectory splines.  Each electron's
data lives at indices `offsets[e]+1 : offsets[e]+Ns[e]` in the flat arrays.
The last row of `c1`/`c2` per electron is unused (interval count = Ns[e]-1)
but kept aligned with knot indices for simpler addressing.
"""
struct BatchedGPUSplines{D, V, M, VO, VK}
    t::V             # flat knots, length Σ Ns[e]
    h::V             # flat interval widths, length Σ Ns[e]
    z::M             # (Σ Ns[e], D) second derivatives
    c1::M            # (Σ Ns[e], D) precomputed linear coeffs
    c2::M            # (Σ Ns[e], D) precomputed linear coeffs
    offsets::VO      # (B+1,) — electron e at offsets[e]+1 : offsets[e+1]
    Ns::VO           # (B,)   — actual knot count per electron
    Ks::VK           # (B,)   — Liénard-Wiechert prefactor
    τis::VK          # (B,)   — first knot time
    τfs::VK          # (B,)   — last knot time
    x_idxs::SVector{4, Int}   # shared across batch
    u_idxs::SVector{4, Int}
end

function Adapt.adapt_structure(to, s::BatchedGPUSplines{D}) where {D}
    t = Adapt.adapt(to, s.t)
    z = Adapt.adapt(to, s.z)
    offsets = Adapt.adapt(to, s.offsets)
    Ks = Adapt.adapt(to, s.Ks)
    return BatchedGPUSplines{D, typeof(t), typeof(z), typeof(offsets), typeof(Ks)}(
        t,
        Adapt.adapt(to, s.h),
        z,
        Adapt.adapt(to, s.c1),
        Adapt.adapt(to, s.c2),
        offsets,
        Adapt.adapt(to, s.Ns),
        Ks,
        Adapt.adapt(to, s.τis),
        Adapt.adapt(to, s.τfs),
        s.x_idxs,
        s.u_idxs,
    )
end

"""
    to_batched_cpu(trajs_chunk)

Pack B `TrajectoryInterpolant`s into a host-side `BatchedGPUSplines` (still
`Vector`/`Matrix`-backed).  Adapt.adapt to a backend afterward to upload.
"""
function to_batched_cpu(trajs_chunk::AbstractVector{<:TrajectoryInterpolant})
    B = length(trajs_chunk)
    Ns_vec = [length(traj.itp.t) for traj in trajs_chunk]
    offsets = Vector{Int}(undef, B + 1)
    offsets[1] = 0
    for e in 1:B
        offsets[e + 1] = offsets[e] + Ns_vec[e]
    end
    total_N = offsets[end]
    D = length(first(first(trajs_chunk).itp.u))

    t_flat = Vector{Float64}(undef, total_N)
    h_flat = Vector{Float64}(undef, total_N)
    z_flat = Matrix{Float64}(undef, total_N, D)
    c1_flat = zeros(Float64, total_N, D)   # last row per electron unused; zero is safe
    c2_flat = zeros(Float64, total_N, D)

    Ks = Vector{Float64}(undef, B)
    τis = Vector{Float64}(undef, B)
    τfs = Vector{Float64}(undef, B)

    for (e, traj) in enumerate(trajs_chunk)
        itp = traj.itp
        N = Ns_vec[e]
        base = offsets[e]
        @inbounds for i in 1:N
            t_flat[base + i] = itp.t[i]
            h_flat[base + i] = itp.h[i]
            zi = itp.z[i]
            for d in 1:D
                z_flat[base + i, d] = zi[d]
            end
        end
        # c1[i], c2[i] for interval i = 1..N-1 (last row left zero)
        @inbounds for i in 1:(N - 1)
            hi = itp.h[i + 1]
            inv_hi = inv(hi)
            hi_over_6 = hi / 6
            ui = itp.u[i]
            uip1 = itp.u[i + 1]
            zi = itp.z[i]
            zip1 = itp.z[i + 1]
            for d in 1:D
                c1_flat[base + i, d] = uip1[d] * inv_hi - zip1[d] * hi_over_6
                c2_flat[base + i, d] = ui[d] * inv_hi - zi[d] * hi_over_6
            end
        end
        Ks[e] = traj.K
        τis[e] = itp.t[1]
        τfs[e] = itp.t[N]
    end

    x_idxs = first(trajs_chunk).x_idxs
    u_idxs = first(trajs_chunk).u_idxs
    return BatchedGPUSplines{D, typeof(t_flat), typeof(z_flat), typeof(offsets), typeof(Ks)}(
        t_flat, h_flat, z_flat, c1_flat, c2_flat,
        offsets, Ns_vec, Ks, τis, τfs, x_idxs, u_idxs,
    )
end

# Per-electron binary search inside flat knot array.
# `base = offsets[e]`, `N_e = Ns[e]`. Returns local index in [1, N_e-1].
@inline function _searchsorted_left_batched(t, base, N_e, x)
    lo, hi = 1, N_e - 1
    while lo < hi
        mid = (lo + hi + 1) >> 1
        @inbounds if t[base + mid] ≤ x
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

# Spline evaluation at τ for electron `e` of the batch. Returns SVector{D}.
@inline function _eval_batched_spline(splines::BatchedGPUSplines{D}, e, τ) where {D}
    @inbounds base = splines.offsets[e]
    @inbounds N_e = splines.Ns[e]
    idx = _searchsorted_left_batched(splines.t, base, N_e, τ)
    @inbounds h_idx = splines.h[base + idx + 1]
    @inbounds dt1 = τ - splines.t[base + idx]
    @inbounds dt2 = splines.t[base + idx + 1] - τ
    inv_6h = inv(6 * h_idx)
    return SVector{D}(
        ntuple(Val(D)) do d
            @inbounds splines.z[base + idx, d] * dt2^3 * inv_6h +
                splines.z[base + idx + 1, d] * dt1^3 * inv_6h +
                splines.c1[base + idx, d] * dt1 +
                splines.c2[base + idx, d] * dt2
        end
    )
end

# RHS of dτ_r/dt = 1 / (u⁰ - u⃗·n̂), batched form.
@inline function _rt_rhs_batched(τ, splines, e, r_obs)
    v = _eval_batched_spline(splines, e, τ)
    @inbounds x¹ = v[splines.x_idxs[2]]
    @inbounds x² = v[splines.x_idxs[3]]
    @inbounds x³ = v[splines.x_idxs[4]]
    @inbounds u⁰ = v[splines.u_idxs[1]]
    @inbounds u¹ = v[splines.u_idxs[2]]
    @inbounds u² = v[splines.u_idxs[3]]
    @inbounds u³ = v[splines.u_idxs[4]]
    d¹ = r_obs[1] - x¹
    d² = r_obs[2] - x²
    d³ = r_obs[3] - x³
    inv_r = inv(sqrt(d¹ * d¹ + d² * d² + d³ * d³))
    return inv(u⁰ - (u¹ * d¹ + u² * d² + u³ * d³) * inv_r)
end

# One Tsit5 step for the autonomous ODE dτ_r/dt = f(τ_r) with FSAL chaining.
# Caller passes `k1 = f(τ)` (reused from previous step's `k7` when available).
# Returns (τ_new, k7) where `k7 = f(τ_new)` for the next step's `k1`.
# Tableau values copied from SimpleDiffEq.jl/src/tsit5/tsit5.jl:134.
@inline function _tsit5_step_fsal(τ, dt, k1, splines, e, r_obs)
    a21 = 0.161
    a31 = -0.008480655492356989
    a32 = 0.335480655492357
    a41 = 2.8971530571054935
    a42 = -6.359448489975075
    a43 = 4.3622954328695815
    a51 = 5.325864828439257
    a52 = -11.748883564062828
    a53 = 7.4955393428898365
    a54 = -0.09249506636175525
    a61 = 5.86145544294642
    a62 = -12.92096931784711
    a63 = 8.159367898576159
    a64 = -0.071584973281401
    a65 = -0.028269050394068383
    a71 = 0.09646076681806523
    a72 = 0.01
    a73 = 0.4798896504144996
    a74 = 1.379008574103742
    a75 = -3.290069515436081
    a76 = 2.324710524099774

    k2 = _rt_rhs_batched(τ + dt * a21 * k1, splines, e, r_obs)
    k3 = _rt_rhs_batched(τ + dt * (a31 * k1 + a32 * k2), splines, e, r_obs)
    k4 = _rt_rhs_batched(τ + dt * (a41 * k1 + a42 * k2 + a43 * k3), splines, e, r_obs)
    k5 = _rt_rhs_batched(τ + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), splines, e, r_obs)
    k6 = _rt_rhs_batched(τ + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), splines, e, r_obs)
    τ_new = τ + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
    k7 = _rt_rhs_batched(τ_new, splines, e, r_obs)

    return (τ_new, k7)
end

# Per-batch kernel: Nx × Ny pixel threads, each looping over B electrons.
function _gpu_batched_tsit5_one_batch!(
        A_buf, splines::BatchedGPUSplines,
        x_grid, y_grid, z_screen,
        x⁰_first, δx⁰, N_samples, Nx, Ny, B,
        pixel_iter, backend,
    )
    AK.foreachindex(pixel_iter, backend) do i_lin
        ix = ((i_lin - 1) % Nx) + 1
        iy = ((i_lin - 1) ÷ Nx) + 1
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)
        inv_δ = inv(δx⁰)

        for e in 1:B
            @inbounds K_e = splines.Ks[e]
            @inbounds τi_e = splines.τis[e]
            @inbounds τf_e = splines.τfs[e]

            v_i = _eval_batched_spline(splines, e, τi_e)
            @inbounds d_i¹ = r_obs[1] - v_i[splines.x_idxs[2]]
            @inbounds d_i² = r_obs[2] - v_i[splines.x_idxs[3]]
            @inbounds d_i³ = r_obs[3] - v_i[splines.x_idxs[4]]
            @inbounds x⁰_i_px = v_i[splines.x_idxs[1]] +
                sqrt(d_i¹ * d_i¹ + d_i² * d_i² + d_i³ * d_i³)

            v_f = _eval_batched_spline(splines, e, τf_e)
            @inbounds d_f¹ = r_obs[1] - v_f[splines.x_idxs[2]]
            @inbounds d_f² = r_obs[2] - v_f[splines.x_idxs[3]]
            @inbounds d_f³ = r_obs[3] - v_f[splines.x_idxs[4]]
            @inbounds x⁰_f_px = v_f[splines.x_idxs[1]] +
                sqrt(d_f¹ * d_f¹ + d_f² * d_f² + d_f³ * d_f³)

            # Strict-interior slot range (matches Tsit5 + save_start/save_end=false).
            k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
            k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))
            if k_start > k_end
                continue
            end

            # Bridge x⁰_i_px → x⁰_samples[k_start] with a single Tsit5 step.
            τ = τi_e
            bridge_dt = x⁰_first + (k_start - 1) * δx⁰ - x⁰_i_px
            if bridge_dt > 0
                k1_bridge = _rt_rhs_batched(τ, splines, e, r_obs)
                (τ_after_bridge, _k7_bridge) = _tsit5_step_fsal(
                    τ, bridge_dt, k1_bridge, splines, e, r_obs
                )
                τ = clamp(τ_after_bridge, τi_e, τf_e)
            end

            # Seed FSAL k1 for the slot loop.  We reseed (rather than reusing
            # k7_bridge) because the bridge dt differs from δx⁰ — but since the
            # ODE is autonomous, k7_bridge = f(τ_after_bridge) would still be
            # valid; reseeding is just clearer and the cost is one RHS eval.
            k1_carry = _rt_rhs_batched(τ, splines, e, r_obs)

            for k in k_start:k_end
                τ_safe = clamp(τ, τi_e, τf_e)
                v = _eval_batched_spline(splines, e, τ_safe)
                @inbounds x¹ = v[splines.x_idxs[2]]
                @inbounds x² = v[splines.x_idxs[3]]
                @inbounds x³ = v[splines.x_idxs[4]]
                @inbounds u⁰ = v[splines.u_idxs[1]]
                @inbounds u¹ = v[splines.u_idxs[2]]
                @inbounds u² = v[splines.u_idxs[3]]
                @inbounds u³ = v[splines.u_idxs[4]]

                d¹ = r_obs[1] - x¹
                d² = r_obs[2] - x²
                d³ = r_obs[3] - x³
                r_norm = sqrt(d¹ * d¹ + d² * d² + d³ * d³)
                xr_dot_u = r_norm * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³)
                coeff = K_e / xr_dot_u

                @inbounds A_buf[k, 1, ix, iy] += coeff * u⁰
                @inbounds A_buf[k, 2, ix, iy] += coeff * u¹
                @inbounds A_buf[k, 3, ix, iy] += coeff * u²
                @inbounds A_buf[k, 4, ix, iy] += coeff * u³

                if k < k_end
                    (τ_next, k7) = _tsit5_step_fsal(
                        τ, δx⁰, k1_carry, splines, e, r_obs
                    )
                    τ = clamp(τ_next, τi_e, τf_e)
                    k1_carry = k7
                end
            end
        end
    end
    return
end

"""
    accumulate_potential(trajs, screen, ::GPUKernelTsit5, backend; batch_size = 8)

Batched GPU path: process electrons in chunks of `batch_size`, with each
kernel launch handling all B electrons in the batch.  Compared to the
single-electron `GPUKernelRK4` path:
  - one kernel launch per batch instead of per electron (300/B launches at
    N_macro = 300, B = 8 ⇒ ~38 launches), amortizing launch + sync overhead;
  - flat-with-offsets spline storage handles variable `length(itp.t)` per
    electron with zero padding waste;
  - 5th-order Tsit5 with FSAL replaces 4th-order RK4, ~2× better effective
    accuracy per RHS evaluation.

Per-batch device memory is bounded by `B × (spline_size + small)`, so this
scales to arbitrary `N_macro`.

# Arguments
- `batch_size`: B, number of electrons per kernel launch.  Pick large enough
  to saturate GPU occupancy (≈ Nx · Ny · B ≳ 4× SM count × 32 cores) but
  small enough that `B × spline_size` fits comfortably in VRAM.  8 is a
  reasonable starting point on a 16 GB card.
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelTsit5,
        backend::Backend;
        batch_size::Int = 8,
    )
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)

    A_buf = Adapt.adapt(backend, zeros(N_samples, 4, Nx, Ny))
    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)
    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny))

    N_macro = length(trajs)
    for batch_start in 1:batch_size:N_macro
        batch_end = min(batch_start + batch_size - 1, N_macro)
        chunk = view(trajs, batch_start:batch_end)
        B = length(chunk)

        cpu_splines = to_batched_cpu(chunk)
        splines = Adapt.adapt(backend, cpu_splines)

        _gpu_batched_tsit5_one_batch!(
            A_buf, splines,
            screen.x_grid, screen.y_grid, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny, B,
            pixel_iter, backend,
        )
        KernelAbstractions.synchronize(backend)

        finalize(splines.t)
        finalize(splines.h)
        finalize(splines.z)
        finalize(splines.c1)
        finalize(splines.c2)
        finalize(splines.offsets)
        finalize(splines.Ns)
        finalize(splines.Ks)
        finalize(splines.τis)
        finalize(splines.τfs)
    end

    return Array(A_buf)
end


# ── Per-pixel-per-electron saveat: time-integrated |A_μ|² accumulator ──
#
# For each (electron, pixel) pair, place `N_local` saveat slots inside that
# pair's *own* advanced-time arrival window — `[advanced_time(traj, τi, r_obs),
# advanced_time(traj, τf, r_obs)]` — and accumulate the time integral
# `∫|A_μ|² dx⁰` directly into a 4 × Nx × Ny output buffer.  This sidesteps the
# global `screen.x⁰_samples` grid entirely: the global grid had to be wide
# enough to cover σ_z bunch arrival spread (~1 mm in x⁰), so resolving the
# per-electron pulse (~0.12 μm) on it was infeasible.  Per-pair local saveat
# uses a window ~10⁴× narrower, so `N_local ≈ 200` resolves the pulse with
# ~50 slots across its FWHM.
#
# Squares per electron before summing → discards cross-electron coherence.
# That is correct for σ_z·k_rad ≫ 1 (Petrillo Fig 3): per-electron pulses on
# any one pixel are temporally disjoint, so cross-terms vanish anyway and
# coherent ≡ incoherent in expectation.  For OAM Fig 5(d) the cross-terms
# matter and a different path is needed (slice estimator).

function _gpu_intensity_one_electron!(
        intensity, gpu_traj,
        x_grid, y_grid, z_screen,
        N_local, Nx, Ny,
        τi, τf, pixel_iter, backend,
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        ix = ((i_lin - 1) % Nx) + 1
        iy = ((i_lin - 1) ÷ Nx) + 1
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Per-pixel arrival window from advanced_time:
        v_i = gpu_traj.itp(τi)
        d_i¹ = r_obs[1] - v_i[gpu_traj.x_idxs[2]]
        d_i² = r_obs[2] - v_i[gpu_traj.x_idxs[3]]
        d_i³ = r_obs[3] - v_i[gpu_traj.x_idxs[4]]
        x⁰_i_px = v_i[gpu_traj.x_idxs[1]] + sqrt(d_i¹ * d_i¹ + d_i² * d_i² + d_i³ * d_i³)

        v_f = gpu_traj.itp(τf)
        d_f¹ = r_obs[1] - v_f[gpu_traj.x_idxs[2]]
        d_f² = r_obs[2] - v_f[gpu_traj.x_idxs[3]]
        d_f³ = r_obs[3] - v_f[gpu_traj.x_idxs[4]]
        x⁰_f_px = v_f[gpu_traj.x_idxs[1]] + sqrt(d_f¹ * d_f¹ + d_f² * d_f² + d_f³ * d_f³)

        local_δx⁰ = (x⁰_f_px - x⁰_i_px) / (N_local - 1)

        # TODO(human): RK4-march the retarded-time ODE through `N_local`
        # uniformly-spaced observer-time slots in [x⁰_i_px, x⁰_f_px], and at
        # each slot accumulate `|A_μ|² × local_δx⁰` into intensity[μ, ix, iy]
        # for μ = 1..4.

        τ = τi
        for k in 1:N_local
            τ_safe = clamp(τ, τi, τf)
            v = gpu_traj.itp(τ_safe)
            x¹ = v[gpu_traj.x_idxs[2]]
            x² = v[gpu_traj.x_idxs[3]]
            x³ = v[gpu_traj.x_idxs[4]]
            u⁰ = v[gpu_traj.u_idxs[1]]
            u¹ = v[gpu_traj.u_idxs[2]]
            u² = v[gpu_traj.u_idxs[3]]
            u³ = v[gpu_traj.u_idxs[4]]

            d¹ = r_obs[1] - x¹
            d² = r_obs[2] - x²
            d³ = r_obs[3] - x³
            r_norm = sqrt(d¹ * d¹ + d² * d² + d³ * d³)
            # m_dot(xr, uμ) with xr = (r_norm, d¹, d², d³)
            xr_dot_u = r_norm * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³)
            coeff = K / xr_dot_u

            A_μ⁰ = coeff * u⁰
            A_μ¹ = coeff * u¹
            A_μ² = coeff * u²
            A_μ³ = coeff * u³

            @inbounds intensity[1, ix, iy] += A_μ⁰ * A_μ⁰ * local_δx⁰
            @inbounds intensity[2, ix, iy] += A_μ¹ * A_μ¹ * local_δx⁰
            @inbounds intensity[3, ix, iy] += A_μ² * A_μ² * local_δx⁰
            @inbounds intensity[4, ix, iy] += A_μ³ * A_μ³ * local_δx⁰

            if k < N_local
                # τ = _rk4_step(τ, local_δx⁰, gpu_traj, r_obs)
                τ = clamp(_rk4_step(τ, local_δx⁰, gpu_traj, r_obs), τi, τf)
            end
        end
        # See lines ~362-388 of `_gpu_unified_one_electron!` in this file for
        # the field-evaluation pattern (Liénard-Wiechert kernel).  Differences:
        #   - No global slot-index `k`.  intensity[:, ix, iy] is a single
        #     accumulator; thread owns its (ix, iy), so plain `+=` is safe.
        #   - No bridge step: by construction τ = τi corresponds to x⁰ = x⁰_i_px
        #     exactly, so start with τ = τi.
        #   - Step with `_rk4_step(τ, local_δx⁰, gpu_traj, r_obs)` between slots.
        #   - Square the per-component A_μ contribution before adding.
        #   - Multiply by `local_δx⁰` so the accumulator holds ∫|A_μ|² dx⁰
        #     (consistent units across pixels even though local_δx⁰ varies).
    end
    return
end

"""
    accumulate_intensity(trajs, screen, ::GPUKernelRK4, backend; N_local = 200)

Per-pixel time-integrated `|A_μ|²` for each 4-potential component, summed
incoherently across electrons.  Returns `intensity[μ, ix, iy]` of size
`(4, Nx, Ny)` — total memory `4 × Nx × Ny × 8` bytes (≈ 3 MB at 301²).

For each `(electron, pixel)` pair, places `N_local` saveat slots inside
that pair's own advanced-time arrival window and RK4-marches the retarded-
time ODE through them on the GPU.  Resolves the Doppler-compressed
per-electron pulse cleanly (its FWHM is typically ~10–50× the local slot
width at `N_local = 200`), which the global-grid path cannot do.

The output is `Σ_e ∫ |A_μ_e(x⁰)|² dx⁰` (length × |A|²-units).  To get a
time-averaged intensity matching the existing `mean(abs2, A_s; dims=1)`
of the coherent path, divide by the global observation window length
`(last(screen.x⁰_samples) - first(screen.x⁰_samples))` and multiply by
the macroparticle weight `w`.

Cross-electron coherence is **lost** (squaring is per-electron).  In the
σ_z·k_rad ≫ 1 regime per-electron pulses don't overlap on a pixel anyway,
so this matches the coherent answer in expectation; for OAM coherent
patterns, this method is the wrong tool.

`screen.x⁰_samples` is read only for output normalization at the call site
— this kernel ignores it entirely.
"""
function accumulate_intensity(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelRK4,
        backend::Backend;
        N_local::Int = 200,
    )
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    intensity = Adapt.adapt(backend, zeros(4, Nx, Ny))

    # Sentinel iteration target: one element per pixel.
    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny))

    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        _gpu_intensity_one_electron!(
            intensity, gpu_traj,
            screen.x_grid, screen.y_grid, screen.z,
            N_local, Nx, Ny,
            τi, τf, pixel_iter, backend,
        )
        KernelAbstractions.synchronize(backend)
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    return Array(intensity)
end
