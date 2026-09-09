# ── Unified GPU kernel: retarded-time solve + LW accumulation per electron ──
# Dispatched via `accumulate_potential(trajs, screen, GPUKernelRK4(), backend)`.
# One thread per pixel, fixed-step RK4 at the saveat resolution, immediate
# accumulation into the device output buffer — no τ_all intermediate, no
# CPU-GPU synchronization mid-flight.
#
# Defined directly in the ElectronDynamicsModels module (not Experimental):
# `accumulate_potential`, `TrajectoryInterpolant`, `ObserverScreen`, `to_gpu`,
# `Adapt`, `AK`, `KernelAbstractions`, `Backend`, and `SVector` are all already
# in scope from the parent module.

"""
    GPUKernelRK4

Sentinel solver type that selects the unified GPU kernel path.
"""
struct GPUKernelRK4 end

"""
    recommended_n_substeps(screen, c; ω_max = π * c / step(screen.x⁰_samples), rtol = 1e-6)

Suggest an `n_substeps` value for the fixed-step GPU kernels from the screen's
own sampling, with an optional override for the highest relevant frequency.

`c` is the speed of light in the working units (e.g. `getdefault(world.c)`).
The screen stores the time-like coordinate `x⁰ = c·t`, so `δx⁰ = step(x⁰_samples)`
is a *length* and the integration time step is `δt = δx⁰ / c`.

Fixed-step RK4 stepping the retarded-time ODE `dτ_r/dx⁰ = f` multiplies a mode
of angular frequency `ω` by the stability polynomial `|R(iθ)|`, `θ = ω·δt`.
Expanding, `|R(iθ)| − 1 ≈ θ⁶/144`, so at `θ = π/2` the per-step amplitude loss
is ~7.5%, while at `θ ≈ 0.23` it is ~1e-6.  Sub-stepping each saveat interval
into `n` pieces drives `θ = ω_max·δt/n` down into the accurate range.

`ω_max` is a temporal angular frequency (rad/time) and defaults to the screen's
**Nyquist angular frequency** `π·c/δx⁰ = π/δt`: a uniform `x⁰_samples` grid
cannot represent anything faster, so resolving the RK4 step to Nyquist
guarantees the integration is as accurate as the output grid can carry.  With
this default `c` and `δx⁰` cancel and the result is a screen-independent
constant (≈14 at `rtol = 1e-6`).  A caller who knows their top harmonic sits
below Nyquist (e.g. `ω_max = n_harmonics · 2π·c/λ`) can pass it explicitly to
take fewer sub-steps.

`rtol` is the target per-step amplitude error.

!!! note
    Sub-stepping cannot rescue an under-sampled screen.  If the radiation has
    content above the Nyquist frequency the screen is already aliasing it —
    that is a `δx⁰`-too-coarse problem upstream, not one more sub-steps can fix.
"""
function recommended_n_substeps(
        screen::ObserverScreen;
        ω_max = π * screen.c / step(screen.x⁰_samples), rtol = 1.0e-6
    )
    # x⁰ = c·t, so the saveat spacing is a length; the integration time step is
    # δt = δx⁰ / c and the per-step phase is θ = ω_max · δt.
    δt = step(screen.x⁰_samples) / screen.c
    # Largest inner-step angle holding the per-step amplitude error θ⁶/144 ≤ rtol,
    # then the fewest sub-steps (≥ 1) that keep ω_max·(δt/n) ≤ θ_max.
    θ_max = (144 * rtol)^(1 / 6)
    return max(1, ceil(Int, δt * (ω_max / θ_max)))
end

# RHS of dτ_r/dt = 1 / (u⁰ - u⃗·n̂); allocation-free, kernel-callable. State components
# are read with literal indices — canonical order guaranteed by `to_gpu` (see kernel_newton.jl).
@muladd @inline function _rt_rhs_from_state(v, r_obs)
    x¹ = v[2]
    x² = v[3]
    x³ = v[4]
    u⁰ = v[5]
    u¹ = v[6]
    u² = v[7]
    u³ = v[8]
    d¹ = r_obs[1] - x¹
    d² = r_obs[2] - x²
    d³ = r_obs[3] - x³
    inv_r = inv(sqrt(d¹ * d¹ + d² * d² + d³ * d³))
    return inv(u⁰ - (u¹ * d¹ + u² * d² + u³ * d³) * inv_r)
end

# Spline evaluation + RHS; the warm-started search carries `idx` from the previous eval.
@inline function _rt_rhs_kernel(τ, gpu_traj, r_obs, idx)
    v, idx = gpu_traj.itp(τ, idx)
    return _rt_rhs_from_state(v, r_obs), idx
end

# One classical RK4 step for the autonomous ODE dτ_r/dt = f(τ_r).
@muladd @inline function _rk4_step(τ, dt, gpu_traj, r_obs, idx, ::Val{false})
    k1, idx = _rt_rhs_kernel(τ, gpu_traj, r_obs, idx)
    k2, idx = _rt_rhs_kernel(τ + 0.5 * dt * k1, gpu_traj, r_obs, idx)
    k3, idx = _rt_rhs_kernel(τ + 0.5 * dt * k2, gpu_traj, r_obs, idx)
    k4, idx = _rt_rhs_kernel(τ + dt * k3, gpu_traj, r_obs, idx)
    return τ + dt * (k1 + 2 * (k2 + k3) + k4) * (1 / 6), idx
end

# Same step with the spline interval's coefficients held in registers (`coef_reuse = Val(true)`):
# the four stages sit in one interval nearly always; a stage that leaves it refetches.
# Identical arithmetic ⇒ bit-identical results.
@muladd @inline function _rk4_step(τ, dt, gpu_traj, r_obs, idx, ::Val{true})
    itp = gpu_traj.itp
    idx = _searchsorted_left(itp.t, τ, idx)
    coef = _fetch_interval(itp, idx)
    k1 = _rt_rhs_from_state(_eval_poly(coef, τ), r_obs)
    τ2 = τ + 0.5 * dt * k1
    if !_in_interval(coef, τ2)
        idx = _searchsorted_left(itp.t, τ2, idx)
        coef = _fetch_interval(itp, idx)
    end
    k2 = _rt_rhs_from_state(_eval_poly(coef, τ2), r_obs)
    τ3 = τ + 0.5 * dt * k2
    if !_in_interval(coef, τ3)
        idx = _searchsorted_left(itp.t, τ3, idx)
        coef = _fetch_interval(itp, idx)
    end
    k3 = _rt_rhs_from_state(_eval_poly(coef, τ3), r_obs)
    τ4 = τ + dt * k3
    if !_in_interval(coef, τ4)
        idx = _searchsorted_left(itp.t, τ4, idx)
        coef = _fetch_interval(itp, idx)
    end
    k4 = _rt_rhs_from_state(_eval_poly(coef, τ4), r_obs)
    return τ + dt * (k1 + 2 * (k2 + k3) + k4) * (1 / 6), idx
end

# Cold-start forms (host diagnostics, tests): same values, search from the first interval.
_rt_rhs_kernel(τ, gpu_traj, r_obs) = first(_rt_rhs_kernel(τ, gpu_traj, r_obs, 1))
_rk4_step(τ, dt, gpu_traj, r_obs) = first(_rk4_step(τ, dt, gpu_traj, r_obs, 1, Val(false)))

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
        τi, τf, pixel_iter, backend, n_substeps, coef_reuse = Val(false), n_chunks = 1
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Chunk-major unpacking of the (pixels × chunks) index space: ix fastest, iy, then chunk.
        ix, iy, chunk = _chunk_pixel(i_lin, Nx, Ny, n_chunks)
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Pixel-specific advanced-time window x⁰_i, x⁰_f.  NOTE: kernel_newton.jl
        # computes the same window in screen-relative offsets (light-front
        # residual) — the two kernels' window blocks are intentionally no longer
        # line-identical.
        v_i = gpu_traj.itp(τi)
        d_i¹ = r_obs[1] - v_i[2]
        d_i² = r_obs[2] - v_i[3]
        d_i³ = r_obs[3] - v_i[4]
        x⁰_i_px = v_i[1] + sqrt(d_i¹^2 + d_i²^2 + d_i³^2)

        v_f = gpu_traj.itp(τf)
        d_f¹ = r_obs[1] - v_f[2]
        d_f² = r_obs[2] - v_f[3]
        d_f³ = r_obs[3] - v_f[4]
        x⁰_f_px = v_f[1] + sqrt(d_f¹^2 + d_f²^2 + d_f³^2)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))

        if k_start > k_end
            return
        end
        # This thread's slice of the executed slots; chunks beyond the range are empty.
        k_first, k_last = _chunk_slots(k_start, k_end, chunk, n_chunks)
        if k_first > k_last
            return
        end

        # Bridge τ from τi (the τ at observer time x⁰_i_px) up to observer time
        # x⁰_samples[k_start], then advance τ by δx⁰ between successive slots.
        # Each advance is taken as `n_substeps` RK4 sub-steps of dt =
        # δx⁰/n_substeps, bringing ω·dt into RK4's accurate range (a single
        # step works only for ω·δx⁰ ≪ 1, which fails at ω·δx⁰ ≈ π/2).
        # A chunk beyond the first has no anchor at its first slot: it takes the
        # retarded time there from N_COLD_ITERS safeguarded Newton corrections on
        # the light-cone residual (kernel_newton.jl), then marches as usual.
        τ = τi
        idx = 1   # spline interval of the previous evaluation: warm start of the knot search
        if chunk == 1
            bridge_dt = x⁰_first + (k_start - 1) * δx⁰ - x⁰_i_px
            if bridge_dt > 0
                sub_dt = bridge_dt / n_substeps
                for _ in 1:n_substeps
                    τ, idx = _rk4_step(τ, sub_dt, gpu_traj, r_obs, idx, coef_reuse)
                end
                τ = clamp(τ, τi, τf)
            end
        else
            tₖc = (x⁰_first - z_screen) + (k_first - 1) * δx⁰
            t_i_px, rhs0 = _window_edge(gpu_traj, r_obs, τi)
            τ, _, _, _, _, _, _, _, _, idx =
                _bracketed_slot_solve(τi, tₖc - t_i_px, rhs0, τi, gpu_traj, r_obs, tₖc, τi, τf, N_COLD_ITERS, 1, Val(false))
        end

        # March through saveat slots, accumulating at each
        for k in k_first:k_last
            τ_safe = clamp(τ, τi, τf)
            v, idx = gpu_traj.itp(τ_safe, idx)

            x¹ = v[2]
            x² = v[3]
            x³ = v[4]
            u⁰ = v[5]
            u¹ = v[6]
            u² = v[7]
            u³ = v[8]

            d¹ = r_obs[1] - x¹
            d² = r_obs[2] - x²
            d³ = r_obs[3] - x³
            r_norm = sqrt(d¹ * d¹ + d² * d² + d³ * d³)
            # m_dot(xr, uμ) with xr = (r_norm, d¹, d², d³)
            xr_dot_u = r_norm * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³)
            coeff = K / xr_dot_u

            @inbounds A_buf[ix, iy, 1, k] += coeff * u⁰
            @inbounds A_buf[ix, iy, 2, k] += coeff * u¹
            @inbounds A_buf[ix, iy, 3, k] += coeff * u²
            @inbounds A_buf[ix, iy, 4, k] += coeff * u³

            if k < k_last
                sub_dt = δx⁰ / n_substeps
                for _ in 1:n_substeps
                    τ, idx = _rk4_step(τ, sub_dt, gpu_traj, r_obs, idx, coef_reuse)
                end
                τ = clamp(τ, τi, τf)
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
`ω · δx⁰` lies outside RK4's accurate range.  The per-step amplitude error of
fixed-step RK4 on a mode of frequency `ω` is `|R(iθ)| − 1 ≈ θ⁶/144` with
`θ = ω · dt` (`R` the RK4 stability polynomial): at `θ = π/2` that is ~7.5%/step,
at `θ ≈ 0.23` it is ~1e-6.  So to hold per-step error ≤ ε, take
`n_substeps = ⌈ ω·δx⁰ / (144 ε)^(1/6) ⌉`.  For the relativistic Thomson script
(a₀ = 10, δt = T/4 ⇒ `ω·δx⁰ ≈ π/2`), the relevant `ω` is the highest harmonic
present (≈ a₀³ × ω_fundamental for nonlinear Thomson), so `n_substeps ≈ 8`
recovers parity with the adaptive-Tsit5 reference.

`sync_per_electron` (default `true`) inserts a `KernelAbstractions.synchronize`
before freeing each trajectory's device buffers — safe but serializes the
electron loop.  Setting it `false` drops the per-electron sync and relies on
stream-ordered async free (the kernel that still reads the buffers is queued
ahead of the free on the same stream), letting electron N+1's upload overlap
kernel N.  Verified correct on CUDA and ROCm backends.

`timer = LaunchTimer()` records a device-event pair per launch (see [`LaunchTimer`](@ref)). `coef_reuse = Val(true)` holds each slot's spline-interval coefficients in registers across the per-slot evaluations (bit-identical; trades ~4D live registers for the coefficient loads, see [`IntervalCoefs`](@ref)).
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelRK4,
        backend::Backend;
        n_substeps::Int = 1,
        coef_reuse::Val = Val(false),
        sample_chunks::Int = 1,
        sync_per_electron::Bool = true,
        timer = nothing,
    )
    sample_chunks ≥ 1 || throw(ArgumentError("sample_chunks must be ≥ 1, got $sample_chunks"))
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)

    # Pixel-fastest layout (Nx, Ny leading) so warp-adjacent pixel-threads
    # (consecutive ix) write consecutive addresses → coalesced accumulation.
    # The kernel writes A_buf[ix, iy, μ, k]; we permute back to the public
    # (N_samples, 4, Nx, Ny) layout (rfft over dim 1) on return.
    A_buf = Adapt.adapt(backend, zeros(Nx, Ny, 4, N_samples))

    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)

    # Iteration target: one element per pixel. Sentinel array; never read.
    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny, sample_chunks))   # one thread per (pixel, chunk)

    # Per-electron upload-and-free streaming: each trajectory's spline
    # arrays (~21 MB at N_t = 10⁵) are uploaded just before the kernel
    # launch and freed immediately after.  This keeps device memory
    # bounded by `A_buf + one trajectory + pixel_iter` regardless of
    # N_macro, so we can scale to thousands of electrons on a 16 GB card.
    lane = launch_lane(timer, backend)
    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        e0 = launch_tick(timer, backend)
        _gpu_unified_one_electron!(
            A_buf, gpu_traj,
            screen.x_grid, screen.y_grid, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_substeps, coef_reuse, sample_chunks,
        )
        launch_tock!(timer, lane, backend, e0)
        # Release the trajectory's device buffers.  With `sync_per_electron`
        # we wait for the kernel first (safe but serializing); otherwise we
        # rely on stream-ordered async free — `finalize` queues `unsafe_free!`
        # behind kernel N on the same stream, so the read completes before the
        # memory is reused, and electron N+1's upload overlaps kernel N.
        sync_per_electron && KernelAbstractions.synchronize(backend)
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    # Permute the pixel-fastest buffer back to the public (N_samples, 4, Nx, Ny)
    # layout on the HOST (GPU permutedims overflows 32-bit indices past 2³¹ elements),
    # chunked so the peak stays ~1× cube instead of 2× — see _download_permuted.
    return _download_permuted(A_buf)
end

# ── Field variant of the unified RK4 kernel ──
# Identical retarded-time march to `_gpu_unified_one_electron!`, but each saveat
# slot writes the Liénard–Wiechert (E, B) field rather than the 4-potential, and
# splits it into separately-accumulated far (1/R) and near (1/R²)
# pieces via `lienard_wiechert_F_split` (see `accumulate_field`). Differences from
# the potential kernel:
#   - it needs the 4-acceleration spline `gpu_traj.a_itp` (the far field is
#     ∝ 𝔞), uploaded via `to_gpu(traj; with_acceleration = true)`
#   - it needs the speed of light `c`, since `Eⁱ = c·Fⁱ⁰` in `extract_EB`
#   - it writes four buckets, not two.
# Writes pixel-fastest into {E,B}{1,2}_buf[ix, iy, j, k] (buffer 1 = far, 2 = near) for
# coalesced accumulation; the caller permutes back to the public (N_samples, 3, Nx, Ny).
function _gpu_unified_field_one_electron!(
        mode::Val, E1_buf, B1_buf, E2_buf, B2_buf, gpu_traj, c,
        x_grid, y_grid, z_screen,
        x⁰_first, δx⁰, N_samples, Nx, Ny,
        τi, τf, pixel_iter, backend, n_substeps, coef_reuse = Val(false), n_chunks = 1
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Chunk-major unpacking of the (pixels × chunks) index space: ix fastest, iy, then chunk.
        ix, iy, chunk = _chunk_pixel(i_lin, Nx, Ny, n_chunks)
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Pixel-specific advanced-time window x⁰_i, x⁰_f.  NOTE: kernel_newton.jl
        # computes the same window in screen-relative offsets (light-front
        # residual) — the two kernels' window blocks are intentionally no longer
        # line-identical.
        v_i = gpu_traj.itp(τi)
        d_i¹ = r_obs[1] - v_i[2]
        d_i² = r_obs[2] - v_i[3]
        d_i³ = r_obs[3] - v_i[4]
        x⁰_i_px = v_i[1] + sqrt(d_i¹^2 + d_i²^2 + d_i³^2)

        v_f = gpu_traj.itp(τf)
        d_f¹ = r_obs[1] - v_f[2]
        d_f² = r_obs[2] - v_f[3]
        d_f³ = r_obs[3] - v_f[4]
        x⁰_f_px = v_f[1] + sqrt(d_f¹^2 + d_f²^2 + d_f³^2)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))

        if k_start > k_end
            return
        end
        # This thread's slice of the executed slots; chunks beyond the range are empty.
        k_first, k_last = _chunk_slots(k_start, k_end, chunk, n_chunks)
        if k_first > k_last
            return
        end

        # Bridge τ from τi up to observer time x⁰_samples[k_start], then advance
        # by δx⁰ between successive slots (each as n_substeps RK4 sub-steps);
        # chunks > 1 start from a cold light-cone solve (see the potential kernel).
        τ = τi
        idx = 1   # spline interval of the previous evaluation: warm start of the knot search
        if chunk == 1
            bridge_dt = x⁰_first + (k_start - 1) * δx⁰ - x⁰_i_px
            if bridge_dt > 0
                sub_dt = bridge_dt / n_substeps
                for _ in 1:n_substeps
                    τ, idx = _rk4_step(τ, sub_dt, gpu_traj, r_obs, idx, coef_reuse)
                end
                τ = clamp(τ, τi, τf)
            end
        else
            tₖc = (x⁰_first - z_screen) + (k_first - 1) * δx⁰
            t_i_px, rhs0 = _window_edge(gpu_traj, r_obs, τi)
            τ, _, _, _, _, _, _, _, _, idx =
                _bracketed_slot_solve(τi, tₖc - t_i_px, rhs0, τi, gpu_traj, r_obs, tₖc, τi, τf, N_COLD_ITERS, 1, Val(false))
        end

        # March through saveat slots, writing the (E, B) field at each
        for k in k_first:k_last
            τ_safe = clamp(τ, τi, τf)

            v, idx = gpu_traj.itp(τ_safe, idx)
            xμ = SVector{4}(v[1], v[2], v[3], v[4])
            uμ = SVector{4}(v[5], v[6], v[7], v[8])
            𝔞μ = _eval_at(gpu_traj.a_itp, τ_safe, idx)   # same knots as itp (checked in to_gpu)

            disp = r_obs - xμ[SA[2, 3, 4]]
            X = SVector{4}(norm(disp), disp[1], disp[2], disp[3])
            F_near, F_far = lienard_wiechert_F_split(X, uμ, 𝔞μ, K, c)
            E_near, B_near = extract_EB(F_near, c)
            E_far, B_far = extract_EB(F_far, c)

            # `:split` keeps far (buffer 1) and near (buffer 2) separate; `:total`
            # sums them into buffer 1 alone (E2/B2 alias E1/B1 in :total — never written here).
            # The branch is on a `Val` singleton, so it folds at compile time (zero per-pixel
            # cost) and the `:split` arm stays byte-identical. The small-a0 conditioning lives
            # in `lienard_wiechert_F_split`, so the collapsed sum is the bit-faithful total.
            for j in 1:3
                if mode === Val(:split)
                    @inbounds E1_buf[ix, iy, j, k] += E_far[j]
                    @inbounds B1_buf[ix, iy, j, k] += B_far[j]
                    @inbounds E2_buf[ix, iy, j, k] += E_near[j]
                    @inbounds B2_buf[ix, iy, j, k] += B_near[j]
                else
                    @inbounds E1_buf[ix, iy, j, k] += E_far[j] + E_near[j]
                    @inbounds B1_buf[ix, iy, j, k] += B_far[j] + B_near[j]
                end
            end

            if k < k_last
                sub_dt = δx⁰ / n_substeps
                for _ in 1:n_substeps
                    τ, idx = _rk4_step(τ, sub_dt, gpu_traj, r_obs, idx, coef_reuse)
                end
                τ = clamp(τ, τi, τf)
            end
        end
    end
    return
end

"""
    accumulate_field(trajs, screen, ::GPUKernelRK4, backend; n_substeps = 1, mode = Val(:split), sync_per_electron = true)

GPU counterpart of the CPU [`accumulate_field`](@ref): per-electron kernel launch
performs the retarded-time RK4 integration *and* Liénard–Wiechert (E, B)
accumulation in one pass, coherently summing over electrons on the device.

Mirrors `accumulate_potential(trajs, screen, ::GPUKernelRK4, backend)`, but
uploads the acceleration spline (`to_gpu(traj; with_acceleration = true)`) since
the far field needs 𝔞μ, and accumulates the split (far/near)
Liénard–Wiechert field instead of the 4-potential. Returns
`(; E, B, E_far, B_far)`, each `(N_samples, 3, Nx, Ny)` — identical shape to the
CPU [`accumulate_field`](@ref): `E, B` total, `E_far, B_far` the far field
alone (see [`lienard_wiechert_F_split`](@ref)). `mode = Val(:total)` returns only
`(; E, B)` (a type-stable trim); `Val(:split)` (the default) keeps all four.

`n_substeps` and `sync_per_electron` behave exactly as in the potential kernel;
see [`accumulate_potential`](@ref) and [`recommended_n_substeps`](@ref).

`buffers` / `finish` drive electron BATCHING: `finish = false` returns the live
[`FieldAccumulator`](@ref) instead of the downloaded cube, and passing it back as `buffers`
accumulates the next batch of electrons into the same device buffers, so the host only ever
holds one batch of trajectory splines. The download happens once, on the call with
`finish = true` (or through [`finish_field`](@ref)); on one device the batched sum is
bit-identical to the single call.

`timer = LaunchTimer()` records a device-event pair per launch (see [`LaunchTimer`](@ref)). `coef_reuse = Val(true)` holds each slot's spline-interval coefficients in registers across the per-slot evaluations (bit-identical; trades ~4D live registers for the coefficient loads, see [`IntervalCoefs`](@ref)).
"""
function accumulate_field(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelRK4,
        backend::Backend;
        n_substeps::Int = 1,
        coef_reuse::Val = Val(false),
        sample_chunks::Int = 1,
        mode::Val = Val(:split),
        sync_per_electron::Bool = true,
        sink = nothing,
        buffers = nothing,
        finish::Bool = true,
        timer = nothing,
    )
    sample_chunks ≥ 1 || throw(ArgumentError("sample_chunks must be ≥ 1, got $sample_chunks"))
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)
    c = screen.c

    # Pixel-fastest accumulators for coalesced writes. `:split` keeps the far and near fields
    # in separate buckets (4 buffers); `:total` collapses far+near in the kernel into a single
    # (E, B) pair (2 buffers), halving device memory — the win that lets a bigger screen fit.
    # In `:total`, E2/B2 alias E1/B1 and are never written.
    # Accumulation buffers: a fresh set per call by default, or the caller's persistent
    # `FieldAccumulator` (electron batching — the buffers then outlive this call).
    acc = _field_buffers(buffers, screen, backend, mode)
    E1_buf, B1_buf, E2_buf, B2_buf = acc.E1, acc.B1, acc.E2, acc.B2

    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)

    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny, sample_chunks))   # one thread per (pixel, chunk)
    lane = launch_lane(timer, backend)

    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj; with_acceleration = true))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        e0 = launch_tick(timer, backend)
        _gpu_unified_field_one_electron!(
            mode, E1_buf, B1_buf, E2_buf, B2_buf, gpu_traj, c,
            screen.x_grid, screen.y_grid, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_substeps, coef_reuse, sample_chunks,
        )
        launch_tock!(timer, lane, backend, e0)
        sync_per_electron && KernelAbstractions.synchronize(backend)
        # Free both splines' device buffers (state + acceleration).
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
        finalize(gpu_traj.a_itp.t)
        finalize(gpu_traj.a_itp.h)
        finalize(gpu_traj.a_itp.z)
        finalize(gpu_traj.a_itp.c1)
        finalize(gpu_traj.a_itp.c2)
    end

    # `finish = false` hands the live buffers back for the next batch of electrons; otherwise
    # download (or hand to `sink`) while they are still alive — in `:total` far+near are already
    # summed in the kernel, so E1/B1 hold the total directly. See _finish_fields /
    # _collect_fields / _add_fields! in accumulate.jl.
    acc.n_electrons += length(trajs)
    if !finish
        # Hand the live buffers back — but drain this call's launches first. A later batch may be
        # accumulated from a DIFFERENT task (the sharded driver spawns one per device per batch),
        # and a different task is a different stream: two streams read-modify-writing the same
        # accumulator would lose updates. One sync per batch, against thousands of launches.
        KernelAbstractions.synchronize(backend)
        return acc
    end
    return _finish_fields(sink, E1_buf, B1_buf, E2_buf, B2_buf, mode)
end
