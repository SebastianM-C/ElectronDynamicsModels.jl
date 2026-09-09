# ── Newton light-cone GPU kernel: per-slot root solve + LW accumulation ──
# Extends `accumulate_potential`/`accumulate_field` as the alternative to the
# GPUKernelRK4 march (kernel_rk4.jl): instead of integrating
# dτ_r/dx⁰ = 1/(u⁰ − u⃗·n̂) between saveat slots, solve the light-cone condition
# at each slot with warm-started Newton corrections.  The residual is spelled
# in screen-relative (light-front) form,
#     f(τ) = tₖ − ψ(τ) − ρ²/(R + d³)  =  0
#     tₖ = x⁰_target − z_screen   (offset sample grid, built small)
#     ψ  = x⁰(τ) − x³(τ)          (light-front coordinate of the emission event)
#     ρ⃗  = (d¹, d²),  d³ = z_screen − x³(τ),  R = |r_obs − x⃗(τ)|
# — algebraically identical to x⁰_target − x⁰(τ) − R (difference-of-squares
# regrouping, R − d³ = ρ²/(R + d³)), but every stored term is interaction-scale,
# which removes the ε·Z cancellation floor of the absolute spelling (τ noise
# ~5e-9 at the production Z ≈ 3e9 a.u., growing ∝ γ² against the sample
# spacing).  Assumes the screen is downstream (z_screen > x³(τ)) and
# trajectories are interaction-centered (|xμ| ≪ Z) — true for all EDM setups.
# f is strictly monotone
# decreasing — f′(τ) = −(u⁰ − u⃗·n̂) < 0 since u⁰ ≥ |u⃗| on a timelike worldline —
# so the root is unique and the Newton derivative comes for free from the same
# spline eval that provides the residual.  Unlike the RK4 march, the per-slot
# error does not accumulate along the observer-time grid.
#
# Which kernel when (newton_a0_{low,high} production campaigns, MI300X 2026-07;
# W7900 potential-path numbers in reports/newton_lightcone_retarded_time.typ):
#   - Boosted forward scattering (γ ≫ 1): Newton. At γ = 10, n_iters = 1 is
#     ~10× more accurate than RK4 n_substeps = 1 and 4.3× cheaper than
#     matched-accuracy RK4 n_substeps = 8; converged by n_iters = 2.
#   - Rest-electron field maps (a₀ ≲ 0.1): RK4. Accuracy is identical in norm
#     (n_iters = 1 already at the 9.4e-11 floor; ≡ n_iters = 2 to 1e-13) but
#     RK4 runs 1.3–1.4× faster there, and its error is smooth/correlated along
#     observer time, so deep-floor harmonic B maps keep coherent ring structure
#     where Newton's white per-slot error raises the floor (h4 B ~129×).
# Per-slot error character is the differentiator near numerical floors:
# norms decide convergence, error spectra decide what survives the FFT.

"""
    GPUKernelNewton

Sentinel solver type selecting the Newton light-cone GPU kernel path: the
retarded proper time at each saveat slot is obtained by solving the light-cone
condition `x⁰_target − x⁰(τ) − |r_obs − x⃗(τ)| = 0` with `n_iters` warm-started
Newton corrections, instead of marching the retarded-time ODE as
[`GPUKernelRK4`](@ref) does.  The residual is evaluated in the equivalent
screen-relative light-front spelling `f = tₖ − ψ(τ) − ρ²/(R + d³)` (see the
file header), which keeps Float64 rounding at the interaction scale instead of
the ε·Z floor of the absolute coordinates.

Per-slot errors are independent across observer-time slots (no accumulation).
Prefer this kernel for boosted forward-scattering geometries (γ ≫ 1), where
`n_iters = 1–2` beats matched-accuracy RK4 by ~4× in cost; prefer
[`GPUKernelRK4`](@ref) for rest-electron field maps and deep-floor harmonic
B-map studies, where its smoother error spectrum preserves near-floor ring
structure. Measured campaign numbers in the header of `gpu/kernel_newton.jl`.
"""
struct GPUKernelNewton end

# One spline eval → light-cone residual f, Newton factor rhs = −1/f′, and the
# geometric pieces the accumulation reuses.  The Newton update is
# τ ← τ + f·rhs with rhs = 1/(u⁰ − u⃗·n̂) = r_norm/xr_dot_u > 0, and the final
# (converged) eval doubles as the accumulation eval: `v` carries the full
# [xμ; uμ] state and K/xr_dot_u = K·rhs/r_norm.
# State components are read with LITERAL indices (x⁰…x³ = v[1:4], u⁰…u³ = v[5:8]):
# `to_gpu` guarantees the canonical order. Indexing through the runtime `x_idxs`/`u_idxs`
# vectors forced the SVector{8} into a private array (scratch, and on AMD a 64 KB LDS
# reservation via promote-alloca that capped residency at one workgroup per WGP).
@muladd @inline function _lightcone_from_state(v, r_obs, tₖ)
    x⁰ = v[1] # x⁰(τ)
    x³ = v[4] # x³(τ)
    d¹ = r_obs[1] - v[2]
    d² = r_obs[2] - v[3]
    d³ = r_obs[3] - v[4]
    # ρ⃗ = x⊥ − r⊥(τ) = (d¹, d²): the transverse pixel offsets are already the
    # subtract-first small differences.  R − d³ = ρ²/(R + d³) exactly
    # (difference of squares); the regrouped form has no O(Z) cancellation left
    # in it.  Conditioning assumes z_screen > x³(τ) (screen downstream).
    ρ² = d¹ * d¹ + d² * d²
    r_norm = sqrt(ρ² + d³ * d³)
    screen_dist = ρ² / (r_norm + d³)
    # ψ(τ) = x⁰(τ) − x³(τ): light-front coordinate of the emission event,
    # small and slowly varying for forward motion (dψ/dτ = u⁰ − u³).
    ψ = x⁰ - x³
    u⁰ = v[5]
    u¹ = v[6]
    u² = v[7]
    u³ = v[8]
    # X^μ = x^μ - x^μ(τ) = (x⁰_k - x⁰(τ), R)
    # m_dot(xr, uμ) with xr = (r_norm, d¹, d², d³)
    xr_dot_u = r_norm * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³)
    # n̂ = R/||R|| => X ⋅ u = r_norm * (u⁰ - n̂ ⋅ u⃗)
    rhs = r_norm / xr_dot_u
    # f = tₖ − ψ(τ) − ρ²/(R + d³)  ≡  x⁰_k − x⁰(τ) − R  with  x⁰_k = z_screen + tₖ
    f = tₖ - ψ - screen_dist
    return f, rhs, r_norm, d¹, d², d³
end

# Spline evaluation + residual; the warm-started search carries `idx` from the previous eval.
@inline function _lightcone_eval(τ, gpu_traj, r_obs, tₖ, idx)
    v, idx = gpu_traj.itp(τ, idx)
    f, rhs, r_norm, d¹, d², d³ = _lightcone_from_state(v, r_obs, tₖ)
    return v, f, rhs, r_norm, d¹, d², d³, idx
end

# Cold-start form (host diagnostics, scripts, tests): same values, search from the first interval.
_lightcone_eval(τ, gpu_traj, r_obs, tₖ) = Base.front(_lightcone_eval(τ, gpu_traj, r_obs, tₖ, 1))

# Pixel arrival-window edge in screen-relative offsets — the light-front
# spelling of x⁰(τ) + R − z_screen — plus the Doppler factor 1/(u⁰ − n̂·u⃗)
# there (used for the warm-start predictor stride).  Shared by the potential
# and field kernels.
@muladd @inline function _window_edge(gpu_traj, r_obs, τ)
    v = gpu_traj.itp(τ)
    x⁰ = v[1]
    x³ = v[4]
    d¹ = r_obs[1] - v[2]
    d² = r_obs[2] - v[3]
    d³ = r_obs[3] - v[4]
    ρ² = d¹ * d¹ + d² * d²
    R = sqrt(ρ² + d³ * d³)
    t_px = (x⁰ - x³) + ρ² / (R + d³)
    u⁰ = v[5]
    u¹ = v[6]
    u² = v[7]
    u³ = v[8]
    rhs = R / (R * u⁰ - (d¹ * u¹ + d² * u² + d³ * u³))
    return t_px, rhs
end

# One slot of the per-pixel march, shared by the potential and field kernels:
# Euler predictor from the previous slot's converged state, then `n_iters`
# bracketed Newton corrections on the light-cone residual.  Every eval
# tightens the enclosure [lo, hi] by one sign test (f is strictly decreasing:
# f > 0 ⇒ τ left of the root).  The safeguarded step trusts the Newton
# proposal iff it stays inside the OPEN interval — strict inequalities,
# because at convergence prop == τ may sit exactly on the boundary it just
# became, and must be accepted — and takes the bisection midpoint otherwise
# (guaranteed progress: the enclosure halves).  Fixed trip count + branchless
# select keep warp lockstep.  Returns the converged eval so the caller's
# payload (potential or field write) reuses it with zero extra spline evals.
@muladd @inline function _bracketed_slot_solve(τ, Δ, rhs, lo, gpu_traj, r_obs, tₖ, τi, τf, n_iters, idx, ::Val{false})
    hi = τf   # the upper bound does not survive the target moving up: rebuilt per slot
    τ = clamp(τ + Δ * rhs, τi, τf)
    v, f, rhs, r_norm, d¹, d², d³, idx = _lightcone_eval(τ, gpu_traj, r_obs, tₖ, idx)
    f > 0 ? (lo = max(lo, τ)) : (hi = min(hi, τ))
    for _ in 1:n_iters
        prop = τ + f * rhs   # Newton proposal (tangent zero-crossing)
        τ = ifelse((prop < lo) | (prop > hi), (lo + hi) / 2, prop)
        v, f, rhs, r_norm, d¹, d², d³, idx = _lightcone_eval(τ, gpu_traj, r_obs, tₖ, idx)
        f > 0 ? (lo = max(lo, τ)) : (hi = min(hi, τ))
    end
    return τ, lo, v, f, rhs, r_norm, d¹, d², d³, idx
end

# Same solve with the spline interval's coefficients held in registers (`coef_reuse = Val(true)`):
# fetched once per slot after the warm-started search, re-evaluated from registers by the Newton
# corrections, refetched only when a correction leaves the interval (a knot crossing). Identical
# arithmetic to the method above ⇒ bit-identical results; trades ~4D live scalars for the loads.
@muladd @inline function _bracketed_slot_solve(τ, Δ, rhs, lo, gpu_traj, r_obs, tₖ, τi, τf, n_iters, idx, ::Val{true})
    hi = τf
    τ = clamp(τ + Δ * rhs, τi, τf)
    itp = gpu_traj.itp
    idx = _searchsorted_left(itp.t, τ, idx)
    coef = _fetch_interval(itp, idx)
    v = _eval_poly(coef, τ)
    f, rhs, r_norm, d¹, d², d³ = _lightcone_from_state(v, r_obs, tₖ)
    f > 0 ? (lo = max(lo, τ)) : (hi = min(hi, τ))
    for _ in 1:n_iters
        prop = τ + f * rhs   # Newton proposal (tangent zero-crossing)
        τ = ifelse((prop < lo) | (prop > hi), (lo + hi) / 2, prop)
        if !_in_interval(coef, τ)
            idx = _searchsorted_left(itp.t, τ, idx)
            coef = _fetch_interval(itp, idx)
        end
        v = _eval_poly(coef, τ)
        f, rhs, r_norm, d¹, d², d³ = _lightcone_from_state(v, r_obs, tₖ)
        f > 0 ? (lo = max(lo, τ)) : (hi = min(hi, τ))
    end
    return τ, lo, v, f, rhs, r_norm, d¹, d², d³, idx
end

# Per-electron AK.foreachindex pass over (Nx × Ny) pixels; same closure pattern
# as _gpu_unified_one_electron! (see the @kernel note at the top of
# kernel_rk4.jl).  The window block computes the same arrival-time edges as the
# RK4 kernel but in screen-relative offsets (t_px = ψ + ρ²/(R + d³) instead of
# x⁰_px = x⁰ + R) — an intentional divergence from kernel_rk4.jl, accepted so
# the light-front conditioning also covers the slot range and the warm-start
# stride.  The RK4 bridge+march is replaced by an Euler predictor +
# fixed-count Newton corrections per slot (uniform trip count → no warp
# divergence).
function _gpu_newton_one_electron!(
        A_buf, gpu_traj,
        x_grid, y_grid, z_screen,
        t_first, δx⁰, N_samples, Nx, Ny,
        τi, τf, pixel_iter, backend, n_iters, coef_reuse = Val(false), n_chunks = 1
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Chunk-major unpacking of the (pixels × chunks) index space: ix fastest, iy, then chunk.
        ix, iy, chunk = _chunk_pixel(i_lin, Nx, Ny, n_chunks)
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Arrival-window edges in screen-relative offsets (shared helper); the
        # τi eval also provides rhs(τi) for the warm-start predictor stride.
        t_i_px, rhs = _window_edge(gpu_traj, r_obs, τi)
        t_f_px, _ = _window_edge(gpu_traj, r_obs, τf)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (t_i_px - t_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (t_f_px - t_first) * inv_δ))

        if k_start > k_end
            return
        end
        # This thread's slice of the executed slots; chunks beyond the range are empty.
        k_first, k_last = _chunk_slots(k_start, k_end, chunk, n_chunks)
        if k_first > k_last
            return
        end

        # Warm start at the window edge: τ(t_i_px) = τi exactly — no RK4 bridge.
        τ = τi

        tₖ = t_first + (k_first - 1) * δx⁰
        Δ = tₖ - t_i_px   # ∈ (0, δx⁰] unless k_start clamped to 1 (chunk 1); the chunk's stride otherwise

        # Bracket lower bound: raised only at sign-verified points (f > 0), so
        # it lower-bounds every later root too (targets increase along k) —
        # carried across slots, the one extra live register.
        lo = τi
        # Spline interval of the previous evaluation: the warm start of the knot search
        # (see _searchsorted_left). Bit-identical to a cold search; saves its dependent loads.
        idx = 1
        # A chunk beyond the first has no anchor at its first slot: solve it here with
        # N_COLD_ITERS safeguarded corrections from the window-edge predictor, then enter the
        # loop with Δ = 0 so its first pass re-solves that slot from the converged τ. Kept
        # outside the loop so the loop body (and its register footprint) is chunk 1's.
        if chunk > 1
            τ, lo, _, _, rhs, _, _, _, _, idx =
                _bracketed_slot_solve(τ, Δ, rhs, lo, gpu_traj, r_obs, tₖ, τi, τf, N_COLD_ITERS, idx, Val(false))
            Δ = zero(Δ)
        end

        for k in k_first:k_last
            τ, lo, v, f, rhs, r_norm, d¹, d², d³, idx =
                _bracketed_slot_solve(τ, Δ, rhs, lo, gpu_traj, r_obs, tₖ, τi, τf, n_iters, idx, coef_reuse)

            # Accumulate from the last residual eval — zero extra spline evals.
            coeff = K * rhs / r_norm   # = K / m_dot(xr, uμ)
            @inbounds A_buf[ix, iy, 1, k] += coeff * v[5]
            @inbounds A_buf[ix, iy, 2, k] += coeff * v[6]
            @inbounds A_buf[ix, iy, 3, k] += coeff * v[7]
            @inbounds A_buf[ix, iy, 4, k] += coeff * v[8]

            tₖ += δx⁰
            Δ = δx⁰
        end
    end
    return
end

"""
    accumulate_potential(trajs, screen, ::GPUKernelNewton, backend; n_iters = 2, sync_per_electron = true)

Newton light-cone GPU path: like the [`GPUKernelRK4`](@ref) unified kernel, but
the retarded proper time at each saveat slot is found by solving the light-cone
condition directly (`n_iters` warm-started Newton corrections per slot) instead
of integrating the retarded-time ODE between slots.  The condition is evaluated
in the screen-relative light-front spelling `f = tₖ − ψ(τ) − ρ²/(R + d³)`
(algebraically identical to `x⁰_target − x⁰(τ) − |r_obs − x⃗(τ)| = 0`; see the
file header), with the sample grid carried as offsets `tₖ = x⁰_k − z_screen`.

Spline-eval budget per slot is `n_iters + 1` (the final residual eval doubles
as the accumulation eval), vs `4·n_substeps + 1` for the RK4 march; and the
per-slot error does not accumulate along the march, so accuracy is set by the
convergence of the last Newton step alone.

`sync_per_electron` as in the `GPUKernelRK4` method.

`buffers` / `finish` drive electron BATCHING: `finish = false` returns the live
[`FieldAccumulator`](@ref) instead of the downloaded cube, and passing it back as `buffers`
accumulates the next batch of electrons into the same device buffers, so the host only ever
holds one batch of trajectory splines. The download happens once, on the call with
`finish = true` (or through [`finish_field`](@ref)); on one device the batched sum is
bit-identical to the single call.

`timer = LaunchTimer()` records a device-event pair per launch (see [`LaunchTimer`](@ref)). `coef_reuse = Val(true)` holds each slot's spline-interval coefficients in registers across the per-slot evaluations (bit-identical; trades ~4D live registers for the coefficient loads, see [`IntervalCoefs`](@ref)).
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelNewton,
        backend::Backend;
        n_iters::Int = 2,
        coef_reuse::Val = Val(false),
        sample_chunks::Int = 1,
        sync_per_electron::Bool = true,
        timer = nothing,
    )
    n_iters ≥ 1 || throw(ArgumentError(
        "n_iters must be ≥ 1 — n_iters = 0 degrades to an unchecked Euler march"))
    sample_chunks ≥ 1 || throw(ArgumentError("sample_chunks must be ≥ 1, got $sample_chunks"))
    step(screen.x⁰_samples) > 0 || throw(ArgumentError(
        "screen.x⁰_samples must be strictly increasing (got step = $(step(screen.x⁰_samples)))"))
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)

    # Pixel-fastest layout (Nx, Ny leading) so warp-adjacent pixel-threads
    # (consecutive ix) write consecutive addresses → coalesced accumulation.
    A_buf = Adapt.adapt(backend, zeros(Nx, Ny, 4, N_samples))

    x⁰_first = first(screen.x⁰_samples)
    # light-front coordinate reformulation
    t_first = x⁰_first - screen.z
    δx⁰ = step(screen.x⁰_samples)

    # Iteration target: one element per pixel. Sentinel array; never read.
    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny, sample_chunks))   # one thread per (pixel, chunk)
    lane = launch_lane(timer, backend)

    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        e0 = launch_tick(timer, backend)
        _gpu_newton_one_electron!(
            A_buf, gpu_traj,
            screen.x_grid, screen.y_grid, screen.z,
            t_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_iters, coef_reuse, sample_chunks,
        )
        launch_tock!(timer, lane, backend, e0)
        sync_per_electron && KernelAbstractions.synchronize(backend)
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    return _download_permuted(A_buf)
end

# ── Field variant of the Newton light-cone kernel ──
# Identical per-slot Newton solve to `_gpu_newton_one_electron!`; each slot then
# writes the split Liénard–Wiechert (E, B) exactly as
# `_gpu_unified_field_one_electron!` does, reusing the converged residual eval
# (`v`, `d`, `r_norm`) for the geometry, plus the unavoidable `a_itp` eval.
function _gpu_newton_field_one_electron!(
        mode::Val, E1_buf, B1_buf, E2_buf, B2_buf, gpu_traj, c,
        x_grid, y_grid, z_screen,
        t_first, δx⁰, N_samples, Nx, Ny,
        τi, τf, pixel_iter, backend, n_iters, coef_reuse = Val(false), n_chunks = 1
    )
    K = gpu_traj.K
    AK.foreachindex(pixel_iter, backend) do i_lin
        # Chunk-major unpacking of the (pixels × chunks) index space: ix fastest, iy, then chunk.
        ix, iy, chunk = _chunk_pixel(i_lin, Nx, Ny, n_chunks)
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)

        # Arrival-window edges in screen-relative offsets (shared helper); the
        # τi eval also provides rhs(τi) for the warm-start predictor stride.
        t_i_px, rhs = _window_edge(gpu_traj, r_obs, τi)
        t_f_px, _ = _window_edge(gpu_traj, r_obs, τf)

        inv_δ = inv(δx⁰)
        # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
        k_start = max(1, floor(Int, (t_i_px - t_first) * inv_δ) + 2)
        k_end = min(N_samples, ceil(Int, (t_f_px - t_first) * inv_δ))

        if k_start > k_end
            return
        end
        # This thread's slice of the executed slots; chunks beyond the range are empty.
        k_first, k_last = _chunk_slots(k_start, k_end, chunk, n_chunks)
        if k_first > k_last
            return
        end

        τ = τi
        tₖ = t_first + (k_first - 1) * δx⁰
        Δ = tₖ - t_i_px

        # Bracket lower bound (see _bracketed_slot_solve) — carried across slots.
        lo = τi
        # Spline interval of the previous evaluation: the warm start of the knot search
        # (see _searchsorted_left). Bit-identical to a cold search; saves its dependent loads.
        idx = 1
        if chunk > 1   # cold start of a later chunk, outside the loop (see the potential kernel)
            τ, lo, _, _, rhs, _, _, _, _, idx =
                _bracketed_slot_solve(τ, Δ, rhs, lo, gpu_traj, r_obs, tₖ, τi, τf, N_COLD_ITERS, idx, Val(false))
            Δ = zero(Δ)
        end

        for k in k_first:k_last
            τ, lo, v, f, rhs, r_norm, d¹, d², d³, idx =
                _bracketed_slot_solve(τ, Δ, rhs, lo, gpu_traj, r_obs, tₖ, τi, τf, n_iters, idx, coef_reuse)

            # Field write from the converged eval (X reuses r_norm and d).
            uμ = SVector{4}(v[5], v[6], v[7], v[8])
            𝔞μ = _eval_at(gpu_traj.a_itp, τ, idx)   # same knots as itp (checked in to_gpu): reuse the interval
            X = SVector{4}(r_norm, d¹, d², d³)
            F_near, F_far = lienard_wiechert_F_split(X, uμ, 𝔞μ, K, c)
            E_near, B_near = extract_EB(F_near, c)
            E_far, B_far = extract_EB(F_far, c)

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

            tₖ += δx⁰
            Δ = δx⁰
        end
    end
    return
end

"""
    accumulate_field(trajs, screen, ::GPUKernelNewton, backend; n_iters = 2, mode = Val(:split), sync_per_electron = true)

Field counterpart of the `GPUKernelNewton` [`accumulate_potential`](@ref)
method: per-slot Newton light-cone solve instead of the RK4 retarded-time
march, otherwise identical in buffers, `mode`, and streaming to the
`GPUKernelRK4` [`accumulate_field`](@ref) method. `timer = LaunchTimer()` records a device-event pair per launch (see [`LaunchTimer`](@ref)). `coef_reuse = Val(true)` holds each slot's spline-interval coefficients in registers across the per-slot evaluations (bit-identical; trades ~4D live registers for the coefficient loads, see [`IntervalCoefs`](@ref)).
"""
function accumulate_field(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelNewton,
        backend::Backend;
        n_iters::Int = 2,
        coef_reuse::Val = Val(false),
        sample_chunks::Int = 1,
        mode::Val = Val(:split),
        sync_per_electron::Bool = true,
        sink = nothing,
        buffers = nothing,
        finish::Bool = true,
        timer = nothing,
    )
    n_iters ≥ 1 || throw(ArgumentError(
        "n_iters must be ≥ 1 — n_iters = 0 degrades to an unchecked Euler march"))
    sample_chunks ≥ 1 || throw(ArgumentError("sample_chunks must be ≥ 1, got $sample_chunks"))
    step(screen.x⁰_samples) > 0 || throw(ArgumentError(
        "screen.x⁰_samples must be strictly increasing (got step = $(step(screen.x⁰_samples)))"))
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)
    c = screen.c

    # Accumulation buffers: a fresh set per call by default, or the caller's persistent
    # `FieldAccumulator` (electron batching — the buffers then outlive this call).
    acc = _field_buffers(buffers, screen, backend, mode)
    E1_buf, B1_buf, E2_buf, B2_buf = acc.E1, acc.B1, acc.E2, acc.B2

    x⁰_first = first(screen.x⁰_samples)
    # light-front coordinate reformulation
    t_first = x⁰_first - screen.z
    δx⁰ = step(screen.x⁰_samples)

    pixel_iter = Adapt.adapt(backend, zeros(Int8, Nx, Ny, sample_chunks))   # one thread per (pixel, chunk)
    lane = launch_lane(timer, backend)

    for traj in trajs
        gpu_traj = Adapt.adapt(backend, to_gpu(traj; with_acceleration = true))
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        e0 = launch_tick(timer, backend)
        _gpu_newton_field_one_electron!(
            mode, E1_buf, B1_buf, E2_buf, B2_buf, gpu_traj, c,
            screen.x_grid, screen.y_grid, screen.z,
            t_first, δx⁰, N_samples, Nx, Ny,
            τi, τf, pixel_iter, backend, n_iters, coef_reuse, sample_chunks,
        )
        launch_tock!(timer, lane, backend, e0)
        sync_per_electron && KernelAbstractions.synchronize(backend)
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
    # download (or hand to `sink`) while they are still alive — see _finish_fields /
    # _collect_fields / _add_fields! in accumulate.jl.
    acc.n_electrons += length(trajs)
    finish || return acc
    return _finish_fields(sink, E1_buf, B1_buf, E2_buf, B2_buf, mode)
end
