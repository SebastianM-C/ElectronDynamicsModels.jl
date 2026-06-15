# ── Batched / electron-parallel GPU field accumulation (grouped private buffers) ──
# An alternative parallelization of `accumulate_field` for large-VRAM GPUs (MI300X 192 GB,
# H200 144 GB), where a single-electron launch underfills the device (~0.59 occupancy at NX=400).
# `Npix × n_groups` threads: group `g` owns a private accumulator slice `buf[:,:,:,:,g]` and
# serially marches the electrons `e = g, g+G, …` of the batch into it — no atomics, no write
# conflicts. `n_groups = 1` is the pure batched electron-loop; `n_groups = G` is G-way electron
# parallelism paid in VRAM (G × the accumulator footprint), collapsed on-device at the end. Reuses
# the flat `BatchedGPUSplines` storage from the Experimental path (one packed upload per batch,
# vs the per-electron upload of the streaming GPUKernelRK4). Fixed-step RK4 marcher only.

using .Experimental: BatchedGPUSplines, _eval_batched_spline, _rt_rhs_batched

"""
    GPUKernelBatched

Sentinel solver type selecting the grouped/electron-parallel field kernel. Strategy is chosen via
keywords of [`accumulate_field`](@ref): `batch_size`, `n_groups`, `mode`, `n_substeps`.
"""
struct GPUKernelBatched end

# ── Generic flat packer (state D=8 and acceleration D=4 splines) ──
# Like `Experimental.to_batched_cpu`, but takes raw CubicSpline interpolants so the same code packs
# both `traj.itp` and `traj.a_itp`.
function _pack_splines(
        itps::AbstractVector, Ks::Vector{Float64},
        x_idxs::SVector{4, Int}, u_idxs::SVector{4, Int}
    )
    B = length(itps)
    Ns_vec = [length(itp.t) for itp in itps]
    offsets = Vector{Int}(undef, B + 1)
    offsets[1] = 0
    for e in 1:B
        offsets[e + 1] = offsets[e] + Ns_vec[e]
    end
    total_N = offsets[end]
    D = length(first(first(itps).u))

    t_flat = Vector{Float64}(undef, total_N)
    h_flat = Vector{Float64}(undef, total_N)
    z_flat = Matrix{Float64}(undef, total_N, D)
    c1_flat = zeros(Float64, total_N, D)   # last row per electron unused; zero is safe
    c2_flat = zeros(Float64, total_N, D)
    τis = Vector{Float64}(undef, B)
    τfs = Vector{Float64}(undef, B)

    for (e, itp) in enumerate(itps)
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
        τis[e] = itp.t[1]
        τfs[e] = itp.t[N]
    end

    return BatchedGPUSplines{D, typeof(t_flat), typeof(z_flat), typeof(offsets), typeof(Ks)}(
        t_flat, h_flat, z_flat, c1_flat, c2_flat,
        offsets, Ns_vec, Ks, τis, τfs, x_idxs, u_idxs,
    )
end

function _free_batched!(s::BatchedGPUSplines)
    finalize(s.t)
    finalize(s.h)
    finalize(s.z)
    finalize(s.c1)
    finalize(s.c2)
    finalize(s.offsets)
    finalize(s.Ns)
    finalize(s.Ks)
    finalize(s.τis)
    finalize(s.τfs)
    return
end

# Classical RK4 step on the batched-spline retarded-time RHS.
@inline function _rk4_step_batched(τ, dt, splines, e, r_obs)
    k1 = _rt_rhs_batched(τ, splines, e, r_obs)
    k2 = _rt_rhs_batched(τ + 0.5 * dt * k1, splines, e, r_obs)
    k3 = _rt_rhs_batched(τ + 0.5 * dt * k2, splines, e, r_obs)
    k4 = _rt_rhs_batched(τ + dt * k3, splines, e, r_obs)
    return τ + dt * (k1 + 2 * (k2 + k3) + k4) * (1 / 6)
end

# Full retarded-time march + LW field accumulation for ONE (electron, pixel) pair, writing into
# group slot `g` of the 5-D buffers. `MODE ∈ {:split, :total}` folds at compile time (split keeps
# far in buffer 1 / near in buffer 2; total sums them into buffer 1, with E2/B2 aliasing E1/B1).
@inline function _field_march_pair!(
        ::Val{MODE}, E1, B1, E2, B2, g,
        st, ac, e, c, r_obs, ix, iy,
        x⁰_first, δx⁰, N_samples, n_substeps,
    ) where {MODE}
    @inbounds K = st.Ks[e]
    @inbounds τi = st.τis[e]
    @inbounds τf = st.τfs[e]

    # Pixel-specific advanced-time window
    v_i = _eval_batched_spline(st, e, τi)
    @inbounds d_i¹ = r_obs[1] - v_i[st.x_idxs[2]]
    @inbounds d_i² = r_obs[2] - v_i[st.x_idxs[3]]
    @inbounds d_i³ = r_obs[3] - v_i[st.x_idxs[4]]
    @inbounds x⁰_i_px = v_i[st.x_idxs[1]] + sqrt(d_i¹ * d_i¹ + d_i² * d_i² + d_i³ * d_i³)

    v_f = _eval_batched_spline(st, e, τf)
    @inbounds d_f¹ = r_obs[1] - v_f[st.x_idxs[2]]
    @inbounds d_f² = r_obs[2] - v_f[st.x_idxs[3]]
    @inbounds d_f³ = r_obs[3] - v_f[st.x_idxs[4]]
    @inbounds x⁰_f_px = v_f[st.x_idxs[1]] + sqrt(d_f¹ * d_f¹ + d_f² * d_f² + d_f³ * d_f³)

    inv_δ = inv(δx⁰)
    # Strict-interior slot range matching Tsit5 with save_start/save_end=false.
    k_start = max(1, floor(Int, (x⁰_i_px - x⁰_first) * inv_δ) + 2)
    k_end = min(N_samples, ceil(Int, (x⁰_f_px - x⁰_first) * inv_δ))
    if k_start > k_end
        return
    end

    # Bridge τ from τi up to observer time x⁰_samples[k_start].
    τ = τi
    bridge_dt = x⁰_first + (k_start - 1) * δx⁰ - x⁰_i_px
    if bridge_dt > 0
        sub_dt = bridge_dt / n_substeps
        for _ in 1:n_substeps
            τ = _rk4_step_batched(τ, sub_dt, st, e, r_obs)
        end
        τ = clamp(τ, τi, τf)
    end

    # March through saveat slots, accumulating the split LW field at each.
    for k in k_start:k_end
        τ_safe = clamp(τ, τi, τf)
        v = _eval_batched_spline(st, e, τ_safe)
        𝔞μ = _eval_batched_spline(ac, e, τ_safe)
        xμ = v[st.x_idxs]
        uμ = v[st.u_idxs]

        disp = r_obs - xμ[SA[2, 3, 4]]
        X = SVector{4}(norm(disp), disp[1], disp[2], disp[3])
        F_near, F_far = lienard_wiechert_F_split(X, uμ, 𝔞μ, K, c)
        E_near, B_near = extract_EB(F_near, c)
        E_far, B_far = extract_EB(F_far, c)

        for j in 1:3
            if MODE === :split
                @inbounds E1[ix, iy, j, k, g] += E_far[j]
                @inbounds B1[ix, iy, j, k, g] += B_far[j]
                @inbounds E2[ix, iy, j, k, g] += E_near[j]
                @inbounds B2[ix, iy, j, k, g] += B_near[j]
            else
                @inbounds E1[ix, iy, j, k, g] += E_far[j] + E_near[j]
                @inbounds B1[ix, iy, j, k, g] += B_far[j] + B_near[j]
            end
        end

        if k < k_end
            sub_dt = δx⁰ / n_substeps
            for _ in 1:n_substeps
                τ = _rk4_step_batched(τ, sub_dt, st, e, r_obs)
            end
            τ = clamp(τ, τi, τf)
        end
    end
    return
end

# Grouped kernel: Npix × G threads; thread (pixel, g) serially marches the electrons
# e = g, g+G, … of the batch into its private buffer slice g.
function _gpu_batched_field_grouped!(
        mode::Val, E1, B1, E2, B2, st, ac, c,
        x_grid, y_grid, z_screen,
        x⁰_first, δx⁰, N_samples, Nx, Ny, B, G,
        iter, backend, n_substeps,
    )
    Npix = Nx * Ny
    AK.foreachindex(iter, backend) do i_lin
        ipx = ((i_lin - 1) % Npix) + 1
        g = ((i_lin - 1) ÷ Npix) + 1
        ix = ((ipx - 1) % Nx) + 1
        iy = ((ipx - 1) ÷ Nx) + 1
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)
        e = g
        while e ≤ B
            _field_march_pair!(
                mode, E1, B1, E2, B2, g,
                st, ac, e, c, r_obs, ix, iy,
                x⁰_first, δx⁰, N_samples, n_substeps,
            )
            e += G
        end
    end
    return
end

# Collapse the group dimension on-device: buf[:,:,:,:,1] += buf[:,:,:,:,g].
# Done slice-wise (4-D broadcasts) to stay below 32-bit linear-index limits.
function _collapse_groups!(buf)
    G = size(buf, 5)
    s1 = view(buf, :, :, :, :, 1)
    for g in 2:G
        s1 .+= view(buf, :, :, :, :, g)
    end
    return s1
end

"""
    default_n_groups(screen, vram_bytes; mode = Val(:total))

Choose a group count for [`GPUKernelBatched`](@ref) from the screen geometry and the device memory
budget. Benchmark facts to encode (MI300X, 2026-06): occupancy saturates near
`Npix × G ≈ 2–3 × 10⁵` threads (G beyond that wastes VRAM for no speedup — at Nx = 128 the
G = 16 → 64 step bought < 4%), and each group costs
`(mode == Val(:split) ? 4 : 2) × Nx·Ny·3·N_samples·8` bytes of accumulator.
"""
function default_n_groups(screen::ObserverScreen, vram_bytes::Integer; mode::Val = Val(:total))
    Nx = length(screen.x_grid)
    Ny = length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)
    n_buffers = mode == Val(:split) ? 4 : 2
    # VRAM per group
    vram_bytes_per_group = n_buffers * Nx * Ny * 3 * N_samples * 8
    # number of groups that fit in the VRAM budget (0.9 headroom for partials + host copy)
    G_vram = 0.9vram_bytes ÷ vram_bytes_per_group
    # occupancy saturates near Npix × G ≈ 3e5 threads; more groups just waste VRAM
    saturation_occupancy = 3.0e5
    G_occ = saturation_occupancy ÷ (Nx * Ny)

    return max(1, floor(Int, min(G_vram, G_occ)))
end

"""
    accumulate_field(trajs, screen, ::GPUKernelBatched, backend;
                     batch_size = length(trajs), n_groups = 1, n_substeps = 1, mode = Val(:split))

Grouped/electron-parallel GPU field accumulation. Electrons are packed `batch_size` at a time into
flat `BatchedGPUSplines` storage (state + acceleration splines) and uploaded once per batch,
replacing the per-electron upload/launch/free of the streaming `GPUKernelRK4` path. Launches
`Npix × n_groups` threads; each group accumulates its share of the batch into a private buffer slice
(VRAM cost: `n_groups ×` the accumulator footprint), collapsed on-device at the end. `n_groups = 1`
is the pure batched electron-loop. `n_substeps`/`mode` behave exactly as in the `GPUKernelRK4` path.
"""
function accumulate_field(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen,
        ::GPUKernelBatched,
        backend::Backend;
        batch_size::Int = length(trajs),
        n_groups::Int = 1,
        n_substeps::Int = 1,
        mode::Val = Val(:split),
    )
    all(traj -> traj.a_itp !== nothing, trajs) ||
        throw(ArgumentError("accumulate_field needs trajectories with acceleration splines (a_itp)"))

    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)
    c = screen.c
    G = n_groups

    E1 = KernelAbstractions.zeros(backend, Float64, Nx, Ny, 3, N_samples, G)
    B1 = KernelAbstractions.zeros(backend, Float64, Nx, Ny, 3, N_samples, G)
    if mode == Val(:split)
        E2 = KernelAbstractions.zeros(backend, Float64, Nx, Ny, 3, N_samples, G)
        B2 = KernelAbstractions.zeros(backend, Float64, Nx, Ny, 3, N_samples, G)
    else
        E2 = E1
        B2 = B1
    end

    x⁰_first = first(screen.x⁰_samples)
    δx⁰ = step(screen.x⁰_samples)
    x_grid_d = Adapt.adapt(backend, collect(screen.x_grid))
    y_grid_d = Adapt.adapt(backend, collect(screen.y_grid))

    for chunk_range in Iterators.partition(1:length(trajs), batch_size)
        chunk = view(trajs, chunk_range)
        B = length(chunk)
        Ks = [traj.K for traj in chunk]
        x_idxs = first(chunk).x_idxs
        u_idxs = first(chunk).u_idxs
        st_cpu = _pack_splines([traj.itp for traj in chunk], Ks, x_idxs, u_idxs)
        ac_cpu = _pack_splines([traj.a_itp for traj in chunk], Ks, SA[1, 2, 3, 4], SA[1, 2, 3, 4])
        st = Adapt.adapt(backend, st_cpu)
        ac = Adapt.adapt(backend, ac_cpu)

        n_threads = Nx * Ny * G
        iter = KernelAbstractions.zeros(backend, Int8, n_threads)   # sentinel; never read

        _gpu_batched_field_grouped!(
            mode, E1, B1, E2, B2, st, ac, c,
            x_grid_d, y_grid_d, screen.z,
            x⁰_first, δx⁰, N_samples, Nx, Ny, B, G,
            iter, backend, n_substeps,
        )
        KernelAbstractions.synchronize(backend)
        _free_batched!(st)
        _free_batched!(ac)
        finalize(iter)
    end

    # Collapse groups on-device, then pull back and restore the public layout.
    if mode == Val(:split)
        E_far = permutedims(Array(_collapse_groups!(E1)), (4, 3, 1, 2))
        B_far = permutedims(Array(_collapse_groups!(B1)), (4, 3, 1, 2))
        E_near = permutedims(Array(_collapse_groups!(E2)), (4, 3, 1, 2))
        B_near = permutedims(Array(_collapse_groups!(B2)), (4, 3, 1, 2))
        foreach(finalize, (E1, B1, E2, B2))
        E = E_far .+ E_near
        B = B_far .+ B_near
        return (; E, B, E_far, B_far)
    else
        E = permutedims(Array(_collapse_groups!(E1)), (4, 3, 1, 2))
        B = permutedims(Array(_collapse_groups!(B1)), (4, 3, 1, 2))
        foreach(finalize, (E1, B1))
        return (; E, B)
    end
end
