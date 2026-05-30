"""
    accumulate_potential(trajs, screen, alg, backend::Backend; solve_kwargs...)

Compute the Liénard-Wiechert 4-potential using CPU retarded-time solve
and GPU-accelerated accumulation via AcceleratedKernels.

`backend` is a KernelAbstractions backend (e.g., `CUDA.CUDABackend()`).
Uses the original `trajs` (CubicSpline-based) for the CPU retarded-time solve,
and converts to `GPUCubicSpline` internally for the GPU accumulation phase.
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen, alg, backend::Backend;
        solve_kwargs...
    )
    x⁰_samples = screen.x⁰_samples
    N_samples = length(x⁰_samples)
    N_x⁰ = N_samples
    x⁰_first = first(x⁰_samples)
    δx⁰_step = (last(x⁰_samples) - x⁰_first) / (N_samples - 1)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    A = zeros(N_samples, 4, Nx, Ny)

    # Pre-allocate GPU buffers (reused across electrons)
    τ_buf = Adapt.adapt(backend, fill(NaN, N_samples, Nx, Ny))
    A_buf = Adapt.adapt(backend, zeros(N_samples, 4, Nx, Ny))

    τ_all = fill(NaN, N_samples, Nx, Ny)

    # Create typed integrator pool once (reused across all electrons)
    traj0 = first(trajs)
    τi0 = first(traj0.itp.t)
    τf0 = last(traj0.itp.t)
    r_obs_0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
    proto_prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs, τi0,
        (advanced_time(traj0, τi0, r_obs_0), advanced_time(traj0, τf0, r_obs_0)),
        (traj0, r_obs_0)
    )
    proto_integ = init(proto_prob, alg; saveat = x⁰_samples, save_start = false, save_end = false, solve_kwargs...)
    nworkers = Threads.nthreads()
    integ_pool = Channel{typeof(proto_integ)}(nworkers)
    put!(integ_pool, proto_integ)
    for _ in 2:nworkers
        put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, save_start = false, save_end = false, solve_kwargs...))
    end

    # Per-electron streaming: upload one trajectory's spline at a time and
    # free it after the kernel completes.  Memory footprint is bounded by
    # `A_buf + τ_buf + one trajectory + small`, independent of N_macro.
    for traj in trajs
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)

        # ── Phase 1: CPU retarded-time solve ──
        fill!(τ_all, NaN)

        Threads.@threads for ix in Base.OneTo(Nx)
            integ = take!(integ_pool)
            for iy in Base.OneTo(Ny)
                r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
                x⁰_i = advanced_time(traj, τi, r_obs)
                x⁰_f = advanced_time(traj, τf, r_obs)

                integ.p = (traj, r_obs)
                reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
                solve!(integ)

                # Map saveat values back to τ_all's first-axis index. When
                # the per-pixel tspan is much narrower than x⁰_samples, the
                # integrator's sol.u contains only the saveat points within
                # [x⁰_i, x⁰_f] — a subset, possibly indexed in the middle of
                # x⁰_samples. Sequential k=1,2,… would misroute the data.
                sol_u = integ.sol.u::Vector{Float64}
                sol_t = integ.sol.t::Vector{Float64}
                for k in eachindex(sol_u)
                    idx = round(Int, (sol_t[k] - x⁰_first) / δx⁰_step) + 1
                    1 ≤ idx ≤ N_x⁰ || continue
                    @inbounds τ_all[idx, ix, iy] = sol_u[k]
                end
            end
            put!(integ_pool, integ)
        end

        # ── Phase 2: upload this trajectory + GPU accumulation ──
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        _gpu_accumulate_kernel!(gpu_traj, screen, τ_all, τ_buf, A_buf, backend)

        # Stream-aware async free: CUDA.jl queues the free behind the kernel
        # that still reads these buffers, so no explicit sync is needed —
        # syncing here would serialize CPU Phase-1(N+1) against GPU kernel(N)
        # and kill the pipeline.  The trailing `Array(A_buf)` syncs on exit.
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    # Single D2H transfer at the end
    copyto!(A, Array(A_buf))
    return A
end

function _gpu_accumulate_kernel!(gpu_traj, screen, τ_all_cpu, τ_buf, A_buf, backend)
    # Copy retarded times to pre-allocated GPU buffer (H2D)
    copyto!(τ_buf, τ_all_cpu)

    # Capture screen grids (isbits LinRange — no adaptation needed)
    x_grid = screen.x_grid
    y_grid = screen.y_grid
    z_screen = screen.z

    # Capture trajectory components
    spline = gpu_traj.itp
    x_idxs = gpu_traj.x_idxs
    u_idxs = gpu_traj.u_idxs
    K = gpu_traj.K

    # Pre-compute CartesianIndices outside closure
    CI = CartesianIndices(τ_buf)

    # Kernel accumulates into A_buf (no zeroing — accumulates across electrons)
    AK.foreachindex(τ_buf, backend) do i
        k, ix, iy = Tuple(CI[i])

        τ = τ_buf[k, ix, iy]
        isnan(τ) && return

        v = spline(τ)
        xμ = v[x_idxs]
        uμ = v[u_idxs]

        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)
        disp = r_obs - xμ[SA[2, 3, 4]]
        xr = SVector{4}(norm(disp), disp[1], disp[2], disp[3])

        coeff = K / m_dot(xr, uμ)
        for j in Base.OneTo(4)
            @inbounds A_buf[k, j, ix, iy] += coeff * uμ[j]
        end
    end
    return
end
