# Trajectory access (thread-safe interpolation wrapper)
struct TrajectoryInterpolant{I, R, U}
    itp::I          # DataInterpolations interpolant (SVector{8} → SVector{8})
    r_idxs::R       # SVector{4, Int} for x⁰, x¹, x², x³
    u_idxs::U       # SVector{4, Int} for u⁰, u¹, u², u³
end

function TrajectoryInterpolant(sol::SciMLBase.AbstractODESolution, x_syms, u_syms)
    r_idxs = SVector{4, Int}(variable_index.((sol,), collect(x_syms)))
    u_idxs = SVector{4, Int}(variable_index.((sol,), collect(u_syms)))
    itp = CubicSpline(sol.u, sol.t; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, r_idxs, u_idxs)
end

function (t::TrajectoryInterpolant)(τ)
    v = t.itp(τ)
    rμ = v[t.r_idxs]
    uμ = v[t.u_idxs]
    return (rμ, uμ)
end

"""
    trajectory_interpolants(ensemble_sol::EnsembleSolution)

Return a vector of `TrajectoryInterpolant` objects, which interpolate the trajectories
in a fast and thread safe manner.
"""
function trajectory_interpolants(ensemble_sol::EnsembleSolution)
    sys = ensemble_sol.u[1].prob.f.sys
    return [TrajectoryInterpolant(ensemble_sol.u[i], sys.x, sys.u) for i in axes(ensemble_sol.u, 1)]
end

# Observer geometry + temporal sampling
struct ObserverScreen{G, T, R}
    x_grid::G       # e.g., LinRange for x
    y_grid::G       # e.g., LinRange for y
    z::T            # screen distance
    x⁰_samples::R  # uniform observer-time sampling grid
end

# dτᵣ/dt = 1/(u⁰(τᵣ) - u⃗(τᵣ)·n̂(τᵣ, r_obs))
function retarded_time_rhs(τᵣ, p, t)
    traj, r_obs = p
    rμ, uμ = traj(τᵣ)
    rⁱ = rμ[SA[2, 3, 4]]
    n̂ = (r_obs - rⁱ) * inv(norm(r_obs - rⁱ))
    return inv(uμ[1] - uμ[SA[2, 3, 4]] ⋅ n̂)
end

function advanced_time(traj, τ, x_obs)
    rμ, _ = traj(τ)
    return rμ[1] + norm(x_obs - rμ[SA[2, 3, 4]])
end

function retarded_time_problem(traj::TrajectoryInterpolant, screen::ObserverScreen)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    CI = CartesianIndices((Nx, Ny))

    r_obs_0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
    τi = first(traj.itp.t)
    τf = last(traj.itp.t)
    (x⁰_i_0, x⁰_f_0) = advanced_time(traj, τi, r_obs_0), advanced_time(traj, τf, r_obs_0)

    prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs,
        τi,
        (x⁰_i_0, x⁰_f_0),
        (traj, r_obs_0),
    )

    function set_pixel(prob, i, repeat)
        ix, iy = Tuple(CI[i])
        r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
        (x⁰_i, x⁰_f) = advanced_time(traj, τi, r_obs), advanced_time(traj, τf, r_obs)
        return remake(prob; p = (traj, r_obs), tspan = (x⁰_i, x⁰_f))
    end

    return EnsembleProblem(prob; prob_func = set_pixel, safetycopy = false)
end

function accumulate_potential(trajs, screen, K; alg, ensemblealg = nothing, solve_kwargs...)
    x⁰_samples = screen.x⁰_samples
    N_samples = length(x⁰_samples)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    A = zeros(N_samples, 4, Nx, Ny)

    for (j, traj) in enumerate(trajs)
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)

        if ensemblealg !== nothing
            # GPU / custom ensemble path
            rt_prob = retarded_time_problem(traj, screen)
            N_pixels = Nx * Ny
            rt_sol = solve(
                rt_prob, alg, ensemblealg;
                trajectories = N_pixels, saveat = x⁰_samples, solve_kwargs...
            )
            LI = LinearIndices((Nx, Ny))

            Threads.@threads for ix in Base.OneTo(Nx)
                for iy in Base.OneTo(Ny)
                    _accumulate_pixel!(A, traj, screen, K, rt_sol.u[LI[ix, iy]].u, ix, iy)
                end
            end
        else
            # CPU path: reinit! to avoid repeated solver initialization
            r_obs_0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
            x⁰_i_0 = advanced_time(traj, τi, r_obs_0)
            x⁰_f_0 = advanced_time(traj, τf, r_obs_0)
            proto_prob = ODEProblem{false, SciMLBase.FullSpecialize}(
                retarded_time_rhs, τi, (x⁰_i_0, x⁰_f_0), (traj, r_obs_0)
            )

            nworkers = Threads.nthreads()
            integ_pool = Channel{Any}(nworkers)
            for _ in 1:nworkers
                put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, solve_kwargs...))
            end

            Threads.@threads for ix in Base.OneTo(Nx)
                integ = take!(integ_pool)
                for iy in Base.OneTo(Ny)
                    r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
                    x⁰_i = advanced_time(traj, τi, r_obs)
                    x⁰_f = advanced_time(traj, τf, r_obs)

                    integ.p = (traj, r_obs)
                    reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
                    solve!(integ)

                    _accumulate_pixel!(A, traj, screen, K, integ.sol.u, ix, iy)
                end
                put!(integ_pool, integ)
            end
        end
    end

    return A
end

function _accumulate_pixel!(A, traj, screen, K, τ_samples, ix, iy)
    r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
    for (k, τ) in enumerate(τ_samples)
        rμ, uμ = traj(τ)
        disp = r_obs - rμ[SA[2, 3, 4]]
        xR = SVector{4}(norm(disp), disp[1], disp[2], disp[3])
        @views A[k, :, ix, iy] .+= K * uμ ./ m_dot(xR, uμ)
    end
    return
end
