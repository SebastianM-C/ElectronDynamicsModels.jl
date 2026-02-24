# Trajectory access (wraps ODE solution for CPU)
struct TrajectoryInterpolant{S,I}
    sol::S
    r_idxs::I
    u_idxs::I
end
(t::TrajectoryInterpolant)(τ) = (t.sol(τ, idxs = t.r_idxs[2:4]), t.sol(τ, idxs = t.u_idxs[2:4]))

# Convenience: extract all trajectories from ensemble solution
# trajectories(ensemble_sol, sys) → Vector{TrajectoryInterpolant}
function trajectories(ensemble_sol::EnsembleSolution)
    sys = ensemble_sol.u[1].prob.f.sys # sys is the same across the ensemble
    [TrajectoryInterpolant(ensemble_sol.u[i], r_idxs=sys.x, u_idxs=sys.u) for i in ensemble_sol.trajectories]
end

# Observer geometry + temporal sampling
struct ObserverScreen{G,T}
    x_grid::G       # e.g., LinRange for x
    y_grid::G       # e.g., LinRange for y
    z::T            # screen distance
    x⁰_start::T
    δt::T
    N_samples::Int
end

# dτᵣ/dt = 1/(u⁰(τᵣ) - u⃗(τᵣ)·n̂(τᵣ, r_obs))
function retarded_time_rhs(τᵣ, p, t)
    traj, r_obs = p
    rⁱ, uⁱ = traj(τᵣ)
    n̂ = (r_obs - rⁱ) * inv(norm(r_obs - rⁱ))
    inv(u⁰ - uⁱ ⋅ n̂)
end

advanced_time(traj, τ, x_obs) = traj.sol(τ, idxs=traj.r_idxs[1]) + norm(x_obs - traj.sol(τ, idxs=traj.u_idxs[1]))

function retarded_time_problem(e_trajectories::Vector{TrajectoryInterpolant}, screen::ObserverScreen)
    dims = (N_electrons, length(screen.x_grid), length(screen.y_grid))
    CI = CartesianIndices(dims)

    j, ix, iy = Tuple(CI[1])
    traj = e_trajectories[j]
    r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)

    (x⁰_i, x⁰_f) = advanced_time(traj, τi, x), advanced_time(traj, τf, x)
    prob = ODEProblem{false,SciMLBase.FullSpecialize}(
        retarded_time_rhs,
        τi,
        (x⁰_i, x⁰_f),
        (traj, r_obs),
    )

    function set_traj_and_pixel(prob, i, repeat)
        j, ix, iy = Tuple(CI[i])
        traj = e_trajectories[j]
        r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)

        x⁰_i, x⁰_f = advanced_time(traj, τ_span, r_obs)
        remake(prob; p=(traj, r_obs), tspan=(x⁰_i, x⁰_f))
    end

    EnsembleProblem(prob; prob_func = set_traj_and_pixel, safetycopy=false)
end

function accumulate_potential(rt_sol, trajs, screen, K)
    # (N_samples, 4, Nx, Ny)
end
