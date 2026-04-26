# Trajectory access (thread-safe interpolation wrapper)
struct TrajectoryInterpolant{I, R, U, T}
    itp::I          # DataInterpolations interpolant → SVector{8}
    x_idxs::R       # SVector{4, Int} indices for xμ = (x⁰, x¹, x², x³)
    u_idxs::U       # SVector{4, Int} indices for uμ = (u⁰, u¹, u², u³)
    K::T            # q_e / (4π ε₀ c) — Liénard-Wiechert prefactor
end

function TrajectoryInterpolant(sol::SciMLBase.AbstractODESolution, x_syms, u_syms)
    x_idxs = SVector{4, Int}(variable_index.((sol,), collect(x_syms)))
    u_idxs = SVector{4, Int}(variable_index.((sol,), collect(u_syms)))
    itp = CubicSpline(sol.u, sol.t; extrapolation = ExtrapolationType.Extension)
    sys = sol.prob.f.sys
    _world = _find_world(sys)
    K = sol.ps[_world.q_e / (4π * _world.ε₀ * _world.c)]
    return TrajectoryInterpolant(itp, x_idxs, u_idxs, K)
end

function (t::TrajectoryInterpolant)(τ)
    v = t.itp(τ)
    xμ = v[t.x_idxs]
    uμ = v[t.u_idxs]
    return (xμ, uμ)
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
    x⁰_samples::R   # uniform observer-time sampling grid
end

function Base.show(io::IO, s::ObserverScreen)
    Nx, Ny = length(s.x_grid), length(s.y_grid)
    N = length(s.x⁰_samples)
    δx⁰ = N > 1 ? step(s.x⁰_samples) : 0.0
    return print(io, "ObserverScreen($(Nx)×$(Ny) pixels, z=$(s.z), $(N) time samples, Δx⁰=$(δx⁰))")
end

function Base.show(io::IO, ::MIME"text/plain", s::ObserverScreen)
    Nx, Ny = length(s.x_grid), length(s.y_grid)
    N = length(s.x⁰_samples)
    δx⁰ = N > 1 ? step(s.x⁰_samples) : 0.0
    println(io, "ObserverScreen")
    println(io, "  pixels:       $(Nx) × $(Ny)")
    println(io, "  x range:      [$(first(s.x_grid)), $(last(s.x_grid))]")
    println(io, "  y range:      [$(first(s.y_grid)), $(last(s.y_grid))]")
    println(io, "  z:            $(s.z)")
    println(io, "  time samples: $(N)")
    println(io, "  x⁰ range:     [$(first(s.x⁰_samples)), $(last(s.x⁰_samples))]")
    return print(io, "  Δx⁰:          $(δx⁰)")
end

# dτᵣ/dt = 1/(u⁰(τᵣ) - u⃗(τᵣ)·n̂(τᵣ, r_obs))
function retarded_time_rhs(τᵣ, p, t)
    traj, r_obs = p
    xμ, uμ = traj(τᵣ)
    xⁱ = xμ[SA[2, 3, 4]]
    n̂ = (r_obs - xⁱ) * inv(norm(r_obs - xⁱ))
    return inv(uμ[1] - uμ[SA[2, 3, 4]] ⋅ n̂)
end

function advanced_time(traj, τ, r_obs)
    xμ, _ = traj(τ)
    return xμ[1] + norm(r_obs - xμ[SA[2, 3, 4]])
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

"""
    accumulate_potential(trajs, screen, alg; solve_kwargs...)
    accumulate_potential(trajs, screen, alg, ensemblealg; solve_kwargs...)

Compute the Liénard-Wiechert 4-potential on `screen` from electron `trajs`.

For each electron trajectory, solves the retarded-time ODE to map observer time to
proper time, then evaluates `Aμ = K uμ / (xr · u)` at uniform observer-time samples.
Returns `A[k, μ, ix, iy]` — the time-domain 4-potential ready for FFT.

The two-argument `alg` form uses a `reinit!`-based integrator pool for efficient CPU
threading. The four-argument form with `ensemblealg` uses `EnsembleProblem` for
compatibility with GPU backends (e.g., `EnsembleGPUKernel`).

# Arguments
- `trajs`: vector of `TrajectoryInterpolant` from [`trajectory_interpolants`](@ref)
- `screen`: `ObserverScreen` defining pixel grid and observer-time samples
- `alg`: ODE solver algorithm for the retarded-time problem (e.g., `Tsit5()`)
- `ensemblealg`: (optional) ensemble algorithm (e.g., `EnsembleGPUKernel(backend)`)
- `solve_kwargs...`: additional keyword arguments passed to the ODE solver
"""
function accumulate_potential(trajs::Vector{<:TrajectoryInterpolant}, screen::ObserverScreen, alg; solve_kwargs...)
    x⁰_samples = screen.x⁰_samples
    N_samples = length(x⁰_samples)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    A = zeros(N_samples, 4, Nx, Ny)

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
    proto_integ = init(proto_prob, alg; saveat = x⁰_samples, solve_kwargs...)
    nworkers = Threads.nthreads()
    integ_pool = Channel{typeof(proto_integ)}(nworkers)
    put!(integ_pool, proto_integ)
    for _ in 2:nworkers
        put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, solve_kwargs...))
    end

    for traj in trajs
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)

        Threads.@threads for ix in Base.OneTo(Nx)
            integ = take!(integ_pool)
            for iy in Base.OneTo(Ny)
                r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
                x⁰_i = advanced_time(traj, τi, r_obs)
                x⁰_f = advanced_time(traj, τf, r_obs)

                integ.p = (traj, r_obs)
                reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
                solve!(integ)

                _accumulate_pixel!(A, traj, screen, integ.sol.u, ix, iy)
            end
            put!(integ_pool, integ)
        end
    end

    return A
end

# Ensemble path (for GPU / custom ensemble algorithms)
function accumulate_potential(trajs::Vector{<:TrajectoryInterpolant}, screen::ObserverScreen, alg, ensemblealg; solve_kwargs...)
    x⁰_samples = screen.x⁰_samples
    N_samples = length(x⁰_samples)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    A = zeros(N_samples, 4, Nx, Ny)

    for (j, traj) in enumerate(trajs)
        rt_prob = retarded_time_problem(traj, screen)
        N_pixels = Nx * Ny
        rt_sol = solve(
            rt_prob, alg, ensemblealg;
            trajectories = N_pixels, saveat = x⁰_samples, solve_kwargs...
        )
        LI = LinearIndices((Nx, Ny))

        Threads.@threads for ix in Base.OneTo(Nx)
            for iy in Base.OneTo(Ny)
                _accumulate_pixel!(A, traj, screen, rt_sol.u[LI[ix, iy]].u, ix, iy)
            end
        end
    end

    return A
end

function _accumulate_pixel!(A, traj, screen, τ_samples, ix, iy)
    r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
    for (k, τ) in enumerate(τ_samples)
        xμ, uμ = traj(τ)
        disp = r_obs - xμ[SA[2, 3, 4]]
        xr = SVector{4}(norm(disp), disp[1], disp[2], disp[3])
        @views A[k, :, ix, iy] .+= traj.K * uμ ./ m_dot(xr, uμ)
    end
    return
end
