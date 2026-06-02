# Trajectory access (thread-safe interpolation wrapper)
struct TrajectoryInterpolant{I, A, R, U, T}
    itp::I          # DataInterpolations interpolant → SVector{8} = [xμ; uμ]
    a_itp::A        # interpolant of the 4-acceleration 𝔞μ = duμ/dτ → SVector{4}
    x_idxs::R       # SVector{4, Int} indices for xμ = (x⁰, x¹, x², x³)
    u_idxs::U       # SVector{4, Int} indices for uμ = (u⁰, u¹, u², u³)
    K::T            # q_e / (4π ε₀ c) — Liénard-Wiechert prefactor
end

# Potential-only convenience constructor: no 4-acceleration spline (`a_itp = nothing`).
# Enough for `accumulate_potential` and the GPUKernelRK4 potential path, which never
# touch 𝔞μ.  `accumulate_field` needs the acceleration, so it requires the 5-arg form
# or the `sol` constructor below (which builds the `a_itp` spline).
TrajectoryInterpolant(itp, x_idxs, u_idxs, K) =
    TrajectoryInterpolant(itp, nothing, x_idxs, u_idxs, K)

function TrajectoryInterpolant(sol::SciMLBase.AbstractODESolution, x_syms, u_syms)
    x_idxs = SVector{4, Int}(variable_index.((sol,), collect(x_syms)))
    u_idxs = SVector{4, Int}(variable_index.((sol,), collect(u_syms)))
    itp = CubicSpline(sol.u, sol.t; extrapolation = ExtrapolationType.Extension)
    # 4-acceleration 𝔞μ = duμ/dτ, sampled from the solver's OWN derivative
    # interpolant at the knots (`sol(t, Val{1})`). Vern9's continuous extension is
    # built to match the RHS D(u) = F_total/m, so these knot values are
    # model-consistent — more accurate than differentiating `itp` afterwards,
    # which would amplify the cubic spline's interpolation error. Storing them in
    # a dedicated spline keeps evaluation a cheap, thread-safe lookup.
    a_knots = [SVector{4}(sol(t, Val{1})[u_idxs]) for t in sol.t]
    a_itp = CubicSpline(a_knots, sol.t; extrapolation = ExtrapolationType.Extension)
    sys = sol.prob.f.sys
    _world = _find_world(sys)
    K = sol.ps[_world.q_e / (4π * _world.ε₀ * _world.c)]
    return TrajectoryInterpolant(itp, a_itp, x_idxs, u_idxs, K)
end

function (t::TrajectoryInterpolant)(τ)
    v = t.itp(τ)
    xμ = v[t.x_idxs]
    uμ = v[t.u_idxs]
    return (xμ, uμ)
end

# 4-position, 4-velocity, and 4-acceleration at proper time τ.  Used by the field
# accumulation (the potential only needs xμ, uμ, so it uses the 2-tuple functor).
function state_with_acceleration(t::TrajectoryInterpolant, τ)
    v = t.itp(τ)
    xμ = v[t.x_idxs]
    uμ = v[t.u_idxs]
    𝔞μ = t.a_itp(τ)
    return (xμ, uμ, 𝔞μ)
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
struct ObserverScreen{G, T, R, C}
    x_grid::G       # e.g., LinRange for x
    y_grid::G       # e.g., LinRange for y
    z::T            # screen distance
    x⁰_samples::R   # uniform observer-time sampling grid
    c::C            # speed of light in the working units (x⁰ = c·t)
end

"""
    ObserverScreen(x_grid, y_grid, z, x⁰_samples; c)

Keyword form requiring the speed of light `c` in the working units (e.g.
`getdefault(world.c)`).  `c` has no default on purpose: the `x⁰ = c·t` axis is
meaningless without the unit system, and a wrong default would silently corrupt
`δt`, FFT frequencies, and `recommended_n_substeps`.
"""
ObserverScreen(x_grid, y_grid, z, x⁰_samples; c) =
    ObserverScreen(x_grid, y_grid, z, x⁰_samples, c)

function Base.show(io::IO, s::ObserverScreen)
    Nx, Ny = length(s.x_grid), length(s.y_grid)
    N = length(s.x⁰_samples)
    δx⁰ = N > 1 ? step(s.x⁰_samples) : 0.0
    return print(io, "ObserverScreen($(Nx)×$(Ny) pixels, z=$(s.z), $(N) time samples, Δx⁰=$(δx⁰), c=$(s.c))")
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
    println(io, "  Δx⁰:          $(δx⁰)")
    println(io, "  c:            $(s.c)")
    return print(io, "  Δt:           $(δx⁰ / s.c)")
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

    function set_pixel(prob, ctx)
        i = ctx.sim_id
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
    proto_integ = init(proto_prob, alg; saveat = x⁰_samples, save_start = false, save_end = false, solve_kwargs...)
    nworkers = Threads.nthreads()
    integ_pool = Channel{typeof(proto_integ)}(nworkers)
    put!(integ_pool, proto_integ)
    for _ in 2:nworkers
        put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, save_start = false, save_end = false, solve_kwargs...))
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

                _accumulate_pixel!(A, traj, screen, integ.sol.u, integ.sol.t, ix, iy)
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
                sol_pixel = rt_sol.u[LI[ix, iy]]
                _accumulate_pixel!(A, traj, screen, sol_pixel.u, sol_pixel.t, ix, iy)
            end
        end
    end

    return A
end

function _accumulate_pixel!(A, traj, screen, τ_samples, t_samples, ix, iy)
    r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
    # Map saveat values back to A's first-axis index. The integrator only
    # saves saveat points within [tspan[1], tspan[2]] — fewer than length(x⁰_samples)
    # for narrow per-pixel arrival windows — so iterating sequentially over
    # τ_samples with k=1,2,… would accumulate radiation into the wrong
    # observer-time slots. Compute the saveat index from t_samples explicitly.
    x⁰_first = first(screen.x⁰_samples)
    N_x⁰ = length(screen.x⁰_samples)
    δx⁰ = (last(screen.x⁰_samples) - x⁰_first) / (N_x⁰ - 1)
    for (k, τ) in enumerate(τ_samples)
        idx = round(Int, (t_samples[k] - x⁰_first) / δx⁰) + 1
        1 ≤ idx ≤ N_x⁰ || continue
        xμ, uμ = traj(τ)
        disp = r_obs - xμ[SA[2, 3, 4]]
        xr = SVector{4}(norm(disp), disp[1], disp[2], disp[3])
        @views A[idx, :, ix, iy] .+= traj.K * uμ ./ m_dot(xr, uμ)
    end
    return
end

"""
    lienard_wiechert_F_split(X, u, 𝔞, K, c) -> (F_vel, F_rad)

Liénard–Wiechert field-strength (Faraday) tensor split **by power of R** into its
velocity (near) and radiation (far) pieces — the form used for numerically clean,
separately-coherent accumulation in [`accumulate_field`](@ref).  With
`Xu = m_dot(X,u) ∝ R` and `X𝔞 = m_dot(X,𝔞) ∝ R`:

    F_vel = K c² (Xᵘuᵛ − Xᵛuᵘ) / (Xu)³                            ∝ 1/R²
    F_rad = K [ (Xᵘ𝔞ᵛ − Xᵛ𝔞ᵘ) / (Xu)²
              − X𝔞 (Xᵘuᵛ − Xᵛuᵘ) / (Xu)³ ]                        ∝ 1/R

`X` is the retarded displacement 4-vector `Xᵘ = xᵘ_obs − xᵘ_src(τᵣ)`
(`= (R, R·n̂)`), `u` the 4-velocity and `𝔞` the 4-acceleration at τᵣ; `K = q/(4πε₀c)`.

The subtlety: the `−X𝔞` term rides the *velocity* bivector `(Xᵘuᵛ − Xᵛuᵘ)` yet
scales as 1/R (because `X𝔞 ∝ R`), so it belongs to the **radiation** field.
[`lienard_wiechert_F`](@ref) groups it with the `c²` term as `(c² − X𝔞)`, which
loses ≈`log₁₀(c²/X𝔞)` significant digits when `X𝔞 ≪ c²` (small a₀); computing
`F_rad` directly never forms that difference.  `F_vel + F_rad` is identically
`lienard_wiechert_F` in exact arithmetic.
"""
function lienard_wiechert_F_split(X, u, 𝔞, K, c)
    Xu = m_dot(X, u)
    X𝔞 = m_dot(X, 𝔞)
    inv_Xu2 = inv(Xu^2)
    inv_Xu3 = inv(Xu^3)
    vel_coef = c^2 * inv_Xu3       # c² piece only → 1/R² near field
    rad_vel_coef = -X𝔞 * inv_Xu3   # −X𝔞 piece → 1/R radiation, on the u-bivector
    F_vel = @SMatrix [
        K * (X[μ] * u[ν] - X[ν] * u[μ]) * vel_coef
            for μ in 1:4, ν in 1:4
    ]
    F_rad = @SMatrix [
        K * (
                (X[μ] * 𝔞[ν] - X[ν] * 𝔞[μ]) * inv_Xu2 +
                (X[μ] * u[ν] - X[ν] * u[μ]) * rad_vel_coef
            )
            for μ in 1:4, ν in 1:4
    ]
    return (F_vel, F_rad)
end

"""
    lienard_wiechert_F(X, u, 𝔞, K, c) -> SMatrix{4,4}

Total Liénard–Wiechert field-strength (Faraday) tensor `F^{μν} = F_vel + F_rad`,
the sum of the two pieces from [`lienard_wiechert_F_split`](@ref):

    F^{μν} = K [ (Xᵘ𝔞ᵛ − Xᵛ𝔞ᵘ) / (Xu)²  +  (c² − X𝔞)(Xᵘuᵛ − Xᵛuᵘ) / (Xu)³ ].

Indices are upper, matching `faraday` and the Lorentz force in `ChargedParticle`.
Use [`lienard_wiechert_F_split`](@ref) when accumulating, to keep the radiation
and velocity fields in separate coherent sums (see [`accumulate_field`](@ref)).
"""
function lienard_wiechert_F(X, u, 𝔞, K, c)
    F_vel, F_rad = lienard_wiechert_F_split(X, u, 𝔞, K, c)
    return F_vel + F_rad
end

"""
    extract_EB(F, c) -> (E, B)

Recover the electric and magnetic 3-vectors from an upper-index Faraday tensor
`F^{μν}` — the inverse of [`faraday`](@ref).  With (+,−,−,−) indexing (slot 1 is
time): `Eⁱ = c·Fⁱ⁰` and `B = (F⁴³, F²⁴, F³²)`.  Linear in `F`, hence commutes
with the electron sum.
"""
function extract_EB(F, c)
    E = c * SVector(F[2, 1], F[3, 1], F[4, 1])
    B = SVector(F[4, 3], F[2, 4], F[3, 2])
    return (E, B)
end

# Typed integrator pool for the retarded-time problem (one integrator per worker
# thread, drawn from a Channel).  Mirrors the setup in `accumulate_potential`.
function _retarded_integ_pool(traj0, screen, alg; solve_kwargs...)
    τi0 = first(traj0.itp.t)
    τf0 = last(traj0.itp.t)
    r0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
    proto = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs, τi0,
        (advanced_time(traj0, τi0, r0), advanced_time(traj0, τf0, r0)),
        (traj0, r0)
    )
    nworkers = Threads.nthreads()
    proto_integ = init(proto, alg; saveat = screen.x⁰_samples, save_start = false, save_end = false, solve_kwargs...)
    pool = Channel{typeof(proto_integ)}(nworkers)
    put!(pool, proto_integ)
    for _ in 2:nworkers
        put!(pool, init(proto, alg; saveat = screen.x⁰_samples, save_start = false, save_end = false, solve_kwargs...))
    end
    return pool
end

"""
    accumulate_field(trajs, screen, alg; solve_kwargs...)

Compute the radiated electromagnetic field on `screen` from electron `trajs`.

For each electron and pixel, solves the retarded-time ODE (as in
[`accumulate_potential`](@ref)), builds the Liénard–Wiechert Faraday tensor at
each observer-time sample via [`lienard_wiechert_F_split`](@ref), and coherently
sums the fields over electrons.  The radiation (1/R) and velocity (1/R²) pieces
are accumulated in **separate** buffers: this avoids the `(c² − X𝔞)` cancellation
at small a₀ and keeps the (near-field-dominated) total from swamping the
radiation sum.  The Faraday tensor is antisymmetric and the fields are *linear*
in it, so each contribution is stored in the compact (E, B) basis rather than the
redundant 16-component matrix (`Σᵢ extract_EB(Fᵢ) = extract_EB(Σᵢ Fᵢ)`).

Returns `(; E, B, E_rad, B_rad)`, each `Array{Float64,4}` of shape
`(N_samples, 3, Nx, Ny)`: `E, B` are the *total* time-domain fields (radiation +
velocity, including their cross term) for [`screen_observables`](@ref); `E_rad,
B_rad` are the radiation field alone, whose observables are the radiated
energy/angular momentum exactly (the velocity is recoverable as `E − E_rad`).
"""
function accumulate_field(trajs::Vector{<:TrajectoryInterpolant}, screen::ObserverScreen, alg; solve_kwargs...)
    N_samples = length(screen.x⁰_samples)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    # Separate coherent accumulators for the radiation (1/R) and velocity (1/R²)
    # fields; summed to the total on return (see `lienard_wiechert_F_split`).
    E_rad = zeros(N_samples, 3, Nx, Ny)
    B_rad = zeros(N_samples, 3, Nx, Ny)
    E_vel = zeros(N_samples, 3, Nx, Ny)
    B_vel = zeros(N_samples, 3, Nx, Ny)

    pool = _retarded_integ_pool(first(trajs), screen, alg; solve_kwargs...)

    for traj in trajs
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)
        Threads.@threads for ix in Base.OneTo(Nx)
            integ = take!(pool)
            for iy in Base.OneTo(Ny)
                r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
                x⁰_i = advanced_time(traj, τi, r_obs)
                x⁰_f = advanced_time(traj, τf, r_obs)

                integ.p = (traj, r_obs)
                reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
                solve!(integ)

                _accumulate_field_pixel!(E_rad, B_rad, E_vel, B_vel, traj, screen, integ.sol.u, integ.sol.t, ix, iy)
            end
            put!(pool, integ)
        end
    end
    E = E_rad .+ E_vel
    B = B_rad .+ B_vel
    return (; E, B, E_rad, B_rad)
end

function _accumulate_field_pixel!(E_rad, B_rad, E_vel, B_vel, traj, screen, τ_samples, t_samples, ix, iy)
    r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
    c = screen.c
    x⁰_first = first(screen.x⁰_samples)
    N_x⁰ = length(screen.x⁰_samples)
    δx⁰ = (last(screen.x⁰_samples) - x⁰_first) / (N_x⁰ - 1)
    for (k, τ) in enumerate(τ_samples)
        idx = round(Int, (t_samples[k] - x⁰_first) / δx⁰) + 1
        1 ≤ idx ≤ N_x⁰ || continue
        xμ, uμ, 𝔞μ = state_with_acceleration(traj, τ)
        disp = r_obs - xμ[SA[2, 3, 4]]
        X = SVector{4}(norm(disp), disp[1], disp[2], disp[3])
        F_vel, F_rad = lienard_wiechert_F_split(X, uμ, 𝔞μ, traj.K, c)
        Eᵥ, Bᵥ = extract_EB(F_vel, c)
        Eᵣ, Bᵣ = extract_EB(F_rad, c)
        @views E_vel[idx, :, ix, iy] .+= Eᵥ
        @views B_vel[idx, :, ix, iy] .+= Bᵥ
        @views E_rad[idx, :, ix, iy] .+= Eᵣ
        @views B_rad[idx, :, ix, iy] .+= Bᵣ
    end
    return
end

# The Faraday tensor `faraday` and stress-energy tensor `stress_energy` are the
# single definitions shared with the symbolic models (`fields.jl`); the metric is
# the package-wide constant `η`.

"""
    angular_momentum_flux_z(T, r) -> Real

z-component of the radiated angular-momentum flux density crossing the screen, at
screen point `r = (x, y, z)`, derived from the stress-energy tensor `T^{μν}`.
Summed over the screen (× dA) and observer-time samples (× dt) it gives the total
radiated `L_z`; divided by the radiated energy it gives the OAM per photon.

Uses the exact Maxwell-stress form `x Tᶻʸ − y Tᶻˣ` (the z-row of `T`), the
covariant angular-momentum flux `M^{z x y} = xᵘ Tᶻᵛ − xᵛ Tᶻᵘ`; no far-field
approximation. With slots 1=time, 2,3,4 = x,y,z: `Tᶻʸ = T[4,3]`, `Tᶻˣ = T[4,2]`.
"""
function angular_momentum_flux_z(T, r)
    return r[1] * T[4, 3] - r[2] * T[4, 2]
end

"""
    screen_observables(field, screen; ε₀) -> NamedTuple

Derive radiation diagnostics from the accumulated `field = (; E, B)` (the output
of [`accumulate_field`]) as functions of the observer-time sample index — all
following from the Faraday tensor through the electromagnetic stress-energy
tensor `Tᵘᵛ`:

  * `S`              — Poynting vector,             `(N, 3, Nx, Ny)`
  * `energy_density` — `u = ½ε₀(E² + c²B²)`,        `(N, Nx, Ny)`
  * `Lz_density`     — z angular-momentum flux dens, `(N, Nx, Ny)`
  * `energy_total`   — `∫∫ Sᶻ dA dt` (energy through the screen)
  * `Lz_total`       — `∫∫ Lz_density dA dt`

Each pixel's field is reassembled into the Faraday tensor [`faraday`](@ref), then
`T^{μν}` is formed covariantly via [`stress_energy`](@ref); S, energy density, and
the angular-momentum flux [`angular_momentum_flux_z`](@ref) are read off `T`.
"""
function screen_observables(field, screen; ε₀)
    E, B = field.E, field.B
    c = screen.c
    μ₀ = 1 / (ε₀ * c^2)
    N, _, Nx, Ny = size(E)
    S = zeros(N, 3, Nx, Ny)
    u = zeros(N, Nx, Ny)
    Lz = zeros(N, Nx, Ny)
    for iy in Base.OneTo(Ny), ix in Base.OneTo(Nx)
        r = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
        for k in Base.OneTo(N)
            Ev = SVector{3}(E[k, 1, ix, iy], E[k, 2, ix, iy], E[k, 3, ix, iy])
            Bv = SVector{3}(B[k, 1, ix, iy], B[k, 2, ix, iy], B[k, 3, ix, iy])
            F = faraday(Ev, Bv, c)
            T = stress_energy(F, η, μ₀)
            u[k, ix, iy] = T[1, 1]                                            # T⁰⁰ energy density
            @views S[k, :, ix, iy] .= c .* SVector(T[1, 2], T[1, 3], T[1, 4]) # Sⁱ = c·T⁰ⁱ
            Lz[k, ix, iy] = angular_momentum_flux_z(T, r)
        end
    end
    dA = step(screen.x_grid) * step(screen.y_grid)
    dt = step(screen.x⁰_samples) / c
    energy_total = sum(@view S[:, 3, :, :]) * dA * dt
    Lz_total = sum(Lz) * dA * dt
    return (; S, energy_density = u, Lz_density = Lz, energy_total, Lz_total)
end

"""
    screen_spectrum(field, screen; ε₀, bins = nothing) -> NamedTuple

Frequency-domain radiation diagnostics from the accumulated `field = (; E, B)`.

The FFT is *linear*, so it acts on the fields `E, B` (not on the quadratic
observables); every component of the stress-energy tensor then has a
frequency-domain image `T̃(ω)` that is a bilinear cross-spectral density
`Re[F̃*(ω) F̃(ω)]` of the transformed field — assembled here per kept bin:

  * `freqs`     — frequencies (1/time units) at the kept bins
  * `E_ω, B_ω`  — complex field spectra,         `(n_bins, 3, Nx, Ny)`
  * `energy_ω`  — `½ε₀(|Ẽ|² + c²|B̃|²)`,          `(n_bins, Nx, Ny)`
  * `S_ω`       — `Re[Ẽ*×B̃] / μ₀`,               `(n_bins, 3, Nx, Ny)`
  * `Lz_ω`      — `x T̃ᶻʸ − y T̃ᶻˣ`,               `(n_bins, Nx, Ny)`

`bins` selects which `rfft` frequency indices to keep (e.g. harmonic bins located
via `rfftfreq`); `nothing` keeps all. Pass a small `bins` when only a few
frequencies are needed — the complex `E_ω, B_ω` are full grid-sized per bin, so
keeping every bin materialises arrays as large as `E` itself.

The transform is blocked over screen rows, so the transient complex array is
`(N÷2+1, 3, Nx)`, never the full grid. Spectra use the δt-scaled `rfft`
(`Ẽ ≈ ∫E e^{iωt} dt`); the phase is independent of this scaling, so a phase map of
a component at bin `m` is simply `angle.(E_ω[m, i, :, :])`.
"""
function screen_spectrum(field, screen; ε₀, bins = nothing)
    E, B = field.E, field.B
    c = screen.c
    μ₀ = 1 / (ε₀ * c^2)
    N, _, Nx, Ny = size(E)
    δt = step(screen.x⁰_samples) / c
    freqs = rfftfreq(N, 1 / δt)
    sel = isnothing(bins) ? (1:length(freqs)) : bins
    nb = length(sel)

    E_ω = zeros(ComplexF64, nb, 3, Nx, Ny)
    B_ω = zeros(ComplexF64, nb, 3, Nx, Ny)
    energy_ω = zeros(nb, Nx, Ny)
    S_ω = zeros(nb, 3, Nx, Ny)
    Lz_ω = zeros(nb, Nx, Ny)

    # Block over screen rows: rfft one (N, 3, Nx) slab at a time so the transient
    # complex array stays (N÷2+1, 3, Nx) rather than the full grid.
    for iy in Base.OneTo(Ny)
        Ê = δt .* rfft(E[:, :, :, iy], 1)
        B̂ = δt .* rfft(B[:, :, :, iy], 1)
        for ix in Base.OneTo(Nx)
            x = screen.x_grid[ix]
            y = screen.y_grid[iy]
            for (mi, m) in enumerate(sel)
                Ev = SVector{3, ComplexF64}(Ê[m, 1, ix], Ê[m, 2, ix], Ê[m, 3, ix])
                Bv = SVector{3, ComplexF64}(B̂[m, 1, ix], B̂[m, 2, ix], B̂[m, 3, ix])
                @views E_ω[mi, :, ix, iy] .= Ev
                @views B_ω[mi, :, ix, iy] .= Bv
                energy_ω[mi, ix, iy] = (ε₀ / 2) * (real(dot(Ev, Ev)) + c^2 * real(dot(Bv, Bv)))
                @views S_ω[mi, :, ix, iy] .= (1 / μ₀) .* real.(cross(conj.(Ev), Bv))
                # T̃ᶻʲ = −ε₀ Re[Ẽ_z* Ẽ_j + c² B̃_z* B̃_j] (off-diagonal Maxwell stress)
                T̃zy = -ε₀ * real(conj(Ev[3]) * Ev[2] + c^2 * conj(Bv[3]) * Bv[2])
                T̃zx = -ε₀ * real(conj(Ev[3]) * Ev[1] + c^2 * conj(Bv[3]) * Bv[1])
                Lz_ω[mi, ix, iy] = x * T̃zy - y * T̃zx
            end
        end
    end
    return (; freqs = freqs[sel], E_ω, B_ω, energy_ω, S_ω, Lz_ω)
end
