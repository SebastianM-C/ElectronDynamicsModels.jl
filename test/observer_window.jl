# observer_window_start / trajectory_span_for_window: the observer window opens at the LATEST
# first arrival (corner pixel, far-rim electron) and the solve span covers every sample at every
# pixel — checked against the kernels' own strict-interior slot formulas by brute force.
using ElectronDynamicsModels
using Test

# Kernel slot formulas (kernel_newton.jl / kernel_rk4.jl): light-front offsets against
# t_first = x⁰_first − z; k_start = floor(Δ/δ) + 2, k_end = ceil(Δ/δ).
k_start(t_px, t_first, δ) = floor(Int, (t_px - t_first) / δ) + 2
k_end(t_px, t_first, δ) = ceil(Int, (t_px - t_first) / δ)

@testset "observer window covers the corner" begin
    c = 1.0
    Z, hw, Rmax = 50.0, 8.0, 3.0
    τi, τf = -4.0, 4.0
    δ = 0.05
    N_samples = 400
    grid = LinRange(-hw, hw, 9)
    rim = [(Rmax * cos(θ), Rmax * sin(θ)) for θ in range(0, 2π; length = 97)]
    arrival(τ, px, e) = c * τ + hypot(Z, hypot(px[1] - e[1], px[2] - e[2]))   # electron at rest

    x0 = observer_window_start(τi, Z, hw, Rmax; c)
    latest = maximum(arrival(τi, (x, y), e) for x in grid, y in grid, e in rim)
    @test x0 ≈ latest atol = 1e-9                           # the bound is attained (corner + far rim)
    @test x0 > c * τi + hypot(Z, hw + Rmax)                 # strictly later than the old edge anchor

    xs = range(x0; step = δ, length = N_samples)
    τ_lo, τ_hi = trajectory_span_for_window(τi, τf, xs, Z, hw, Rmax; c)
    @test τ_lo < τi && τ_hi > τf
    t_first = first(xs) - Z
    for x in grid, y in grid, e in vcat(rim, [(0.0, 0.0), (1.0, -0.5)])
        @test k_start(arrival(τ_lo, (x, y), e) - Z, t_first, δ) <= 1
        @test k_end(arrival(τ_hi, (x, y), e) - Z, t_first, δ) >= N_samples
    end
    # Regression guard for the edge-anchored window: the corner pixel is clipped at the head.
    x0_edge = c * τi + hypot(Z, hw + Rmax)
    t_first_edge = x0_edge - Z
    e_far = (-Rmax / √2, -Rmax / √2)
    @test k_start(arrival(τi, (hw, hw), e_far) - Z, t_first_edge, δ) > 1
    # …and the physics span alone leaves the last samples of a long window uncovered.
    @test k_end(arrival(τf, (0.0, 0.0), (0.0, 0.0)) - Z, t_first, δ) < N_samples

    # A window the physics span already covers is left alone.
    short = range(x0; step = δ, length = 40)
    @test trajectory_span_for_window(τi, τf, short, Z, hw, Rmax; c, margin_samples = 0) == (τi, τf)

    # Boosted flight toward the screen: light-front advance c·τ/stretch ⇒ span scales with stretch.
    s = 20.0
    τ_lo_s, τ_hi_s = trajectory_span_for_window(τi, τf, xs, Z, hw, Rmax; c, stretch = s)
    @test τ_hi_s ≈ s * τ_hi
    @test τ_lo_s ≈ s * τ_lo

    @test_throws ArgumentError trajectory_span_for_window(τi, τf, xs, Z, hw, Rmax; c, margin_samples = -1)
    @test_throws ArgumentError trajectory_span_for_window(τi, τf, xs, Z, hw, Rmax; c, stretch = 0)
end
