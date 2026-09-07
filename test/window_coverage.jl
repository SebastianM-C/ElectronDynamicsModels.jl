# Host-side observer-window coverage (src/diagnostics/window_coverage.jl) — CPU only.
using ElectronDynamicsModels
using ElectronDynamicsModels: _window_edge, _window_edge_extremes, _k_start_raw, _k_end_raw, to_gpu,
    _profile_trajectory
using StaticArrays
using ElectronDynamicsModels.KernelAbstractions: CPU   # not a direct dep of every env that runs this
using Test

# All-pixel enumeration of the arrival offsets — the reference for the five-pixel bounds.
function brute_extremes(traj, x_grid, y_grid, z, τ)
    gt = to_gpu(traj)
    ts = [_window_edge(gt, SVector{3}(x, y, z), τ)[1] for x in x_grid, y in y_grid]
    return minimum(ts), maximum(ts)
end

# All-pixel executed-slot count with the kernels' own formulas.
function brute_slots(traj, screen)
    gt = to_gpu(traj)
    Ns = length(screen.x⁰_samples)
    t_first = first(screen.x⁰_samples) - screen.z
    inv_δ = inv(step(screen.x⁰_samples))
    τi, τf = first(traj.itp.t), last(traj.itp.t)
    total = 0
    for y in screen.y_grid, x in screen.x_grid
        r = SVector{3}(x, y, screen.z)
        ks = max(1, _k_start_raw(_window_edge(gt, r, τi)[1], t_first, inv_δ))
        ke = min(Ns, _k_end_raw(_window_edge(gt, r, τf)[1], t_first, inv_δ))
        total += max(0, ke - ks + 1)
    end
    return total
end

grid = LinRange(-8.0, 8.0, 5)
z = 50.0
mkscreen(N_samples; x⁰_first = 52.0, δ = 0.1) =
    ObserverScreen(grid, grid, z, range(x⁰_first; step = δ, length = N_samples); c = 1.0)

@testset "extremal pixels bound the arrival offsets exactly" begin
    for (x0, y0) in ((0.0, 0.0), (3.3, -1.1), (12.0, 2.0), (-9.5, 9.5)), τ in (0.0, 7.3, 20.0)
        traj = _profile_trajectory(; x0, y0)
        gt = to_gpu(traj)
        lo, hi = brute_extremes(traj, grid, grid, z, τ)
        t_min, t_max = _window_edge_extremes(gt, grid, grid, z, τ)
        @test t_min == lo
        @test t_max == hi
    end
end

@testset "fully covered window" begin
    trajs = [_profile_trajectory(; x0 = 0.0), _profile_trajectory(; x0 = 2.0, y0 = -1.0)]
    screen = mkscreen(120)
    cov = window_coverage(trajs, screen)
    @test cov.ok
    @test cov.electrons_clipped == 0 && cov.slots_dropped == 0
    @test cov.slots_nominal == 2 * 120 * 25 == cov.slots_executed
    @test cov.slot_fill == 1.0
    @test cov.lead_margin_samples >= 0 && cov.tail_margin_samples >= 0
    @test all(p -> p.k_start_max == 1 && p.k_end_min == 120, cov.per_electron)
    @test cov.N_samples == 120
end

@testset "clipped window: exact recount, margins, missing without recount" begin
    trajs = [_profile_trajectory(; x0 = 0.0), _profile_trajectory(; x0 = 2.0, y0 = -1.0)]
    screen = mkscreen(300)   # runs past every pixel's retarded image of τf (≈ 74 + 0.1·220)
    cov = window_coverage(trajs, screen)
    @test !cov.ok
    @test cov.electrons_clipped == 2
    @test cov.tail_margin_samples < 0
    @test cov.slots_executed == sum(brute_slots(t, screen) for t in trajs)
    @test cov.slots_dropped == cov.slots_nominal - cov.slots_executed > 0
    @test 0 < cov.slot_fill < 1
    @test cov.worst_electron in (1, 2)
    @test all(p -> p.k_end_min < 300, cov.per_electron)
    nc = window_coverage(trajs, screen; recount = false)
    @test !nc.ok && nc.slots_executed === missing && nc.slot_fill === missing && nc.slots_dropped === missing
    @test nc.electrons_clipped == 2 && nc.tail_margin_samples == cov.tail_margin_samples
    # a window starting too early clips the head instead
    early = mkscreen(120; x⁰_first = 40.0)
    ce = window_coverage(trajs[1:1], early)
    @test !ce.ok && ce.lead_margin_samples < 0 && ce.slots_executed == brute_slots(trajs[1], early)
    @test ce.per_electron[1].k_start_max > 1
end

@testset "matches what the CPU-backend kernel executes" begin
    traj = _profile_trajectory(; x0 = 1.0)
    screen = mkscreen(300)
    cov = window_coverage([traj], screen)
    ke = cov.per_electron[1].k_end_min
    @test ke < 300
    fld = accumulate_field([traj], screen, GPUKernelNewton(), CPU(); n_iters = 1, mode = Val(:total))
    # the nearest pixel to the electron's final position is the one whose window ends first
    gt = to_gpu(traj)
    v = gt.itp(last(traj.itp.t))
    ix = argmin(abs.(grid .- v[2])); iy = argmin(abs.(grid .- v[3]))
    E = fld.E   # (N_samples, 3, Nx, Ny)
    @test all(iszero, E[(ke + 1):end, :, ix, iy])
    @test any(!iszero, E[ke, :, ix, iy])
    # the farthest corner keeps more samples
    kc = maximum(1:300) do k
        any(!iszero, E[k, :, 1, 1]) ? k : 0
    end
    @test kc >= ke
end

@testset "argument checks" begin
    traj = _profile_trajectory()
    @test_throws ArgumentError window_coverage([traj], ObserverScreen(grid, grid, z, range(52.0; step = -0.1, length = 10); c = 1.0))
    @test_throws ArgumentError window_coverage([traj], ObserverScreen(reverse(collect(grid)), collect(grid), z, range(52.0; step = 0.1, length = 10); c = 1.0))
    e = window_coverage(typeof(traj)[], mkscreen(10))
    @test e.ok && e.slots_nominal == 0 && e.slots_executed == 0
end
