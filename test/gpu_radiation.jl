using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen
using DataInterpolations
using StaticArrays
using KernelAbstractions: CPU
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqVerner
using LinearAlgebra
using Test

# ── Analytic worldline (no ODE solve) ───────────────────────────────────────
# Proper-time-parameterized charge: constant-rate time + z drift `vz` toward the
# screen, plus a transverse sinusoidal wiggle along x.  Future-directed timelike
# (u⁰ > |u⃗|) as long as `g > sqrt(vz² + (A·Ω)²)`, so the retarded-time RHS
# 1/(u⁰ - u⃗·n̂) stays positive.  Two regimes are used below:
#   - `vz = 0`  (transverse): integrand ≈ const, clean L2 agreement — a
#     correctness check for the kernel (geometry, slot mapping, spline eval).
#   - `vz ≈ c`  (forward Doppler): u⃗·n̂ ≈ u⁰, so the integrand swings hard and
#     fixed-step RK4 is genuinely sensitive to `n_substeps` — a convergence check.
function analytic_traj(; g, A, Ω, vz, τspan, N, K = 1.0)
    @assert g > sqrt(vz^2 + (A * Ω)^2) "worldline must stay timelike"
    ts = collect(range(τspan[1], τspan[2], length = N))
    us = [SVector{8}(
        g * τ,                # x⁰
        A * sin(Ω * τ),       # x¹
        0.0,                  # x²
        vz * τ,               # x³
        g,                    # u⁰
        A * Ω * cos(Ω * τ),   # u¹
        0.0,                  # u²
        vz,                   # u³
    ) for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    # 4-acceleration 𝔞μ = duμ/dτ = (0, −A Ω² sin(Ωτ), 0, 0), known in closed form;
    # the field accumulator interpolates it from a dedicated spline.
    as = [SVector{4}(0.0, -A * Ω^2 * sin(Ω * τ), 0.0, 0.0) for τ in ts]
    a_itp = CubicSpline(as, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4),
        SVector{4, Int}(5, 6, 7, 8), K)
end

# a₀ = 10 born-at-rest plane-wave worldline (exact closed form, u·u = 1):
#   φ = τ,  w = (a₀²/2)cos²φ,  u = (1+w, a₀cosφ, 0, w),
#   x = (τ + z, a₀sinφ, 0, z),  z = (a₀²/4)(φ + sinφ·cosφ)
# In proper time this has only 1st/2nd harmonics at any a₀ — all the a₀³
# spectral compression lives in the observer-time pull-back, which is exactly
# what stresses the per-slot root solve at the beaming-angle pixel.
function planewave_traj(; a₀, ncyc, N, K = 1.0)
    ts = collect(range(0.0, 2π * ncyc, length = N))
    w(φ) = (a₀^2 / 2) * cos(φ)^2
    zz(φ) = (a₀^2 / 4) * (φ + sin(φ) * cos(φ))
    us = [SVector{8}(τ + zz(τ), a₀ * sin(τ), 0.0, zz(τ),
                     1 + w(τ), a₀ * cos(τ), 0.0, w(τ)) for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    as = [SVector{4}(-a₀^2 * sin(τ) * cos(τ), -a₀ * sin(τ), 0.0,
                     -a₀^2 * sin(τ) * cos(τ)) for τ in ts]
    a_itp = CubicSpline(as, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4),
        SVector{4, Int}(5, 6, 7, 8), K)
end

# Relative Frobenius error — robust to the single-slot boundary ambiguity at the
# edge of each pixel's arrival window (where even Tsit5 vs Vern9 disagree by a
# few %).  A max-abs metric would report that edge slot instead of the bulk fit.
rel_l2(a, b) = norm(a .- b) / norm(b)

@testset "GPU radiation accumulation" begin

    @testset "GPUKernelRK4 matches reference (bulk pattern)" begin
        # Mild transverse regime: integrand nearly constant, so the kernel's
        # geometry / slot-mapping / spline evaluation is what's under test.
        traj = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
            τspan = (0.0, 20.0), N = 6000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)

        z = 50.0
        Nx = Ny = 9
        half = 8.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 240)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        A_gpu = accumulate_potential(trajs, screen, GPUKernelRK4(), CPU(); n_substeps = 8)

        @test size(A_gpu) == size(A_ref)
        @test all(isfinite, A_gpu)
        @test maximum(abs, A_ref) > 0                 # the reference actually radiates
        @test rel_l2(A_gpu, A_ref) < 5.0e-3           # bulk pattern matches the reference

        # The two-phase CPU-solve + AcceleratedKernels accumulation path
        # (src/gpu/accumulate.jl) should track the same reference.
        A_ak = accumulate_potential(trajs, screen, Tsit5(), CPU())
        @test rel_l2(A_ak, A_ref) < 5.0e-3
    end

    @testset "GPUKernelNewton matches reference (bulk pattern)" begin
        # Same transverse setup as the RK4 bulk-pattern test, through the
        # Newton light-cone kernel. In this gentle regime a single warm-started
        # Newton correction per slot already sits at the reference floor.
        traj = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
            τspan = (0.0, 20.0), N = 6000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)

        z = 50.0
        Nx = Ny = 9
        half = 8.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 240)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        A_newton = accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = 1)

        @test size(A_newton) == size(A_ref)
        @test all(isfinite, A_newton)
        @test rel_l2(A_newton, A_ref) < 5.0e-3
    end

    @testset "n_iters drives Newton convergence (forward Doppler)" begin
        # Stress regime for the light-cone solve: dτ_r per slot is large and f
        # is curved off-axis, so the Euler predictor lands far and iterations
        # are genuinely needed. n_iters = 3 (4 spline evals/slot) matches the
        # RK4 n_substeps = 8 floor (33 evals/slot).
        traj = analytic_traj(; g = 1.05, A = 0.15, Ω = 2.0, vz = 0.95,
            τspan = (0.0, 20.0), N = 8000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)
        g, vz = 1.05, 0.95

        z = 60.0
        Nx = Ny = 7
        half = 6.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(g * τi + (z - vz * τf - half), g * τf + (z + half), 200)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        err(n) = rel_l2(accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = n), A_ref)

        e1, e3 = err(1), err(3)
        @test e1 > 1.0e-2          # a single correction is not enough here
        @test e3 < 5.0e-3          # three corrections reach reference agreement
        @test e3 < e1              # adding corrections reduces the discrepancy
    end

    @testset "light-front residual holds the floor at production-scale Z" begin
        # Regression pin for the screen-relative (light-front) spelling of the
        # light-cone residual.  At Z ~ 2e9 the absolute spelling
        # x⁰_k − x⁰(τ) − R carries an ε·Z ≈ 2.4e-7 storage-quantization floor;
        # the regrouped f = tₖ − ψ(τ) − ρ²/(R + d³) keeps rounding at the
        # interaction scale (~1e-15 here).  Every other testset runs at small Z
        # where the two spellings agree — this one fails (by ~2 orders) if the
        # kernel ever regresses to absolute coordinates.
        g, A, Ω, vz = 1.05, 0.15, 2.0, 0.95
        traj = analytic_traj(; g, A, Ω, vz, τspan = (0.0, 20.0), N = 8000)
        gt = ElectronDynamicsModels.to_gpu(traj)
        LCE = ElectronDynamicsModels._lightcone_eval   # promoted out of Experimental on main

        Z = 2.0e9
        τstar = 10.3                                  # generic mid-span point
        for xp in (0.0, 1.0e5)                        # on-axis, and edge (ρ² term live)
            r_obs = SVector{3}(xp, 0.0, Z)
            # True arrival time of the emission at τstar from the closed-form
            # worldline, in BigFloat; τstar is then the exact retarded time for
            # this target, so the converged residual must vanish to rounding.
            τb = big(τstar)
            x1b = big(A) * sin(big(Ω) * τb)
            arrival = big(g) * τb + sqrt((big(xp) - x1b)^2 + (big(Z) - big(vz) * τb)^2)
            tₖ = Float64(arrival - big(Z))            # screen-relative target (small)
            _, f, rhs = LCE(τstar, gt, r_obs, tₖ)
            @test abs(f) < 1.0e-9                     # old spelling: ~2.4e-7 — fails
            @test rhs > 0                             # future-directed Doppler factor
        end

        # End-to-end at the same Z: offset grid + window block wiring
        # (t_first = x⁰_first − z_screen exact by Sterbenz; tₖ built small).
        trajs = [traj]
        Nx = Ny = 5
        half = 1.0e5
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(Z, Z + 8.0, 240)
        screen = ObserverScreen(x_grid, y_grid, Z, x⁰; c = 1.0)
        A_ref = accumulate_potential(trajs, screen, Vern9())
        A1 = accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = 1)
        A3 = accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = 3)
        @test maximum(abs, A_ref) > 0
        @test rel_l2(A1, A_ref) < 5.0e-3              # matches the adaptive reference
        @test rel_l2(A3, A1) < 1.0e-10                # converged: extra iters are no-ops
    end

    @testset "bracketed step survives aliased sampling (a₀ = 10, worst pixel)" begin
        # The undamped Newton step fails on screens under-sampled for their
        # harmonic content: at the beaming-angle pixel (θ = 2/a₀, Doppler
        # factor swinging ~400× per cycle) with ~64 samples/period, the tangent
        # throws iterates across arrival-curve wiggles, and MORE undamped
        # iterations make it WORSE (report: 98/384 → 265/384 failed slots from
        # n_iters 3 → 6; clamping to the τ-span is not a convergence
        # mechanism).  The bracketed step turns every iteration into
        # guaranteed enclosure shrinkage: error decreases monotonically in
        # n_iters and reaches a converged fixed point.  The residual plateau
        # vs the adaptive reference (~8% here) is spike-slot value sensitivity
        # on an aliased grid, not solver error — hence the monotonicity and
        # self-convergence assertions rather than a tight absolute tolerance.
        a₀, ncyc = 10.0, 6
        traj = planewave_traj(; a₀, ncyc, N = 8000)
        trajs = [traj]
        τf = last(traj.itp.t)

        θ = 2 / a₀
        D = 100 * (a₀^2 / 4) * τf
        z_screen = D * cos(θ)
        x_grid = [D * sin(θ)]
        y_grid = [0.0]
        𝒜(τ) = (v = traj.itp(τ);
            v[1] + hypot(x_grid[1] - v[2], y_grid[1] - v[3], z_screen - v[4]))
        x⁰ = LinRange(𝒜(0.0), 𝒜(τf), 64 * ncyc)     # ~64 samples/period: aliased
        screen = ObserverScreen(x_grid, y_grid, z_screen, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        An(n) = accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = n)
        A1, A3, A8, A12 = An(1), An(3), An(8), An(12)
        err(A) = rel_l2(A, A_ref)

        @test all(isfinite, A12)
        @test err(A3) < err(A1) / 5        # rapid progress once corrections act
        @test err(A12) ≤ err(A3)           # more iterations never hurt (bracket!)
        # The hardest slots converge through the bisection tail (enclosure halves
        # per iteration), so A8 → A12 still moves at the ~1e-4 level; the march is
        # deterministic, so a measured-with-margin bound is stable.
        @test rel_l2(A12, A8) < 1.0e-3
        @test_throws ArgumentError An(0)   # guard: Euler-march degradation is an error
    end

    @testset "n_substeps drives RK4 convergence (forward Doppler)" begin
        # Relativistic drift toward the screen ⇒ u⃗·n̂ ≈ u⁰ ⇒ the retarded-time
        # integrand swings hard, so n_substeps = 1 is visibly under-resolved.
        traj = analytic_traj(; g = 1.05, A = 0.15, Ω = 2.0, vz = 0.95,
            τspan = (0.0, 20.0), N = 8000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)
        g, vz = 1.05, 0.95

        z = 60.0
        Nx = Ny = 7
        half = 6.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        # Forward emission compresses the arrival window (dx⁰/dτ ≈ u⁰ - u³).
        x⁰ = LinRange(g * τi + (z - vz * τf - half), g * τf + (z + half), 200)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        A_ref = accumulate_potential(trajs, screen, Vern9())
        err(n) = rel_l2(accumulate_potential(trajs, screen, GPUKernelRK4(), CPU(); n_substeps = n), A_ref)

        e1, e8 = err(1), err(8)
        @test e1 > 5.0e-2          # single-step RK4 is genuinely under-resolved here
        @test e8 < 5.0e-3          # enough sub-steps recovers reference agreement
        @test e8 < e1              # adding sub-steps reduces the discrepancy
    end

    @testset "GPUKernelRK4 field matches reference (split E/B)" begin
        # Same transverse worldline as the potential bulk-pattern test, run through
        # the field path. Exercises the kernel leaf end-to-end on the CPU backend —
        # the split tensor, extract_EB, and the four-bucket scalar writes — against
        # the adaptive-Vern9 CPU `accumulate_field`. Caught GPU-incompatible code
        # (e.g. colon-slice device writes) would diverge here; a logic bug in the
        # split would show up in the per-bucket errors.
        traj = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
            τspan = (0.0, 20.0), N = 6000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)

        z = 50.0
        Nx = Ny = 9
        half = 8.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 240)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        ref = accumulate_field(trajs, screen, Vern9())
        gpu = accumulate_field(trajs, screen, GPUKernelRK4(), CPU(); n_substeps = 8)

        @test keys(gpu) == (:E, :B, :E_far, :B_far)
        @test size(gpu.E) == size(ref.E)
        @test all(isfinite, gpu.E) && all(isfinite, gpu.B)
        @test maximum(abs, ref.E_far) > 0           # the reference actually radiates
        @test rel_l2(gpu.E, ref.E) < 5.0e-3         # total field matches
        @test rel_l2(gpu.B, ref.B) < 5.0e-3
        @test rel_l2(gpu.E_far, ref.E_far) < 5.0e-3 # radiation bucket matches
        @test rel_l2(gpu.B_far, ref.B_far) < 5.0e-3
    end

    @testset "accumulate_field_sharded ≡ accumulate_field (streamed reduce, both modes)" begin
        # Electron sharding is exact by linearity; the streamed host reduce (one accumulator,
        # partials folded under a lock as each device task finishes) must reproduce the
        # single-call result to roundoff. devices = [1, 1] shards 3 electrons over two tasks
        # on the one CPU "device" — uneven shards (2 + 1) exercise _shard_indices too.
        trajs = [
            analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
            analytic_traj(; g = 1.3, A = 0.20, Ω = 2.5, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
            analytic_traj(; g = 1.1, A = 0.30, Ω = 1.5, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
        ]
        τi, τf = first(trajs[1].itp.t), last(trajs[1].itp.t)
        z = 50.0
        Nx = Ny = 7
        half = 6.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 120)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        for (alg, kw) in ((GPUKernelRK4(), (; n_substeps = 2)), (GPUKernelNewton(), (; n_iters = 2))),
                mode in (Val(:split), Val(:total))
            one = accumulate_field(trajs, screen, alg, CPU(); mode, kw...)
            # both reduces: on the "device" (the one CPU device folds the second shard into the
            # first shard's buffers, then one threaded download) and streamed through the host
            for reduce in (:device, :host)
                stats = ReduceStats()
                shd = accumulate_field_sharded(trajs, screen, alg, CPU(); devices = [1, 1], mode, reduce,
                    reduce_workers = 2, reduce_stats = stats, kw...)
                @test stats.n_folds == 1 && stats.fold_s > 0 && stats.download_s > 0
                @test propertynames(shd) == propertynames(one)
                for k in propertynames(one)
                    @test size(getproperty(shd, k)) == size(getproperty(one, k))
                    @test rel_l2(getproperty(shd, k), getproperty(one, k)) < 1.0e-12
                end
            end
            @test_throws ArgumentError accumulate_field_sharded(trajs, screen, alg, CPU(); devices = [1, 1], mode, reduce = :nope, kw...)
            # a single shard must be the plain path bit-for-bit
            solo = accumulate_field_sharded(trajs, screen, alg, CPU(); devices = [1], mode, kw...)
            @test all(k -> getproperty(solo, k) == getproperty(one, k), propertynames(one))
        end
        # the device-side fold and the threaded permuted download, on plain arrays
        let buf = rand(5, 4, 3, 40), part = rand(5, 4, 3, 40)
            ref = permutedims(buf .+ part, (4, 3, 1, 2))
            acc = copy(buf)
            @test ElectronDynamicsModels._device_add!(CPU(), 1, acc, part) === acc
            @test acc == buf .+ part
            @test ElectronDynamicsModels._download_permuted(acc) == ref
            @test ElectronDynamicsModels._download_permuted(acc; backend = CPU(), dev = 1, workers = 3) == ref
            @test_throws DimensionMismatch ElectronDynamicsModels._device_add!(CPU(), 1, acc, part[:, :, :, 1:2])
        end
        @test ElectronDynamicsModels._shard_indices(5, 2) == [1:3, 4:5]
        @test ElectronDynamicsModels._shard_indices(2, 3) == [1:1, 2:2]   # empty shards dropped
    end

    @testset "electron batching: persistent device buffers" begin
        # `buffers` + `finish` keep the accumulation buffers alive across `accumulate_field`
        # calls, so a driver can solve a batch of electrons, accumulate it and drop its splines
        # (the host cost of a production run: ~7.4 MB of spline per electron for the whole field
        # phase). The kernels, the per-electron uploads and the launch order are untouched, so on
        # ONE device the batched sum adds the same contributions in the same order and comes out
        # bit-identical — the assertions below only require a relative L2 of 1e-12, because that
        # is what the API promises: any reordering of a floating-point sum (a different batch
        # split on the sharded path, a device reduce) is free to move the last bits.
        trajs = [analytic_traj(; g = 1.2 + 0.01i, A = 0.25, Ω = 2.0, vz = 0.0,
            τspan = (0.0, 20.0), N = 800) for i in 1:20]
        τi, τf = first(trajs[1].itp.t), last(trajs[1].itp.t)
        z = 50.0
        Nx, Ny = 7, 5
        half = 6.0
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 60)
        screen = ObserverScreen(LinRange(-half, half, Nx), LinRange(-half, half, Ny), z, x⁰; c = 1.0)

        # Feed `trajs` in batches of `B`, holding the buffers across the calls; the last call
        # finishes and downloads. Returns the cube exactly as the single call would.
        function batched(alg, mode, B; kw...)
            acc = nothing
            res = nothing
            for i0 in 1:B:length(trajs)
                rng = i0:min(i0 + B - 1, length(trajs))
                last_batch = rng[end] == length(trajs)
                res = accumulate_field(trajs[rng], screen, alg, CPU();
                    mode, buffers = acc, finish = last_batch, kw...)
                last_batch || (acc = res)
            end
            return res
        end

        for (alg, kw) in ((GPUKernelRK4(), (; n_substeps = 2)), (GPUKernelNewton(), (; n_iters = 2))),
                mode in (Val(:split), Val(:total))
            one = accumulate_field(trajs, screen, alg, CPU(); mode, kw...)
            for B in (20, 25, 3, 7)   # 20 = one batch of all, 25 > N, then genuine splits
                b = batched(alg, mode, B; kw...)
                @test propertynames(b) == propertynames(one)
                for k in propertynames(one)
                    @test rel_l2(getproperty(b, k), getproperty(one, k)) < 1.0e-12
                end
                # one device, one stream, same launch order ⇒ nothing moved at all
                @test all(k -> getproperty(b, k) == getproperty(one, k), propertynames(one))
            end

            # The accumulator counts what it swallowed and can be finished explicitly, without
            # a last batch of electrons; finishing does not consume it.
            acc = accumulate_field(trajs[1:12], screen, alg, CPU(); mode, finish = false, kw...)
            @test acc isa FieldAccumulator
            @test acc.n_electrons == 12
            acc = accumulate_field(trajs[13:20], screen, alg, CPU(); mode, buffers = acc, finish = false, kw...)
            @test acc.n_electrons == 20
            fin = finish_field(acc)
            @test all(k -> getproperty(fin, k) == getproperty(one, k), propertynames(one))
            @test all(k -> getproperty(finish_field(acc; workers = 2), k) == getproperty(one, k), propertynames(one))
            # `sink` sees the live device buffers, as it does on the single-call path
            seen = Ref(0)
            @test finish_field(acc; sink = (E1, B1, E2, B2, m) -> (seen[] += 1; m)) === mode
            @test seen[] == 1

            # Sharded: each device keeps its shard's buffers across the batches and the reduce
            # runs once, on the finishing call. Splitting into batches re-shards the electrons,
            # so this is a roundoff-level check, not a bit-identity one.
            for reduce in (:device, :host)
                shard_one = accumulate_field_sharded(trajs, screen, alg, CPU();
                    devices = [1, 1], mode, reduce, kw...)
                sacc = nothing
                res = nothing
                for i0 in 1:7:length(trajs)
                    rng = i0:min(i0 + 6, length(trajs))
                    last_batch = rng[end] == length(trajs)
                    res = accumulate_field_sharded(trajs[rng], screen, alg, CPU();
                        devices = [1, 1], mode, reduce, buffers = sacc, finish = last_batch, kw...)
                    last_batch || (sacc = res)
                end
                @test sacc isa ShardedFieldAccumulator
                @test all(a -> a isa FieldAccumulator, sacc.accs)
                @test sum(a -> a.n_electrons, sacc.accs) == length(trajs)   # every batch, the finishing one included
                @test propertynames(res) == propertynames(one)
                for k in propertynames(one)
                    @test rel_l2(getproperty(res, k), getproperty(shard_one, k)) < 1.0e-12
                    @test rel_l2(getproperty(res, k), getproperty(one, k)) < 1.0e-12
                end
            end
        end

        # Mismatched buffers are rejected rather than silently summed into the wrong cube.
        let alg = GPUKernelNewton(), kw = (; n_iters = 2)
            acc = accumulate_field(trajs[1:2], screen, alg, CPU(); mode = Val(:split), finish = false, kw...)
            @test_throws ArgumentError accumulate_field(trajs[3:4], screen, alg, CPU();
                mode = Val(:total), buffers = acc, kw...)
            small = ObserverScreen(LinRange(-half, half, Nx - 1), LinRange(-half, half, Ny), z, x⁰; c = 1.0)
            @test_throws DimensionMismatch accumulate_field(trajs[3:4], small, alg, CPU();
                mode = Val(:split), buffers = acc, kw...)
            @test_throws ArgumentError accumulate_field(trajs[3:4], screen, alg, CPU(); buffers = (;), kw...)
            @test_throws ArgumentError FieldAccumulator(screen, CPU(); mode = Val(:nope))
            sacc = accumulate_field_sharded(trajs[1:4], screen, alg, CPU();
                devices = [1, 1], finish = false, kw...)
            @test_throws ArgumentError accumulate_field_sharded(trajs[5:8], screen, alg, CPU();
                devices = [1], buffers = sacc, kw...)
        end
        # padded shards: always one range per device (empty where the batch runs out), and the
        # same split as `_shard_indices` whenever there are at least as many electrons as devices
        @test ElectronDynamicsModels._shard_indices_padded(5, 2) == [1:3, 4:5]
        @test ElectronDynamicsModels._shard_indices_padded(2, 3) == [1:1, 2:2, 3:2]
        @test ElectronDynamicsModels._shard_indices_padded(0, 2) == [1:0, 1:0]
    end

    @testset "sample_chunks: chunk grid reproduces the per-pixel walk" begin
        # One thread per (pixel, chunk) walking a slice of the pixel's executed slots. Chunk 1 is
        # today's path (bit-identical); later chunks start from a cold light-cone solve, so the
        # Newton kernel agrees to roundoff and the RK4 march, whose per-step drift the restart
        # removes, to its own convergence floor.
        trajs = [
            analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
            analytic_traj(; g = 1.3, A = 0.20, Ω = 2.5, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
        ]
        τi, τf = first(trajs[1].itp.t), last(trajs[1].itp.t)
        z = 50.0
        Nx, Ny = 7, 5
        half = 6.0
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 240)
        screen = ObserverScreen(LinRange(-half, half, Nx), LinRange(-half, half, Ny), z, x⁰; c = 1.0)
        for (alg, kw, tol) in ((GPUKernelNewton(), (; n_iters = 2), 1.0e-12), (GPUKernelRK4(), (; n_substeps = 2), 1.0e-6)),
                mode in (Val(:split), Val(:total))
            one = accumulate_field(trajs, screen, alg, CPU(); mode, kw...)
            c1 = accumulate_field(trajs, screen, alg, CPU(); mode, sample_chunks = 1, kw...)
            @test all(k -> getproperty(c1, k) == getproperty(one, k), propertynames(one))
            for C in (3, 8, 500)   # 500 > executed slots per pixel: most chunks empty
                cC = accumulate_field(trajs, screen, alg, CPU(); mode, sample_chunks = C, kw...)
                for k in propertynames(one)
                    @test rel_l2(getproperty(cC, k), getproperty(one, k)) < tol
                end
            end
        end
        pot1 = accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = 2)
        pot4 = accumulate_potential(trajs, screen, GPUKernelNewton(), CPU(); n_iters = 2, sample_chunks = 4)
        @test rel_l2(pot4, pot1) < 1.0e-12
        @test_throws ArgumentError accumulate_field(trajs, screen, GPUKernelNewton(), CPU(); sample_chunks = 0)
        # the slice arithmetic: exact cover, disjoint, near-even, empty beyond the range
        let cs = ElectronDynamicsModels._chunk_slots
            @test cs(10, 29, 1, 1) == (10, 29)
            slices = [cs(10, 29, c, 4) for c in 1:4]
            @test slices == [(10, 14), (15, 19), (20, 24), (25, 29)]
            slices = [cs(1, 10, c, 3) for c in 1:3]
            @test slices == [(1, 4), (5, 7), (8, 10)]
            @test cs(5, 6, 3, 4)[1] > cs(5, 6, 3, 4)[2]
            @test ElectronDynamicsModels._chunk_pixel(1, 7, 5, 3) == (1, 1, 1)
            @test ElectronDynamicsModels._chunk_pixel(7 * 5 + 2, 7, 5, 3) == (2, 1, 2)
        end
    end

    @testset "LaunchTimer: one event pair per launch, results untouched" begin
        # Device-event kernel timing (GPUDiagnostics.jl). On the CPU backend the events are host
        # clocks (kernels are synchronous), so the plumbing — one pair per electron, keyed by
        # device, shared safely across the sharded driver's tasks — is testable without a GPU.
        # (@testset scopes are local: rebuild the 3-electron setup of the sharded test above.)
        trajs = [
            analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
            analytic_traj(; g = 1.3, A = 0.20, Ω = 2.5, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
            analytic_traj(; g = 1.1, A = 0.30, Ω = 1.5, vz = 0.0, τspan = (0.0, 20.0), N = 3000),
        ]
        τi, τf = first(trajs[1].itp.t), last(trajs[1].itp.t)
        z = 50.0
        Nx = Ny = 7
        half = 6.0
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 120)
        screen = ObserverScreen(LinRange(-half, half, Nx), LinRange(-half, half, Ny), z, x⁰; c = 1.0)
        e0 = gpu_event(CPU()); sleep(0.01); e1 = gpu_event(CPU())
        @test 0.005 < gpu_elapsed(e0, e1) < 5.0
        @test isempty(launch_times(LaunchTimer()))
        for (alg, kw) in ((GPUKernelRK4(), (; n_substeps = 2)), (GPUKernelNewton(), (; n_iters = 2)))
            ref = accumulate_field(trajs, screen, alg, CPU(); kw...)
            t = LaunchTimer()
            got = accumulate_field(trajs, screen, alg, CPU(); timer = t, kw...)
            @test all(k -> getproperty(got, k) == getproperty(ref, k), propertynames(ref))
            lt = launch_times(t)
            @test collect(keys(lt)) == [1]
            @test length(lt[1]) == length(trajs) && all(>(0), lt[1])
            # sharded: both tasks run on device 1 → one lane with every launch (locked pushes)
            t = LaunchTimer()
            accumulate_field_sharded(trajs, screen, alg, CPU(); devices = [1, 1], timer = t, kw...)
            @test length(launch_times(t)[1]) == length(trajs)
            t = LaunchTimer()
            accumulate_potential(trajs, screen, alg, CPU(); timer = t, kw...)
            @test length(launch_times(t)[1]) == length(trajs)
        end
    end

    @testset "GPUKernelNewton field matches reference (split E/B)" begin
        # Field path through the Newton light-cone kernel, same setup as the
        # RK4 field test above.
        traj = analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0,
            τspan = (0.0, 20.0), N = 6000)
        trajs = [traj]
        τi, τf = first(traj.itp.t), last(traj.itp.t)

        z = 50.0
        Nx = Ny = 9
        half = 8.0
        x_grid = LinRange(-half, half, Nx)
        y_grid = LinRange(-half, half, Ny)
        x⁰ = LinRange(1.2τi + (z - 2half), 1.2τf + (z + 2half), 240)
        screen = ObserverScreen(x_grid, y_grid, z, x⁰; c = 1.0)

        ref = accumulate_field(trajs, screen, Vern9())
        gpu = accumulate_field(trajs, screen, GPUKernelNewton(), CPU(); n_iters = 2)

        @test keys(gpu) == (:E, :B, :E_far, :B_far)
        @test all(isfinite, gpu.E) && all(isfinite, gpu.B)
        @test rel_l2(gpu.E, ref.E) < 5.0e-3
        @test rel_l2(gpu.B, ref.B) < 5.0e-3
        @test rel_l2(gpu.E_far, ref.E_far) < 5.0e-3
        @test rel_l2(gpu.B_far, ref.B_far) < 5.0e-3
    end
end
