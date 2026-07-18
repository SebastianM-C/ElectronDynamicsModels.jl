# ── LL-regime reachability: the newton-hardening ladder, steps 0–1 ──
# Ladder 0 (dynamics): the Landau–Lifshitz term drives a k·u drift of exactly
#   the size the exact LL plane-wave solution predicts, with the classical
#   electron conserving k·u to solver precision (Noether check) and the drift
#   scaling as a₀².  This is "we are in the classical RR regime", measured.
# Ladder 1 (kernel): the light-front Newton kernel consumes the RR-perturbed
#   worldline — k·u drifting is precisely the assumption §2.4 of the
#   newton-hardening report forbids baking in — and converges to a
#   tight-tolerance adaptive referee, with the LL-vs-classical difference in
#   the accumulated potential far above the kernel's numerical floor: the
#   pipeline resolves the correction, not merely survives the parameters.

using ElectronDynamicsModels
using ElectronDynamicsModels: TrajectoryInterpolant, ObserverScreen
using ModelingToolkit
using OrdinaryDiffEqVerner
using SciMLBase, StaticArrays, LinearAlgebra
using KernelAbstractions: CPU
using Test

const c_au = 137.03599908330932
const ω_ll = 0.057                                # 800 nm laser (a.u.)
const ncyc_ll = 10
const τspan_ll = (0.0, 2π * ncyc_ll / ω_ll)       # φ = ωτ for an electron born at rest
const δτₑω = 2π * 2.8179403e-15 / 800.0e-9        # τₑ·ω = 2π r_e/λ at 800 nm

function _ll_solve(sys, a₀)
    # amplitude is the E-field amplitude: a₀ = A/(cω) in atomic units
    op = [sys.x => zeros(4), sys.u => [c_au, 0.0, 0.0, 0.0],
        sys.laser.A => a₀ * c_au * ω_ll]
    prob = ODEProblem{false, SciMLBase.FullSpecialize}(sys, op, τspan_ll;
        u0_constructor = SVector{8}, fully_determined = true)
    # SVector states are required downstream: to_gpu's spline conversion
    # assumes isbits knots.
    return solve(prob, Vern9(); abstol = 1.0e-12, reltol = 1.0e-12,
        saveat = range(τspan_ll..., length = 400 * ncyc_ll), dtmax = (2π / ω_ll) / 50)
end

# light-front momentum h = (u⁰ − u³)/c along the worldline (k̂ = +ẑ)
_lightfront(sys, sol) = [(sol[sys.u, i][1] - sol[sys.u, i][4]) / c_au for i in eachindex(sol.t)]

_rel_l2(a, b) = norm(a .- b) / norm(b)

function _ll_build(ll::Bool)
    @named world = Worldline(:τ, :atomic)
    @named laser = PlaneWave(; amplitude = 10.0 * c_au * ω_ll, frequency = ω_ll, world)
    electron = if ll
        @named electron = LandauLifshitzElectron(; laser)
    else
        @named electron = ClassicalElectron(; laser)
    end
    return ll ? mtkcompile(electron, allow_symbolic = true) : mtkcompile(electron)
end

@testset "LL regime is reachable (plane wave, a₀ = 10, born at rest)" begin
    cl_sys = _ll_build(false)
    ll_sys = _ll_build(true)

    sol_cl = _ll_solve(cl_sys, 10.0)
    sol_ll = _ll_solve(ll_sys, 10.0)
    sol_ll20 = _ll_solve(ll_sys, 20.0)
    @test SciMLBase.successful_retcode(sol_cl)
    @test SciMLBase.successful_retcode(sol_ll)
    @test SciMLBase.successful_retcode(sol_ll20)

    @testset "ladder 0 — dynamics at the predicted RR size" begin
        h_cl = _lightfront(cl_sys, sol_cl)
        @test abs(1 - h_cl[end] / h_cl[1]) < 1.0e-10   # Noether: k·u exact classically

        drifts = Dict{Float64, Float64}()
        for (sol, a₀) in ((sol_ll, 10.0), (sol_ll20, 20.0))
            h = _lightfront(ll_sys, sol)
            drifts[a₀] = 1 - h[end] / h[1]
            # Exact LL plane-wave solution, CW carrier:
            #   1/h(φ) = 1/h₀ + (2/3)·τₑω·∫a′² ⇒ drift ≈ (2π/3)·τₑω·a₀²·n_cyc
            pred = (2π / 3) * δτₑω * a₀^2 * ncyc_ll
            @test isapprox(drifts[a₀], pred; rtol = 0.05)   # measured: 0.1% agreement
            @test all(<(1.0e-12), diff(h))                  # monotone decrease
        end
        @test isapprox(drifts[20.0] / drifts[10.0], 4.0; rtol = 0.02)  # ∝ a₀²
    end

    @testset "ladder 1 — the kernel on the RR-perturbed worldline" begin
        traj_cl = TrajectoryInterpolant(sol_cl, cl_sys.x, cl_sys.u)
        traj_ll = TrajectoryInterpolant(sol_ll, ll_sys.x, ll_sys.u)

        a₀ = 10.0
        θ = 2 / a₀                                     # worst (beaming-angle) pixel
        zexc = traj_ll.itp(τspan_ll[2])[traj_ll.x_idxs[4]]
        D = 100 * zexc
        z_screen = D * cos(θ)
        x_grid = [D * sin(θ)]
        y_grid = [0.0]
        arr(traj, τ) = begin
            v = traj.itp(τ); xi = traj.x_idxs
            v[xi[1]] + hypot(x_grid[1] - v[xi[2]], y_grid[1] - v[xi[3]], z_screen - v[xi[4]])
        end
        x⁰ = LinRange(max(arr(traj_cl, 0.0), arr(traj_ll, 0.0)),
            min(arr(traj_cl, τspan_ll[2]), arr(traj_ll, τspan_ll[2])), 256 * ncyc_ll)
        screen = ObserverScreen(x_grid, y_grid, z_screen, x⁰; c = c_au)

        # Tight-tolerance referee.  The DEFAULT adaptive reference is unusable
        # in this regime: reltol ~1e-3 on the retarded-time ODE gives O(1)
        # errors at the Doppler-spike slots (measured 94% L2 against it) —
        # the referee must be tighter than what it referees.
        A_ref = accumulate_potential([traj_ll], screen, Vern9();
            reltol = 1.0e-13, abstol = 1.0e-13)
        An(traj, n) = accumulate_potential([traj], screen, GPUKernelNewton(), CPU(); n_iters = n)
        A3, A6, A12 = An(traj_ll, 3), An(traj_ll, 6), An(traj_ll, 12)
        e3, e6, e12 = _rel_l2(A3, A_ref), _rel_l2(A6, A_ref), _rel_l2(A12, A_ref)

        @test all(isfinite, A12)
        @test e12 < 1.0e-3            # measured 1.0e-4: converged to the tight referee
        @test e12 < e6 < e3           # bracket: monotone through the bisection tail
        # The physics is resolved: LL vs classical differs in the accumulated
        # potential (measured ~6.7e-2, SNR ~650 over the kernel floor).
        @test _rel_l2(An(traj_cl, 12), A12) > 10 * e12
    end
end
