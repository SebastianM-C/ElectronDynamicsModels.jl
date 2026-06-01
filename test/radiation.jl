using ElectronDynamicsModels
using ElectronDynamicsModels: m_dot   # Minkowski inner product (+,-,-,-); reuse the package's, don't redefine
using ModelingToolkit
using OrdinaryDiffEqVerner
using Test
using SciMLBase

# Build a LandauLifshitzElectron in `laser`, solve a short worldline, return (sol, sys).
function solve_ll(laser; x0, v0, tspan)
    @named electron = LandauLifshitzElectron(; laser)
    sys = mtkcompile(electron, allow_symbolic = true)
    γ₀ = 1 / sqrt(1 - v0^2)
    u0 = [sys.x => x0, sys.u => [γ₀, γ₀ * v0, 0.0, 0.0]]
    sol = solve(ODEProblem(sys, u0, tspan), Vern9(), abstol = 1.0e-12, reltol = 1.0e-12)
    return sol, sys
end

# Largest |F_rad| component on the worldline — confirms a test is not vacuous
# (the field must actually drive a radiation-reaction force).
max_F_rad(sol, sys) = maximum(maximum(abs, sol[sys.radiation.F_rad, i]) for i in eachindex(sol.t))

"""
    max_orthogonality_violation(sol, sys) -> Real

Quantify how badly the Landau–Lifshitz radiation-reaction 4-force breaks the
mass-shell identity  uᵤ F^μ_rad = 0  across the solved worldline, returning a
single dimensionless number to compare against a tolerance.

`sol[sys.u, i]` and `sol[sys.radiation.F_rad, i]` give the 4-velocity and the RR
4-force at sample `i`; `m_dot` (imported from the package) is the Minkowski inner product.
"""
function max_orthogonality_violation(sol, sys)
    return maximum(abs(m_dot(sol[sys.u, i], sol[sys.radiation.F_rad, i])) for i in eachindex(sol.t))
end

@testset "Landau–Lifshitz radiation reaction" begin

    @testset "uᵤ F^μ_rad = 0 across external fields" begin
        # Orthogonality of the RR force to u holds by construction for ANY (u, F),
        # so a pass really validates the symbolic assembly + index contractions
        # survive mtkcompile for fields of increasing complexity.
        cases = [
            (
                name = "UniformField (E×B, ∂F = 0)",
                make = world -> UniformField(; E_field = [0.05, 0, 0], B_field = [0, 0, 0.1], world, name = :laser),
                x0 = [0.0, 0, 0.2, 0], v0 = 0.3, tspan = (0.0, 2.0),
            ),
            (
                name = "PlaneWave",
                make = world -> PlaneWave(; amplitude = 0.05, frequency = 1.0, world, name = :laser),
                x0 = [0.0, 0, 0.2, 0], v0 = 0.3, tspan = (0.0, 2.0),
            ),
            (
                name = "GaussLaser",
                make = world -> GaussLaser(; wavelength = 1.0, a0 = 0.5, world, name = :laser),
                x0 = [0.0, 0, 0.2, 0], v0 = 0.3, tspan = (0.0, 8.0),
            ),
            # Laguerre–Gauss with OAM: the most complex Fμν expression in the suite.
            # This genuinely exercises the symbolic field differentiation — ∂_ν F of
            # the hypergeometric LG field feeds the gradient term — and the full
            # three-term RR force must still be orthogonal to u.
            (
                name = "LaguerreGauss (m = 1, CW)",
                make = world -> LaguerreGaussLaser(;
                    wavelength = 1.0, a0 = 0.5, azimuthal_index = 1,
                    temporal_profile = :constant, world, name = :laser
                ),
                x0 = [0.0, 0.3, 0.3, 0], v0 = 0.3, tspan = (0.0, 4.0),
            ),
        ]
        @testset "$(c.name)" for c in cases
            @named world = Worldline(:τ, :natural)
            sol, sys = solve_ll(c.make(world); x0 = c.x0, v0 = c.v0, tspan = c.tspan)
            @test SciMLBase.successful_retcode(sol)
            @test max_F_rad(sol, sys) > 1.0e-8                  # non-vacuous
            @test max_orthogonality_violation(sol, sys) < 1.0e-10
        end
    end

    @testset "Mass shell uᵘuᵤ = c² preserved" begin
        # Orthogonality is exactly what keeps uᵘuᵤ on the shell; check the consequence.
        @named world = Worldline(:τ, :natural)
        sol, sys = solve_ll(
            UniformField(; E_field = [0, 0, 0.05], B_field = [0, 0, 0], world, name = :laser);
            x0 = zeros(4), v0 = 0.0, tspan = (0.0, 5.0),
        )
        @test SciMLBase.successful_retcode(sol)
        for i in eachindex(sol.t)
            u = sol[sys.u, i]
            @test abs(m_dot(u, u) - 1.0) < 1.0e-8   # c = 1 in natural units
        end
    end

    @testset "Larmor power is a charged-particle property" begin
        # P_rad moved up to ChargedParticle, so a non-radiating ClassicalElectron
        # also exposes it. In a weak field the two powers nearly coincide (RR is an
        # O(τₑ) correction to the acceleration).
        @named world = Worldline(:τ, :natural)
        @named cfield = UniformField(; E_field = [0, 0, 0.01], B_field = [0, 0, 0], world)
        @named ce = ClassicalElectron(; laser = cfield)
        csys = mtkcompile(ce)

        @named world2 = Worldline(:τ, :natural)
        @named lfield = UniformField(; E_field = [0, 0, 0.01], B_field = [0, 0, 0], world = world2)
        @named lle = LandauLifshitzElectron(; laser = lfield)
        lsys = mtkcompile(lle, allow_symbolic = true)

        tspan = (0.0, 0.1)
        solc = solve(
            ODEProblem(csys, [csys.x => zeros(4), csys.u => [1.0, 0, 0, 0]], tspan),
            Vern9(), abstol = 1.0e-12, reltol = 1.0e-12
        )
        soll = solve(
            ODEProblem(lsys, [lsys.x => zeros(4), lsys.u => [1.0, 0, 0, 0]], tspan),
            Vern9(), abstol = 1.0e-12, reltol = 1.0e-12
        )

        @test solc[csys.P_rad, end] ≥ 0        # ClassicalElectron radiates too
        @test soll[lsys.P_rad, end] ≥ 0
        @test isapprox(solc[csys.P_rad, end], soll[lsys.P_rad, end]; rtol = 0.05)
    end

    @testset "No runaway in a moderate field" begin
        @named world = Worldline(:τ, :natural)
        sol, sys = solve_ll(
            UniformField(; E_field = [0, 0, 0.05], B_field = [0, 0, 0], world, name = :laser);
            x0 = zeros(4), v0 = 0.0, tspan = (0.0, 5.0),
        )
        @test SciMLBase.successful_retcode(sol)
        @test maximum(sol[sys.u[1]]) < 10.0     # γ stays bounded (no runaway)
    end

    @testset "Field-gradient term ∂_ν F^μλ" begin
        # The gradient term is live: ∂_ν F is computed symbolically at the field
        # boundary and contracted into the RR force. It must be nonzero for a
        # varying field and exactly zero for a constant one.
        @named world = Worldline(:τ, :natural)
        sol_pw, sys_pw = solve_ll(
            PlaneWave(; amplitude = 0.05, frequency = 1.0, world, name = :laser);
            x0 = [0.0, 0, 0.2, 0], v0 = 0.3, tspan = (0.0, 2.0),
        )
        max_grad_pw = maximum(maximum(abs, sol_pw[sys_pw.radiation.grad_F_dot_u, i]) for i in eachindex(sol_pw.t))
        @test max_grad_pw > 1.0e-8                       # varying field ⇒ nonzero gradient

        @named world2 = Worldline(:τ, :natural)
        sol_uf, sys_uf = solve_ll(
            UniformField(; E_field = [0.05, 0, 0], B_field = [0, 0, 0.1], world = world2, name = :laser);
            x0 = [0.0, 0, 0.2, 0], v0 = 0.3, tspan = (0.0, 2.0),
        )
        max_grad_uf = maximum(maximum(abs, sol_uf[sys_uf.radiation.grad_F_dot_u, i]) for i in eachindex(sol_uf.t))
        @test max_grad_uf == 0.0                         # constant field ⇒ zero gradient
    end
end
