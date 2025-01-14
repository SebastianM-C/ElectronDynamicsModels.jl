using ModelingToolkit, OrdinaryDiffEq
using ElectronDynamicsModels
using LinearAlgebra
using Plots

@named laser = ElectronDynamicsModels.PlaneWave2()
@named electron = ElectronDynamicsModels.ClassicalElectronModel2()

eqs = [
    electron.external_field.E ~ laser.E
    electron.external_field.B ~ laser.B
    electron.x ~ laser.x
]

model = ODESystem(eqs, electron.t; name = :sys, systems = [electron, laser])
sys = structural_simplify(model)

(; a₀, c) = laser

_γ(v) = 1 / √(1 - (v ⋅ v) / c^2)
v0 = -c / ((2 / a₀)^2 + 1)

tspan = (0, 50)

prob = ODEProblem(
    sys,
    [
        electron.γ[1] => _γ(v0)
        electron.v[3] => v0
        # electron.t => tspan[1]
        # electron.x => [c * electron.t, 0, 0, 0]
        electron.p⃗[2] => 0
        # electron.p⃗[3] => 0
    ],
    tspan,
    [],
)

# sol = solve(prob, Vern9(), abstol = 1e-9, reltol = 1e-9)
sol = solve(prob, Rodas5P(), abstol = 1e-7)

plot(sol, idxs = (electron.x[2], electron.x[4]), xlabel = "x", ylabel = "z")



@named laser = ElectronDynamicsModels.PlaneWave2()
@named electron = ElectronDynamicsModels.LandauLifshitzRadiationReaction()

eqs = [
    electron.external_field.E ~ laser.E
    electron.external_field.B ~ laser.B
    electron.x ~ laser.x
]

model = ODESystem(eqs, electron.t; name = :sys, systems = [electron, laser])
sys = structural_simplify(model)

tspan = (0, 50)

prob = ODEProblem(
    sys,
    [
        electron.v[1] => 0
        electron.v[2] => 0
        electron.γ[1] => _γ(v0)
        electron.x[4] => 0
        electron.v[3] => v0
    ],
    tspan,
    [],
)

sol = solve(prob, Rodas5P(), abstol = 1e-7)

plot(sol, idxs = (electron.x[2], electron.x[4]), xlabel = "x", ylabel = "z")
