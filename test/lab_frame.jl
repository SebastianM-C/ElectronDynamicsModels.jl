using ModelingToolkit, OrdinaryDiffEqVerner, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
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

model = System(eqs, electron.t; name = :sys, systems = [electron, laser])
sys = mtkcompile(model)

(; a₀, c) = laser

_γ(v) = 1 / √(1 - (v ⋅ v) / c^2)
v0 = -c / ((2 / a₀)^2 + 1)

tspan = (0, 350) # t[end] has to be ∫γdτ to be comparable with the proper frame

prob = ODEProblem(
    sys,
    [
        electron.γ => _γ(v0)
        electron.p⃗[1] => 0
        electron.p⃗[2] => 0
        electron.p⃗[3] => _γ(v0)*electron.m*v0
        electron.τ => 0
        electron.x[2] => 0
        electron.x[3] => 0
        electron.x[4] => 0
    ],
    tspan
)

sol = solve(prob, Vern9(), abstol = 1e-9, reltol = 1e-9)
# sol = solve(prob, Rodas5P(), abstol = 1e-7)

plot(sol, idxs = (electron.x[2], electron.x[4]), xlabel = "x", ylabel = "z")



@named laser = ElectronDynamicsModels.PlaneWave2()
@named electron = ElectronDynamicsModels.LandauLifshitzRadiationReaction()

eqs = [
    electron.external_field.E ~ laser.E
    electron.external_field.B ~ laser.B
    electron.x ~ laser.x
]

model = System(eqs, electron.t; name = :sys, systems = [electron, laser])
sys = mtkcompile(model)

tspan = (0, 350)

prob = ODEProblem(
    sys,
    [
        electron.γ => _γ(v0)
        electron.p⃗[1] => 0
        electron.p⃗[2] => 0
        electron.p⃗[3] => _γ(v0) * electron.m * v0
        electron.τ => 0
        electron.x[2] => 0
        electron.x[3] => 0
        electron.x[4] => 0
    ],
    tspan
)

sol = solve(prob, Vern9(), abstol = 1e-9, reltol = 1e-9)
sol = solve(prob, Rodas5P(), abstol = 1e-7)

plot(sol, idxs = (electron.x[2], electron.x[4]), xlabel = "x", ylabel = "z")
