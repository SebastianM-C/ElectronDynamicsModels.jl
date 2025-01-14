module ElectronDynamicsModels

using ModelingToolkit
using Unitful, UnitfulAtomic, PhysicalConstants
using PhysicalConstants.CODATA2018: e, m_e, c_0, ε_0
using LinearAlgebra
using Symbolics

const gμν = [
    1 0 0 0
    0 -1 0 0
    0 0 -1 0
    0 0 0 -1
]

@independent_variables τ
const dτ = Differential(τ)

export τ, dτ
export ClassicalElectronModel, LandauLifshitzRadiationReaction
export ElectromagneticField, PlaneWave, GaussLaser

include("lorentz.jl")
include("radiation_reaction.jl")
include("external_fields.jl")

end
