module ElectronDynamicsModels

using ModelingToolkit
using Unitful, UnitfulAtomic, PhysicalConstants
using PhysicalConstants.CODATA2018: e, m_e, c_0, ε_0
using LinearAlgebra
using Symbolics
using HypergeometricFunctions: HypergeometricFunctions, _₁F₁, pochhammer

m_dot(x, y) = x[1] * y[1] - x[2] * y[2] - x[3] * y[3] - x[4] * y[4]

export GaussLaser, LaguerreGaussLaser
export ReferenceFrame, ProperFrame, LabFrame,
    UniformField,
    PlaneWave,
    ParticleDynamics,
    LandauLifshitzRadiation,
    AbrahamLorentzRadiation,
    ChargedParticle,
    ClassicalElectron,
    RadiatingElectron,
    LandauLifshitzElectron

include("base.jl")
include("dynamics.jl")
include("fields.jl")
include("radiation_reaction.jl")
include("external_fields.jl")
include("systems.jl")

end
