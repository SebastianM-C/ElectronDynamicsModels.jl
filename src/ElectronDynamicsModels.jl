module ElectronDynamicsModels

using ModelingToolkit
using ModelingToolkitBase: AbstractSystem, SymbolicT, build_explicit_observed_function, get_systems
using SymbolicIndexingInterface: getname, setsym_oop, variable_index
using PhysicalConstants, Unitful, UnitfulAtomic
using PhysicalConstants.CODATA2018: c_0, e, m_e, ε_0
using LinearAlgebra
using Symbolics
using HypergeometricFunctions: HypergeometricFunctions, _₁F₁, pochhammer
using StaticArrays
using SciMLBase
using DataInterpolations

m_dot(x, y) = x[1] * y[1] - x[2] * y[2] - x[3] * y[3] - x[4] * y[4]

export GaussLaser, LaguerreGaussLaser
export ReferenceFrame, ProperFrame, LabFrame,
    UniformField, PlaneWave,
    ParticleDynamics,
    LandauLifshitzRadiation, AbrahamLorentzRadiation,
    ChargedParticle,
    ClassicalElectron, RadiatingElectron, LandauLifshitzElectron,
    FieldEvaluator,
    ObserverScreen, trajectory_interpolants, TrajectoryInterpolant, accumulate_potential

include("base.jl")
include("dynamics.jl")
include("fields.jl")
include("radiation.jl")
include("radiation_reaction.jl")
include("external_fields.jl")
include("systems.jl")
include("field_evaluator.jl")

end
