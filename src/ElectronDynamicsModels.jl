module ElectronDynamicsModels

using ModelingToolkit
using ModelingToolkitBase: AbstractSystem, SymbolicT, build_explicit_observed_function, get_systems, getdefault
using SymbolicIndexingInterface: getname, setsym_oop, variable_index
using PhysicalConstants, Unitful, UnitfulAtomic
using PhysicalConstants.CODATA2018: c_0, e, m_e, ε_0
using LinearAlgebra
using Symbolics
using HypergeometricFunctions: HypergeometricFunctions, _₁F₁, pochhammer
using StaticArrays
using SciMLBase
using DataInterpolations

import Adapt
import AcceleratedKernels as AK
import KernelAbstractions
using KernelAbstractions: Backend, @kernel, @index, @Const

m_dot(x, y) = x[1] * y[1] - x[2] * y[2] - x[3] * y[3] - x[4] * y[4]

export GaussLaser, LaguerreGaussLaser, a0_from_pulse_energy
export ReferenceFrame, Worldline,
    UniformField, PlaneWave,
    ParticleDynamics,
    LandauLifshitzRadiation, AbrahamLorentzRadiation,
    ChargedParticle,
    ClassicalElectron, RadiatingElectron, LandauLifshitzElectron,
    FieldEvaluator,
    ObserverScreen, trajectory_interpolants, TrajectoryInterpolant, accumulate_potential,
    accumulate_intensity,
    gpu_trajectory_interpolants, GPUCubicSpline, GPUKernelRK4, GPUKernelTsit5,
    retarded_time_problem

include("base.jl")
include("dynamics.jl")
include("fields.jl")
include("radiation.jl")
include("radiation_reaction.jl")
include("external_fields.jl")
include("systems.jl")
include("field_evaluator.jl")
include("gpu_interp.jl")
include("gpu_radiation.jl")

# Experimental
module Experimental
    include("experimental.jl")
end

end
