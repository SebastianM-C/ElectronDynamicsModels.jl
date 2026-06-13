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
using FFTW: rfft, rfftfreq

import Adapt
import AcceleratedKernels as AK
import KernelAbstractions
using KernelAbstractions: Backend, @kernel, @index, @Const

m_dot(x, y) = x[1] * y[1] - x[2] * y[2] - x[3] * y[3] - x[4] * y[4]

# Single Minkowski metric g_{μν} = g^{μν} = diag(1,−1,−1,−1) for the whole package:
# the default value of the symbolic `gμν` parameter (see `ReferenceFrame`) and the
# concrete metric used by `stress_energy` and the numeric screen reduction.
const η = @SMatrix [
    1.0  0.0  0.0  0.0
    0.0 -1.0  0.0  0.0
    0.0  0.0 -1.0  0.0
    0.0  0.0  0.0 -1.0
]

export GaussLaser, LaguerreGaussLaser, a0_from_pulse_energy
export ReferenceFrame, Worldline,
    UniformField, PlaneWave,
    ParticleDynamics,
    LandauLifshitzRadiation,
    ChargedParticle,
    ClassicalElectron, LandauLifshitzElectron,
    FieldEvaluator,
    ObserverScreen, trajectory_interpolants, TrajectoryInterpolant, accumulate_potential,
    accumulate_field, screen_observables, screen_spectrum,
    harmonic_bins, harmonic_maps, power_spectrum,
    plot_harmonic_grid, plot_phase_grid, plot_phase_rings_grid, plot_power_spectrum, harmonic_colorrange, symmetric_colorrange,
    lienard_wiechert_F, lienard_wiechert_F_split, extract_EB, faraday, stress_energy,
    GPUCubicSpline, GPUKernelRK4, GPUKernelTsit5, recommended_n_substeps,
    retarded_time_problem

include("base.jl")
include("dynamics.jl")
include("fields.jl")
include("radiation.jl")
include("harmonics.jl")
include("plotting.jl")
include("radiation_reaction.jl")
include("external_fields.jl")
include("systems.jl")
include("field_evaluator.jl")
include("gpu/interp.jl")
include("gpu/accumulate.jl")
include("gpu/kernel_rk4.jl")

# Experimental: batched/Tsit5 GPU path, under active development.
module Experimental
    include("gpu/experimental.jl")
end

using .Experimental: GPUKernelTsit5

end
