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
using FFTW: rfft, rfftfreq, plan_rfft

import Adapt
import AcceleratedKernels as AK
import KernelAbstractions
import KernelAbstractions as KA
using KernelAbstractions: Backend, @kernel, @index, @Const
using CountedFloats: CountedFloats, Counted, Counts, @count
using GPUDiagnostics   # vendor GPU API, device-event LaunchTimer, sampler, FP64 peak (lib/GPUDiagnostics)

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
    ObserverScreen, observer_window_start, trajectory_span_for_window,
    ObserverScreen, trajectory_interpolants, TrajectoryInterpolant, canonical_state_order, accumulate_potential,
    accumulate_field, screen_observables, screen_spectrum,
    harmonic_bins, harmonic_maps, power_spectrum,
    ring_pixels, phase_winding_fit,
    plot_harmonic_grid, plot_phase_grid, plot_phase_with_rings, plot_phase_polar, plot_power_spectrum, harmonic_colorrange, symmetric_colorrange,
    lienard_wiechert_F, lienard_wiechert_F_split, extract_EB, faraday, stress_energy,
    GPUCubicSpline, GPUKernelRK4, GPUKernelTsit5, GPUKernelNewton, recommended_n_substeps,
    retarded_time_problem,
    # re-exported from GPUDiagnostics (lib/GPUDiagnostics) — the vendor GPU API + LaunchTimer
    gpu_device_count, gpu_device, gpu_device!, gpu_name, gpu_power, gpu_utilization,
    gpu_memory_info, gpu_telemetry_child_cmd, gpu_sm_count, gpu_max_threads_per_sm,
    gpu_arch, gpu_peak_fp64_flops, measure_peak_fp64_flops, thread_fill_occupancy,
    gpu_event, gpu_elapsed, LaunchTimer, launch_times,
    CompiledKernel, compiled_kernels, kernel_resources,
    accumulate_field_sharded,
    window_coverage, flop_profile,
    hann, blackman_harris

include("base.jl")
include("dynamics.jl")
include("fields.jl")
include("radiation.jl")
include("harmonics.jl")
include("oam_analysis.jl")
include("plotting.jl")
include("radiation_reaction.jl")
include("external_fields.jl")
include("systems.jl")
include("field_evaluator.jl")
include("gpu/interp.jl")
include("gpu/accumulate.jl")
include("gpu/kernel_rk4.jl")
include("gpu/kernel_newton.jl")
include("rpr_api.jl")   # RPR rendering plumbing; impls in ext/EDMRPRMakieExt.jl + EDMIsoMeshExt.jl
include("gpu/multidevice.jl")   # accumulate_field_sharded: electron sharding across GPUs
include("diagnostics/window_coverage.jl")   # host-side executed-slot / window-coverage check
include("diagnostics/flops.jl")             # algorithmic FLOP profile of the field kernels (CountedFloats)

# Experimental: batched/Tsit5 GPU path — production staging, under active development.
module Experimental
    include("gpu/experimental.jl")
end

using .Experimental: GPUKernelTsit5

end
