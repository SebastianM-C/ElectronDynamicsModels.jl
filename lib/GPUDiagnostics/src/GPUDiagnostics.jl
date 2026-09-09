"""
    GPUDiagnostics

Vendor-neutral runtime diagnostics for KernelAbstractions code. Every entry point dispatches on
the KA `Backend`; the CUDA.jl / AMDGPU.jl package extensions supply the vendor methods, loaded
on demand when the vendor package is in the session, and the KA `CPU` backend gets host
fallbacks so the plumbing (and its tests) run without a GPU.

- **Device API** — `gpu_device_count`, `gpu_device`, `gpu_device!`, `gpu_name`, `gpu_arch`,
  `gpu_sm_count`, `gpu_max_threads_per_sm`, `gpu_memory_info`, `gpu_power`, `gpu_utilization`,
  `thread_fill_occupancy`. KernelAbstractions has no device management of its own.
- **Device-event kernel timing** — `gpu_event` / `gpu_elapsed` on the task-local launch stream,
  and `LaunchTimer` + `launch_times` for one event pair per launch: the only kernel clock that
  works when launches are queued asynchronously (a host clock measures enqueue latency).
- **Telemetry** — `gpu_sample` is one snapshot of a device (power / utilization / VRAM, plus
  NVIDIA GPM hardware counters where available: ACHIEVED SM occupancy, FP64 / FP32 / tensor pipe
  and DRAM-bandwidth utilization, PCIe / NVLink traffic); `with_gpu_sampler` runs a function while
  a child process takes that sample per device per tick into a TSV, and `gpu_telemetry_stats`
  reduces the resulting `GPUTelemetry` column table. In-process sampling wedges behind a backed-up
  kernel stream or is suspended by Julia's GC/timer coupling; a child is immune.
- **Measured FP64 peak** — `measure_peak_fp64_flops` / `gpu_peak_fp64_flops`: a dependent-FMA-chain
  kernel gives the attainable vector FP64 rate of the device at the clocks it actually holds
  (never routed to matrix/tensor units); BLAS `peakflops` on the CPU backend.
- **Compile-time resource report** — `compiled_kernels` inventories the kernels this process
  compiled (from the vendor's kernel cache, so closures inside driver functions are reachable
  too) and `kernel_resources` reports registers, spill/stack and shared (LDS) memory, and the
  runtime's theoretical occupancy for the block size actually launched — plus the SGPR/VGPR/
  spill counts from the AMD ISA dump.
- **Static instruction mix** — `kernel_instruction_mix` counts the compiled instruction stream
  (AMD ISA / NVIDIA SASS) by class — FP64 fma/add/mul/transcendental/other/packed, FP32,
  integer, scalar, memory, LDS, control, waits — for the whole kernel and for each loop of its
  control-flow graph (the hot per-slot loop in particular), natively or **cross-compiled** for a
  target that is not present (`target = "gfx942"`, `"sm_90"`); `kernel_ir_mix` counts the same
  job's optimized LLVM IR by typed opcode (the arithmetic before backend contraction);
  `fp64_issue_floor` turns a per-slot FP64 count and the measured FP64 rate into the FP64-pipe
  time floor per launch.
- **AMD hardware counters (rocprofv3)** — `rocprof_command` wraps a process in a rocprofv3 counter
  collection (`ROCPROF_COUNTER_SETS`: instruction issue, wave residency, the L1 pipe, FP64, L2 —
  the sets that fit one pass on gfx942, and the one that does not), `rocprof_counters` parses its
  CSV into per-dispatch values of one kernel, and `rocprof_derived` / `rocprof_summary` /
  `rocprof_manifest_section` reduce them: medians with spreads, per-slot instruction counts,
  unit-busy fractions and achieved occupancy with the normalisation rocprofv3's own derived
  metrics use (die-summed `GRBM_GUI_ACTIVE`, quad-cycle SQ counters). Pure Julia, no GPU needed
  to parse.
"""
module GPUDiagnostics

import Adapt
import KernelAbstractions
import KernelAbstractions as KA
using KernelAbstractions: Backend, @kernel, @index, @Const
using LinearAlgebra: LinearAlgebra

export gpu_device_count, gpu_device, gpu_device!, gpu_name, gpu_arch,
    gpu_sm_count, gpu_max_threads_per_sm, gpu_memory_info, gpu_power, gpu_utilization,
    thread_fill_occupancy,
    gpu_event, gpu_elapsed, LaunchTimer, launch_times, launch_lane, launch_tick, launch_tock!,
    gpu_sample, gpu_sampler_sources, sampler_source, SamplerSource, telemetry_child_main,
    with_gpu_sampler, GPUTelemetry, gpu_telemetry_stats,
    measure_peak_fp64_flops, gpu_peak_fp64_flops,
    CompiledKernel, compiled_kernels, kernel_resources,
    MIX_CLASSES, SASS_RULES, AMD_RULES, instruction_mix, kernel_instruction_mix, fp64_issue_floor,
    IR_CLASSES, kernel_ir_mix,
    ROCPROF_COUNTER_SETS, RocprofCounters, rocprof_available, rocprof_command, rocprof_counters,
    rocprof_median, rocprof_derived, rocprof_summary, rocprof_manifest_section

include("device_api.jl")   # generics + CPU fallbacks + LaunchTimer; vendor methods in ext/
include("sampler.jl")      # gpu_sample sources, the telemetry child, with_gpu_sampler, gpu_telemetry_stats
include("peakflops.jl")    # FMA-chain FP64 peak probe
include("resources.jl")    # compile-time resource report: registers / spills / LDS / occupancy
include("instruction_mix.jl")   # static instruction mix of the disassembly + loop nest + FP64-issue floor
include("rocprof.jl")           # AMD hardware counters: rocprofv3 wrapper + CSV parser + normalised derived metrics

end
