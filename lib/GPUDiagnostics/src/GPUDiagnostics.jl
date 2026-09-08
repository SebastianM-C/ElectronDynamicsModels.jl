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
- **Out-of-process telemetry** — `with_gpu_sampler` runs a function while a child process
  samples power / utilization / VRAM per device into a TSV. In-process sampling wedges behind a
  backed-up kernel stream or is suspended by Julia's GC/timer coupling; a child is immune.
- **Measured FP64 peak** — `measure_peak_fp64_flops` / `gpu_peak_fp64_flops`: a dependent-FMA-chain
  kernel gives the attainable vector FP64 rate of the device at the clocks it actually holds
  (never routed to matrix/tensor units); BLAS `peakflops` on the CPU backend.
"""
module GPUDiagnostics

import Adapt
import KernelAbstractions
import KernelAbstractions as KA
using KernelAbstractions: Backend, @kernel, @index, @Const
using LinearAlgebra: LinearAlgebra

export gpu_device_count, gpu_device, gpu_device!, gpu_name, gpu_arch,
    gpu_sm_count, gpu_max_threads_per_sm, gpu_memory_info, gpu_power, gpu_utilization,
    gpu_telemetry_child_cmd, thread_fill_occupancy,
    gpu_event, gpu_elapsed, LaunchTimer, launch_times, launch_lane, launch_tick, launch_tock!,
    with_gpu_sampler,
    measure_peak_fp64_flops, gpu_peak_fp64_flops

include("device_api.jl")   # generics + CPU fallbacks + LaunchTimer; vendor methods in ext/
include("sampler.jl")      # with_gpu_sampler: out-of-process telemetry child (bin/gputrace*.sh)
include("peakflops.jl")    # FMA-chain FP64 peak probe

end
