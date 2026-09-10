```@meta
CurrentModule = ElectronDynamicsModels
```

# ElectronDynamicsModels

Documentation for [ElectronDynamicsModels](https://github.com/SebastianM-C/ElectronDynamicsModels.jl).

```@index
```

```@autodocs
Modules = [ElectronDynamicsModels]
```

## GPU diagnostics (GPUDiagnostics.jl)

Vendor-neutral runtime diagnostics for the KernelAbstractions kernels, shipped as the
[GPUDiagnostics.jl](https://github.com/SebastianM-C/GPUDiagnostics.jl) package (formerly the `lib/GPUDiagnostics` sub-package) and re-exported by `ElectronDynamicsModels`: the device API
(`gpu_device_count`, `gpu_device!`, `gpu_name`, …), device-event kernel timing (`gpu_event`,
`LaunchTimer`, `launch_times`), telemetry (`gpu_sample` for one snapshot; the out-of-process
sampler `with_gpu_sampler` whose Julia child takes that sample per device per tick — power /
utilization / VRAM everywhere, plus NVIDIA GPM hardware counters where the device has them:
achieved SM occupancy, FP64 / DRAM-bandwidth utilization — and `gpu_telemetry_stats`, which
reduces the resulting column table into the manifest's `[gpu]` keys incl. `gpm_*`), the
measured FP64 peak (`measure_peak_fp64_flops`), the compile-time resource report
(`compiled_kernels`, `kernel_resources`: registers, spills, LDS and theoretical occupancy of the
kernels the process actually compiled) and the static instruction mix
(`kernel_instruction_mix`: the compiled AMD ISA / NVIDIA SASS counted by class — FP64
fma/add/mul/transcendental/other/packed, FP32, integer, scalar, memory, LDS, control, waits —
for the whole kernel and for the hot loop of its control-flow graph, natively or cross-compiled
for a target that is not present such as `gfx942` or `sm_90`; `fp64_issue_floor` turns the
per-slot FP64 count and the measured FP64 rate into the FP64-pipe time floor per launch;
`kernel_ir_mix` counts the same job's optimized LLVM IR by typed opcode as the pre-backend
reference; `scripts/instruction_mix.jl` prints the table). The CUDA.jl / AMDGPU.jl extensions supply the
vendor methods; the `CPU` backend has host fallbacks.

AMD hardware counters go through **rocprofv3** (there is no in-process counter API on AMD, so
the profiler wraps the whole solver process): `rocprof_command` builds that wrapper around a
`Cmd` under `timeout -k` (a counter set the hardware refuses aborts rocprofv3 and leaves the
child hung — `ROCPROF_COUNTER_SETS` names the sets verified to collect in one pass on gfx942:
instruction issue `:sq_issue`, wave residency `:sq_waves`, the L1 pipe `:l1_pipe`, `:fp64`,
`:l2`, and its docstring the six-counter L2 set that does not), `rocprof_counters` parses the
resulting CSV into the per-dispatch counters of one kernel (`RocprofCounters`: values,
dispatch durations, the VGPR/AGPR/SGPR/LDS/scratch footprint, the device from the agent info),
and `rocprof_derived` / `rocprof_summary` / `rocprof_manifest_section` reduce it — medians with
spreads across dispatches, per-slot instruction counts (`SQ_INSTS_* × wave size / slots`),
unit-busy fractions (`X_BUSY_sum / (cycles × n_cu)`), L1/L2 miss rates, wave wait/active
fractions and achieved occupancy — with the normalisation of rocprofv3's own derived metrics:
the CSV sums `GRBM_GUI_ACTIVE` over the dies (8 XCDs on the MI300X), so `cycles =
GRBM_GUI_ACTIVE / n_xcd`, and the SQ wave-cycle counters are in quad-cycles.
`orchestration/profile_cell.sh <set> <outdir> <tag> [EDM_VAR=val …]` runs one solver cell under
a set and merges the result into the run manifest's `[gpu]` as `rocprof_*` keys
(`scripts/rocprof_merge.jl`, which takes the slots per dispatch from `[flops].slots_executed`
and the launch count) — profile the same cell once per set.

```@autodocs
Modules = [ElectronDynamicsModels.GPUDiagnostics]
```
