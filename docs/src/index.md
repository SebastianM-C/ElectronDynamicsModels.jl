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

## GPU diagnostics (lib/GPUDiagnostics)

Vendor-neutral runtime diagnostics for the KernelAbstractions kernels, shipped as the
`lib/GPUDiagnostics` sub-package and re-exported by `ElectronDynamicsModels`: the device API
(`gpu_device_count`, `gpu_device!`, `gpu_name`, …), device-event kernel timing (`gpu_event`,
`LaunchTimer`, `launch_times`), telemetry (`gpu_sample` for one snapshot; the out-of-process
sampler `with_gpu_sampler` whose Julia child takes that sample per device per tick — power /
utilization / VRAM everywhere, plus NVIDIA GPM hardware counters where the device has them:
achieved SM occupancy, FP64 / DRAM-bandwidth utilization — and `gpu_telemetry_stats`, which
reduces the resulting column table into the manifest's `[gpu]` keys incl. `gpm_*`), the
measured FP64 peak (`measure_peak_fp64_flops`) and the compile-time resource report
(`compiled_kernels`, `kernel_resources`: registers, spills, LDS and theoretical occupancy of the
kernels the process actually compiled). The CUDA.jl / AMDGPU.jl extensions supply the
vendor methods; the `CPU` backend has host fallbacks.

```@autodocs
Modules = [ElectronDynamicsModels.GPUDiagnostics]
```
