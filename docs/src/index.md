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

## GPU diagnostics

The GPU instrumentation — device API and device-event kernel timing (`LaunchTimer`), the
out-of-process telemetry sampler, the measured FP64 peak, the compile-time resource report and
the static instruction mix, the per-dispatch hardware counters, and the `diagnostics_dict` report
layer — is the separate package [GPUDiagnostics.jl](https://github.com/SebastianM-C/GPUDiagnostics.jl),
documented at [SebastianM-C.github.io/GPUDiagnostics.jl](https://sebastianm-c.github.io/GPUDiagnostics.jl/).
ElectronDynamicsModels re-exports the entry points it uses; `scripts/gpu_telemetry.jl` reduces
their results into the run manifests' `[gpu]`, `[host]`, `[flops]` and `[timing]` sections, and
`orchestration/profile_cell.sh` + `scripts/hw_counter_merge.jl` run a solver cell under a rocprofv3
counter set and merge the result as `hw_*` keys in the GPUDiagnostics schema 2 layout.
`scripts/instruction_mix.jl` prints the production field kernel's instruction mix, natively or
cross-compiled for a GPU that is not present.

