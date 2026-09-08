```@meta
CurrentModule = ElectronDynamicsModels
```

# Experimental

The `ElectronDynamicsModels.Experimental` submodule holds the batched Tsit5
GPU path (`GPUKernelTsit5`). It is production staging under active
development: retained for electron-batching experiments and not yet validated
against the reference solver (the committed accuracy test exercises
`GPUKernelRK4` only). Prefer the promoted GPU API documented on the
[Home](index.md) page.

```@autodocs
Modules = [ElectronDynamicsModels.Experimental]
```
