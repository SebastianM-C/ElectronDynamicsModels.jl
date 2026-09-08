# GPUDiagnostics.jl

Vendor-neutral runtime diagnostics for [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
code. Everything dispatches on the KA `Backend`; CUDA.jl and AMDGPU.jl package extensions supply the
vendor methods, and the `CPU` backend gets host fallbacks so the plumbing runs (and is tested) without a GPU.
No kernel is modified by any of it.

```julia
using GPUDiagnostics, CUDA          # or AMDGPU
backend = CUDABackend()

gpu_device_count(backend), gpu_name(backend), gpu_arch(backend)
gpu_sm_count(backend) * gpu_max_threads_per_sm(backend)   # resident-thread capacity
gpu_memory_info(backend), gpu_power(backend), gpu_utilization(backend)
```

## Device-event kernel timing

When launches are queued asynchronously, a host clock around a launch measures enqueue latency.
An event recorded on the launch stream fires when the GPU reaches it in stream order, so a pair
around a launch brackets exactly that kernel — two barrier packets per launch, no host stall.

```julia
timer = LaunchTimer()
lane = launch_lane(timer, backend)          # once per loop (device id)
for item in work
    e0 = launch_tick(timer, backend)
    my_kernel!(backend)(item; ndrange = n)  # asynchronous
    launch_tock!(timer, lane, backend, e0)
end
launch_times(timer)                          # Dict(device => [seconds per launch, …])
```

Pass `nothing` instead of a timer and the hooks are no-ops. The timer is safe to share across
per-device tasks (pushes are locked; events are per-stream).

## Out-of-process telemetry

```julia
result, telem = with_gpu_sampler(backend, 1.0; devices = 1:2, tracefile = "gputrace.tsv") do
    run_the_workload()
end
telem.samples   # (t_rel_s, device, power_W, compute_util, mem_util, vram_used_B) rows
```

The sampler is a child process (`bin/gputrace.sh` over the amdgpu driver's sysfs; `bin/gputrace_cuda.sh`
over `nvidia-smi`) rather than a Julia task: an in-process tick either wedges on the vendor runtime
behind a backed-up kernel stream, or is suspended with the sleeping task by Julia's GC/timer coupling
while the host thread allocates. The child appends rows as it samples (the trace survives a crash)
and stops cooperatively via a stopfile, or on its own if the parent dies.

## Measured FP64 peak

```julia
measure_peak_fp64_flops(backend)   # FLOP/s, dependent-FMA-chain kernel, best of 5
gpu_peak_fp64_flops(backend)       # same; BLAS peakflops on the CPU backend
```

No per-architecture table: the probe measures the attainable vector FP64 rate at the clocks the device
actually holds, and is never routed to matrix/tensor units (the wrong yardstick for scalar kernels).
The result is checked against a host reference so a mis-launched kernel cannot be credited.
