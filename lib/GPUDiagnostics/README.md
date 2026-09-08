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

## Telemetry: one sample, one child, one table

```julia
gpu_sample(backend, 1)              # NamedTuple: power_W, compute_util, mem_util, vram_used_B (+ GPM counters on NVIDIA)

result, telem = with_gpu_sampler(backend, 1.0; devices = 1:2, tracefile = "gputrace.tsv") do
    run_the_workload()
end
telem.columns                       # [:t_rel_s, :device, :power_W, :compute_util, :mem_util, :vram_used_B, (:sm_util, :sm_occupancy, :fp64_util, …)]
telem[:compute_util]                # a column; length(telem) rows over all devices
gpu_telemetry_stats(telem)          # "<col>_mean" / "_peak" / "_busy_mean" (rows with compute_util ≥ 0.5), "samples", "busy_samples"
```

`gpu_sample` is exactly what the sampler child calls per tick. The child is a Julia process
(`telemetry_child_main`, started with the parent's julia binary and load path) rather than a Julia
task: an in-process tick either wedges on the vendor runtime behind a backed-up kernel stream, or
is suspended with the sleeping task by Julia's GC/timer coupling while the host thread allocates.
The vendor extension resolves per-device *source specs* in the parent (`gpu_sampler_sources`) so the
child needs no vendor runtime: on AMD it reads the amdgpu driver's sysfs files it was handed and
starts in about a second; on NVIDIA it loads CUDA.jl for its NVML bindings only (no CUDA context,
a few seconds of startup that `telem.first_sample_s` records and the starvation watchdog discounts).
The child declares its own column header, so the parent parses whatever metric set it emits; it
appends rows as it samples (the trace survives a crash) and stops cooperatively via a stopfile, or
on its own if the parent dies. Any failure to build or start the child logs a warning and the
function runs unsampled.

**GPM counters (NVIDIA Hopper and newer).** With `counters = :auto` (default), devices that support
GPU Performance Monitoring — H100 / H200 / GH200 / B200 and, with recent drivers, consumer Blackwell
(RTX 5090 on driver 580 verified); no profiling privileges needed — add ACHIEVED SM occupancy (the
number to hold against the compile-time theoretical occupancy of `kernel_resources`), FP64 / FP32 /
FP16 / tensor / integer pipe utilization, DRAM-bandwidth utilization and PCIe / NVLink traffic, each
averaged over the interval between two consecutive ticks. `counters = :none` skips them. Note that
`fp64_util` is normalised to the SM's full-rate issue slots: a saturated FP64 FMA chain reads ≈ 0.9
on an H100 but only ≈ 1.5 % on a 1/64-rate consumer board.

## Measured FP64 peak

```julia
measure_peak_fp64_flops(backend)   # FLOP/s, dependent-FMA-chain kernel, best of 5
gpu_peak_fp64_flops(backend)       # same; BLAS peakflops on the CPU backend
```

No per-architecture table: the probe measures the attainable vector FP64 rate at the clocks the device
actually holds, and is never routed to matrix/tensor units (the wrong yardstick for scalar kernels).
The result is checked against a host reference so a mis-launched kernel cannot be credited.

## Compile-time resource report

```julia
cks = compiled_kernels(backend; pattern = r"_my_driver!")   # kernels this process compiled
r = kernel_resources(backend, only(cks))                    # at the kernel's static workgroup size
r.registers, r.local_mem_bytes, r.shared_mem_bytes           # per-thread regs, spill/stack bytes, LDS/block
r.active_blocks_per_sm, r.occupancy                          # the runtime's occupancy calculator
r.isa                                                        # AMD: sgpr/vgpr/spill counts, compiler occupancy
```

Both vendor packages cache every kernel instance the process compiles, so the inventory reaches
kernels that are closures inside driver functions (an AcceleratedKernels `foreachindex` body, say)
without wrapping, recompiling or modifying them; on Julia ≥ 1.12 the closure type carries the
enclosing function's name, which is what `pattern` matches. `shared_mem_bytes` is what the kernel
descriptor *reserves* — LLVM's AMDGPU backend promotes private arrays it cannot keep in registers
to LDS, sized for the kernel's maximum block size, and that reservation, not the source, is what
caps the resident blocks per CU. On NVIDIA the report also re-runs CUDA.jl's bundled `ptxas --verbose` on the
regenerated module PTX, which separates the call-ABI stack frame from true register spills and lists the device
functions assembled out of line (what `CUDABackend(always_inline = true)` removes). Nothing is launched; the ISA
dumps cost a few seconds of compiler time.

## Static instruction mix (native or cross-compiled)

```julia
m = kernel_instruction_mix(backend, only(cks))                 # this device's code
m.counts.fp64_fma, m.counts.fp64_add, m.counts.fp64_mul         # whole binary, every path once
m.hot_loop.counts, m.hot_loop_confidence                        # the per-slot loop: one pass, nested loops once
m942 = kernel_instruction_mix(backend, only(cks); target = "gfx942")   # the MI300X code, without an MI300X
m90 = kernel_instruction_mix(backend, only(cks); target = "sm_90")     # the H100 code, from any CUDA context
fp64_issue_floor(m; n_slots = Nx * Ny * N_samples, peak_fp64_flops = measure_peak_fp64_flops(backend),
    kernel_time_s = median_launch_s)                            # FP64-pipe time floor and fraction
```

The disassembly (AMD ISA text from `code_native`; NVIDIA SASS from CUDA.jl's bundled `nvdisasm` on
the cubin) is counted by class — FP64 fma / add / mul / transcendental seeds / other / CDNA packed,
FP32, integer, scalar, memory loads / stores / atomics, constant loads, LDS, control, waits, nops,
other (`MIX_CLASSES`) — for the whole kernel and for every loop of its control-flow graph (natural loops
from dominators), the hot loop being the outermost loop with the most instructions. On AMD the
loop nest is checked against the LLVM assembly printer's own loop annotations (`:high` confidence
when they agree); SASS has none, so the CFG result stands alone (`:medium` with a single dominant
outer loop). `target = "gfx942"` / `"sm_90"` compiles the same kernel — same function, argument
types, `always_inline`, static workgroup size — for another architecture through GPUCompiler, so
the code a rented machine will run can be read before renting it (e.g. whether the CDNA3 compiler
emits `v_pk_fma_f64` at all — for the kernels here it does not; or that ptxas pads the sm_120
Newton loop with 1274 NOPs where the sm_90 loop has none). Static counts count code, not
execution; against Nsight Compute on an RTX 5090 the hot-loop + second-Newton-pass DFMA / DADD+DMUL
counts of the Newton field kernel reproduce the measured per-slot warp-instruction counts exactly
(219 / 208 per slot). `fp64_issue_floor` divides a per-slot FP64 count × executed slots by the
measured FP64 lane-instruction rate (`measure_peak_fp64_flops` / 2 — an FMA is 2 FLOP): the time the
FP64 pipe alone needs per launch, assuming every FP64 instruction issues at the FMA rate.

The classifiers are ordered rule tables, `SASS_RULES` / `AMD_RULES :: Vector{Pair{Regex, Symbol}}`,
built on the mnemonic grammars (SASS: the leading operand-type letter `D`/`F`/`H`/`I`/`U`, the
`LD`/`ST`/`ATOM`/`RED` + `G`/`L`/`S`/`C` memory suffixes; AMD: the `v_`/`s_`/`ds_`/`global_` prefixes and
`_f64`/`_f32` suffixes) with a short exception list ahead of each generic rule; the last rule is a
catch-all whose hits are `other` but reported as `unclassified` / `unclassified_opcodes` / `coverage`
(1.0 on all four validated targets). The vectors are mutable — `pushfirst!(SASS_RULES, r"^MYOP" =>
:int)` overrides for the session. `kernel_ir_mix` (also the `ir` field of `kernel_instruction_mix`)
walks the OPTIMIZED LLVM IR of the same job with LLVM.jl, typed by opcode and operand type (with the `contract` fast-math flag counted as `fp64_contract`: Julia lowers `muladd` to a
contract-flagged `fmul`/`fadd` pair, which the AMD backend fuses only at instruction selection)
(`IR_CLASSES`: fp64 fma / add / mul / div / neg / sqrt / cmp / cvt / intrinsic, fp32, int, memory,
call, control, other) — the arithmetic before the backend: IR `fma` 0 against 132 `v_fma_f64` /
301 DFMA is the backend's FMA contraction, IR `div` 14 / `sqrt` 3 against the machine's
reciprocal seeds + FMA sequences its expansion. Whole-module totals (no IR loop attribution).

Gotchas: CUDA.jl's `code_sass` loads the module on the current device (via CUPTI), which an
`sm_90` cubin cannot do on an `sm_120` GPU — the extension compiles with `CUDACore.compile` and runs
`nvdisasm` on the image directly, so no CUPTI and no device of the target architecture are needed.
AMDGPU.jl caches the OCLC ISA-version device library under an ISA-agnostic key; the extension evicts
it around a cross-compile so gfx942 links `oclc_isa_version_942`, not the current device's.
