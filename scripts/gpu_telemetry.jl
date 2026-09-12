# GPU telemetry for the solver manifests' [gpu] section — shared by thomson_scattering.jl,
# lpwa.jl, inverse_thomson_scattering.jl and occupancy_bench.jl. The instruments live in
# GPUDiagnostics.jl (device snapshot, out-of-process sampler `with_gpu_sampler` incl. GPM
# counters, device-event `LaunchTimer`, measured FP64 peak); this file reduces them into manifest
# sections. Everything
# is wrapped so a telemetry hiccup NEVER breaks a run — it just omits the section.

using GPUDiagnostics

# Static device snapshot + reduced sampler stats → the manifest's [gpu] table (a plain Dict
# that RunManifests writes verbatim as a top-level section). `n_threads` = the field kernel's
# launched index space per device, Nx·Ny·sample_chunks (one thread per pixel and chunk), recorded
# as `launch_threads` and reduced to `thread_fill_occupancy` = launch_threads / (SMs × threads
# per SM): the launch against the device's whole resident capacity, an upper bound on any
# occupancy and > 1 whenever the grid exceeds one full wave. Stats reduce over ALL devices' rows
# (device_count records the fan-out; the per-device time series lives in the gputrace TSV) via
# `gpu_telemetry_stats`; NaN entries (counters a device doesn't expose) are skipped per column.
# The base columns keep their historical keys (`power_mean/_peak`, `compute_util_mean/_peak`,
# `memory_util_mean/_peak`, `vram_used_peak`, `samples`, `sample_dt`, `sampler_starved` — the
# dashboard, scaling_report.jl and compare_hmaps.jl read them); hardware-counter columns (NVIDIA
# GPM: achieved `sm_occupancy` to hold against the compile-time `kernel_occupancy`, `fp64_util`,
# `dram_bw_util`, …) land as `gpm_<column>_mean/_peak/_busy_mean`, where `_busy_mean` averages
# the rows with compute_util ≥ 0.5 — the kernel-active part of the field window, without the
# JIT / upload / drain idle diluting it. Returns `nothing` if telemetry is unavailable (e.g. no
# vendor extension) so the caller just omits [gpu].
const GPU_BASE_COLUMNS = ("power_W" => "power", "compute_util" => "compute_util", "mem_util" => "memory_util",
    "vram_used_B" => "vram_used")
# The other base columns of every sample (clocks, temperatures, power limit, throttle bitmask —
# GPUDiagnostics ≥ 0.2) land as `sampler_<column>_<stat>`; anything else is a GPM counter.
const GPU_SAMPLER_COLUMNS = ("sm_clock_MHz", "mem_clock_MHz", "temperature_C", "hotspot_C", "power_limit_W", "throttle_reasons")
function gpu_manifest_section(backend, backend_name::AbstractString, n_threads::Integer,
        device_count::Integer, telem)
    try
        gpu = Dict{String, Any}(
            "backend" => String(backend_name),
            "device" => gpu_name(backend),
            "device_count" => Int(device_count),
            "sm_count" => Int(gpu_sm_count(backend)),
            "max_threads_per_sm" => Int(gpu_max_threads_per_sm(backend)),
            "memory_total" => Int(gpu_memory_info(backend).total),
            "launch_threads" => Int(n_threads),
            "thread_fill_occupancy" => Float64(thread_fill_occupancy(backend, n_threads)),
        )
        if telem.ticks > 0
            gpu["samples"] = telem.ticks
            gpu["sample_dt"] = telem.dt
            gpu["sampler_starved"] = telem.starved
            ismissing(telem.first_sample_s) || (gpu["sampler_first_sample_s"] = telem.first_sample_s)
            st = gpu_telemetry_stats(telem)
            gpu["sampler_busy_samples"] = st["busy_samples"]
            for (col, key) in GPU_BASE_COLUMNS
                if col == "vram_used_B"   # historically the peak only
                    haskey(st, col * "_peak") && (gpu["vram_used_peak"] = st[col * "_peak"])
                    continue
                end
                haskey(st, col * "_mean") || continue
                gpu[key * "_mean"] = st[col * "_mean"]
                gpu[key * "_peak"] = st[col * "_peak"]
            end
            for (k, v) in st
                k in ("samples", "busy_samples") && continue
                any(startswith(k, col * "_") for (col, _) in GPU_BASE_COLUMNS) && continue
                if k == "power_capped_fraction" || any(startswith(k, col * "_") for col in GPU_SAMPLER_COLUMNS)
                    gpu["sampler_" * k] = v
                else
                    gpu["gpm_" * k] = v
                end
            end
            haskey(gpu, "sampler_power_capped_fraction") && gpu["sampler_power_capped_fraction"] > 0 &&
                @warn "power-capped for $(round(100 * gpu["sampler_power_capped_fraction"]; digits = 1)) % of the compute-busy samples" sm_clock_busy_median_MHz = get(st, "sm_clock_MHz_busy_median", missing) power_limit_W = get(st, "power_limit_W_mean", missing)
            if haskey(st, "sm_occupancy_busy_mean")
                @info "GPM counters (field window, compute-busy rows)" busy_samples = st["busy_samples"] sm_occupancy = round(st["sm_occupancy_busy_mean"]; digits = 3) fp64_util = round(get(st, "fp64_util_busy_mean", NaN); digits = 3) dram_bw_util = round(get(st, "dram_bw_util_busy_mean", NaN); digits = 3) sm_util = round(get(st, "sm_util_busy_mean", NaN); digits = 3)
            end
        end
        return gpu
    catch err
        @warn "GPU telemetry unavailable — omitting [gpu] from the manifest" exception = err
        return nothing
    end
end

# ── Device-event kernel timing → [timing].kernel + [gpu].kernel_* + the kerneltimes TSV ──────────
#
# `LaunchTimer` (GPUDiagnostics.jl) records a device-event pair per kernel launch. It is the only
# kernel clock that works with the production `sync_per_electron = false`: [timing].field is the
# whole phase (first-launch JIT, per-electron spline conversion + upload, the drain, the sampler's
# stop wait, and the slowest shard), while [timing].kernel is the BUSIEST device's summed kernel
# seconds — the denominator for achieved FLOP/s and the clean multi-device scaling metric
# (scaling_report.jl prefers it over the ≥ 99 %-utilization gputrace proxy). Per-device sums,
# launch counts, first-launch (JIT) and median/max per-launch seconds go to [gpu]; the full
# per-launch series to `kerneltimes_<tag>.tsv` (device, launch, seconds), which the caller
# declares as `[outputs].kernel_times` once written. Same contract as the other helpers: a
# failure logs and leaves the manifest without the fields.
function record_kernel_timing!(timing::AbstractDict, gpu, timer; tracefile = nothing)
    try
        lt = launch_times(timer)
        isempty(lt) && return nothing
        devs = sort!(collect(keys(lt)))
        per = [lt[d] for d in devs]
        timing["kernel"] = maximum(sum, per)
        # kernel_devices / _s / _launches / _first_s / _median_s / _max_s, one vector per statistic
        gpu === nothing || merge!(gpu, diagnostics_dict(timer; prefix = "kernel_"))
        if tracefile !== nothing
            open(tracefile, "w") do io
                println(io, "# device\tlaunch\tkernel_s")
                for (d, v) in zip(devs, per), (i, s) in enumerate(v)
                    println(io, d, '\t', i, '\t', s)
                end
            end
        end
        return timing["kernel"]
    catch err
        @warn "kernel timing unavailable — manifest keeps the phase wall time only" exception = err
        return nothing
    end
end

# ── Compile-time resource report → [gpu].kernel_registers / _local_mem_bytes / _shared_mem_bytes /
#    _occupancy … (+ kernel_isa_* extras) ─────────────────────────────────────────────────────────
#
# After the field phase the production kernel sits in the vendor's compiled-kernel cache;
# `compiled_kernels` + `kernel_resources` (GPUDiagnostics.jl) read it back from there — the exact
# code that ran, not a re-creation of it, and no kernel change — and report what the compiler gave
# it (registers per thread, spill/stack bytes, the shared/LDS bytes the descriptor reserves) with
# the theoretical occupancy the runtime derives at the launched block size (the kernel's static
# 256-thread AcceleratedKernels workgroup, read from its signature). The kernels are closures inside
# the `accumulate_*` drivers; on Julia ≥ 1.12 their types carry the driver's name, which `pattern`
# selects (`kernel_driver` records the match). Multi-device runs compile one instance per device —
# the first is reported, `kernel_compiled_matches` keeps the count. AMD ISA figures (SGPR/VGPR/
# spill counts, the compiler's own waves-per-SIMD estimate) flatten to `kernel_isa_*`. Same
# contract as the other helpers: a failure logs and leaves the manifest without the fields.
#
# The same kernel's STATIC instruction mix (`kernel_instruction_mix`: the disassembly counted by
# class) follows as `kernel_mix_<class>` — whole-binary totals, every path counted once, cold
# exception code included — plus `kernel_mix_total`, `kernel_mix_fp64` (the six FP64 classes) and,
# when the control-flow graph yields a hot loop, `kernel_mix_hot_loop_total` / `_fp64` (one pass
# through the per-slot loop, nested loops counted once) with `kernel_mix_hot_loop_confidence`,
# `kernel_mix_coverage` (share of instructions a classification rule matched) and the typed
# LLVM-IR counts of the same job, `kernel_ir_fp64_fma/_add/_mul/_div/_sqrt` (the arithmetic
# before the backend fused and expanded it). Cross-compiled targets (`target = "gfx942"`) are an offline tool (scripts/instruction_mix.jl),
# never written to manifests.
const FIELD_KERNEL_PATTERN = r"_gpu_\w*field_one_electron!"
function record_kernel_resources!(gpu, backend; pattern = FIELD_KERNEL_PATTERN, block_size = nothing)
    gpu === nothing && return nothing
    try
        cks = compiled_kernels(backend; pattern)
        if isempty(cks)
            @warn "kernel resource report: no compiled kernel matches $pattern — omitting [gpu].kernel_registers/…"
            return nothing
        end
        r = block_size === nothing ? kernel_resources(backend, first(cks)) :
            kernel_resources(backend, first(cks); block_size)
        m = match(pattern, r.signature)
        gpu["kernel_name"] = r.name
        gpu["kernel_driver"] = m === nothing ? "" : String(m.match)
        gpu["kernel_compiled_matches"] = length(cks)
        # kernel_registers / _local_mem_bytes / … / _occupancy / _isa_*: a count the runtime cannot
        # report is omitted (never a sentinel), as is the whole occupancy block without a calculator.
        merge!(gpu, diagnostics_dict(r; prefix = "kernel_"))
        @info "kernel resources (compile time)" kernel = gpu["kernel_driver"] report = sprint(show, MIME"text/plain"(), r)
        record_kernel_mix!(gpu, backend, first(cks))
        return r
    catch err
        @warn "kernel resource report unavailable — omitting [gpu].kernel_registers/…" exception = (err, catch_backtrace())
        return nothing
    end
end

# `[gpu].kernel_mix_*` from the static instruction mix of the reported kernel (see above).
function record_kernel_mix!(gpu, backend, ck)
    gpu === nothing && return nothing
    try
        mix = kernel_instruction_mix(backend, ck)
        # kernel_mix_<class> / _total / _fp64 / _coverage / _hot_loop_* (the historical keys plus the
        # rest of the mix) and, separately prefixed, the typed LLVM-IR counts kernel_ir_<class>.
        merge!(gpu, diagnostics_dict(mix; prefix = "kernel_mix_"))
        mix.ir === nothing || merge!(gpu, diagnostics_dict(mix.ir; prefix = "kernel_ir_"))
        hot = mix.hot_loop
        @info "kernel instruction mix (static)" target = mix.target total = mix.total coverage = round(mix.coverage; digits = 4) fp64 = mix.fp64 fp64_fma = mix.counts.fp64_fma fp64_add = mix.counts.fp64_add fp64_mul = mix.counts.fp64_mul fp64_packed = mix.counts.fp64_packed waits = mix.counts.wait hot_loop_total = hot === nothing ? missing : hot.total hot_loop_fp64 = hot === nothing ? missing : gpu["kernel_mix_hot_loop_fp64"] confidence = mix.hot_loop_confidence
        return mix
    catch err
        @warn "kernel instruction mix unavailable — omitting [gpu].kernel_mix_*" exception = (err, catch_backtrace())
        return nothing
    end
end

# → the manifest's [host] table: what the run happened on (Julia / BLAS threads against the host
# cores AND the cgroup CPU quota, memory limits, OS kernel, driver / runtime / vendor package
# versions), with any oversubscription finding under `warnings`. Same contract: `nothing` ⇒ omitted.
function host_manifest_section(backend)
    try
        h = host_snapshot(backend)
        isempty(h.warnings) || @warn "host environment" warnings = h.warnings
        return diagnostics_dict(h)
    catch err
        @warn "host snapshot unavailable — omitting [host] from the manifest" exception = err
        return nothing
    end
end

# Sampler cadence (s); coarse is fine — field runs are seconds→hours. Override with EDM_GPU_SAMPLE_DT.
const GPU_SAMPLE_DT = parse(Float64, get(ENV, "EDM_GPU_SAMPLE_DT", "1.0"))

# ── Observer-window coverage + algorithmic FLOP accounting (host side, no GPU impact) ──────────
#
# `check_window_coverage` runs BEFORE the field phase (milliseconds): every pixel must see every
# electron's full history inside the sampled window, else the GPU kernels silently skip the
# clipped slots and the cube lacks those contributions — a warning here is cheaper than a wasted
# GPU run. Its executed-slot count is the exact work figure of the FLOP section. Both helpers
# follow gpu_manifest_section's contract: a failure never breaks a run, the section is omitted.

function check_window_coverage(trajs, screen)
    try
        t = @elapsed cov = window_coverage(trajs, screen)
        if cov.ok
            @info "observer window fully covered" slot_fill = 1.0 lead_margin_samples = cov.lead_margin_samples tail_margin_samples = cov.tail_margin_samples check_s = round(t; digits = 3)
        else
            @warn "observer window NOT fully covered — $(cov.electrons_clipped) electron(s) end (or start) before the window does at some pixel; those slots are skipped by the kernel and the cube lacks their contribution" slot_fill = cov.slot_fill slots_dropped = cov.slots_dropped lead_margin_samples = cov.lead_margin_samples tail_margin_samples = cov.tail_margin_samples worst_electron = cov.worst_electron
        end
        return cov
    catch err
        @warn "window coverage check failed — running without it" exception = err
        return nothing
    end
end

# → the manifest's [window] table (`nothing` ⇒ omitted).
function window_manifest_section(cov)
    cov === nothing && return nothing
    try
        w = Dict{String, Any}(
            "ok" => cov.ok,
            "slots_nominal" => cov.slots_nominal,
            "electrons_clipped" => cov.electrons_clipped,
            "lead_margin_samples" => cov.lead_margin_samples,
            "tail_margin_samples" => cov.tail_margin_samples,
            "worst_electron" => cov.worst_electron,
        )
        if cov.slots_executed !== missing
            w["slots_executed"] = cov.slots_executed
            w["slots_dropped"] = cov.slots_dropped
            w["slot_fill"] = Float64(cov.slot_fill)
        end
        return w
    catch err
        @warn "window coverage section failed — omitting [window] from the manifest" exception = err
        return nothing
    end
end

# → the manifest's [flops] table: the algorithmic FLOP profile of the kernel actually used
# (`flop_profile`, CountedFloats on the CPU backend, milliseconds), scaled by the run's executed
# slots, with the per-device FLOP rate over the field wall time and the fraction of the device's
# attainable vector FP64 peak (MEASURED on the device by the FMA-chain probe, ~1.5 s, after the
# field phase). `slots_executed = missing` falls back to the nominal N·N_samples·Nx·Ny.
function flops_manifest_section(backend, alg, mode::Symbol, solver_kw, N, Nx, Ny, N_samples,
        slots_executed, t_field, ndev)
    try
        p = flop_profile(alg; mode = Val(mode), solver_kw...)
        slots_nominal = N * N_samples * Nx * Ny
        executed = slots_executed === missing ? slots_nominal : Int(slots_executed)
        flop_total = p.flop_per_slot * executed + p.flop_per_pixel_launch * N * Nx * Ny
        rate = flop_total / (t_field * ndev)
        f = Dict{String, Any}(
            "convention" => p.convention,
            "counter_version" => 1,
            "alg" => p.alg,
            "mode" => String(p.mode),
            String(p.n_name) => p.n,
            "flop_per_slot" => p.flop_per_slot,
            "flop_per_slot_add" => p.per_slot.add,
            "flop_per_slot_mul" => p.per_slot.mul,
            "flop_per_slot_div" => p.per_slot.div,
            "flop_per_slot_sqrt" => p.per_slot.sqrt,
            "flop_per_slot_fma" => p.per_slot.fma,
            "flop_per_slot_pow" => p.per_slot.pow,
            "flop_per_slot_trans" => p.per_slot.trans,
            "nonflop_per_slot_cmp" => p.per_slot.cmp,
            "nonflop_per_slot_other" => p.per_slot.other,
            "flop_per_pixel_launch" => p.flop_per_pixel_launch,
            "bytes_per_slot" => p.bytes_per_slot,
            "arithmetic_intensity" => p.arithmetic_intensity,
            "slots_nominal" => slots_nominal,
            "slots_executed" => executed,
            "slots_executed_source" => slots_executed === missing ? "nominal" : "window_coverage",
            "slot_fill" => executed / slots_nominal,
            "flop_total" => flop_total,
            "bytes_total" => p.bytes_per_slot * executed,
            "device_count" => Int(ndev),
            "flop_rate_field" => rate,   # FLOP/s per device over the field wall time
        )
        arch = try
            gpu_arch(backend)
        catch
            nothing
        end
        arch === nothing || (f["gpu_arch"] = String(arch))
        peak = try
            Float64(measure_peak_flops(backend))
        catch err
            @warn "FP64 peak measurement failed — omitting peak_fp64_flops / peak_fraction_field" exception = (err, catch_backtrace())
            NaN
        end
        if isfinite(peak) && peak > 0
            f["peak_fp64_flops"] = peak
            f["peak_fp64_method"] = "fma-chain-measured"   # the same probe on every backend, the CPU included
            f["peak_fraction_field"] = rate / peak
        end
        return f
    catch err
        @warn "FLOP accounting unavailable — omitting [flops] from the manifest" exception = err
        return nothing
    end
end
