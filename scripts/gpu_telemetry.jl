# GPU telemetry for the solver manifests' [gpu] section — shared by thomson_scattering.jl,
# lpwa.jl, inverse_thomson_scattering.jl and occupancy_bench.jl. The instruments live in
# lib/GPUDiagnostics (device snapshot, out-of-process sampler `with_gpu_sampler`, device-event
# `LaunchTimer`, measured FP64 peak); this file reduces them into manifest sections. Everything
# is wrapped so a telemetry hiccup NEVER breaks a run — it just omits the section.

using GPUDiagnostics

# Static device snapshot + reduced sampler stats → the manifest's [gpu] table (a plain Dict
# that RunManifests writes verbatim as a top-level section). `n_threads` = the pixel-parallel
# launch size (Nx·Ny) for thread-fill occupancy. Stats reduce over ALL devices' rows
# (device_count records the fan-out; the per-device time series lives in the gputrace TSV);
# NaN entries (counters a device doesn't expose) are skipped per column. Returns `nothing` if
# telemetry is unavailable (e.g. no vendor extension) so the caller just omits [gpu].
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
            "thread_fill_occupancy" => Float64(thread_fill_occupancy(backend, n_threads)),
        )
        if telem.ticks > 0
            gpu["samples"] = telem.ticks
            gpu["sample_dt"] = telem.dt
            gpu["sampler_starved"] = telem.starved
            col(i) = Float64[s[i] for s in telem.samples if !isnan(s[i])]
            for (key, i) in (("power", 3), ("compute_util", 4), ("memory_util", 5))
                v = col(i)
                isempty(v) && continue
                gpu[key * "_mean"] = sum(v) / length(v)
                gpu[key * "_peak"] = maximum(v)
            end
            vr = col(6)
            isempty(vr) || (gpu["vram_used_peak"] = maximum(vr))
        end
        return gpu
    catch err
        @warn "GPU telemetry unavailable — omitting [gpu] from the manifest" exception = err
        return nothing
    end
end

# ── GPM counters → [gpu].gpm_* ────────────────────────────────────────────────────────────────
#
# `with_gpm_sampler` (lib/GPUDiagnostics) runs a second child beside the telemetry sampler on GPUs
# with GPU Performance Monitoring counters (NVIDIA Hopper and newer; elsewhere it is a no-op and
# this adds nothing). Its rows carry what the busy percentage cannot: ACHIEVED SM
# occupancy (to hold against the compile-time `kernel_occupancy`), FP64 / FP32 / tensor pipe
# utilization and DRAM-bandwidth utilization. `gpm_stats` reduces them over all devices' rows;
# every `<metric>_mean` / `_peak` lands as `gpm_<metric>_mean` / `_peak`, and the `_busy_mean`
# variants (rows with sm_util ≥ 0.5 — the kernel-active part of the field window, without the
# JIT / upload / drain idle diluting them) as `gpm_<metric>_busy_mean`, plus `gpm_samples`,
# `gpm_busy_samples`, `gpm_sample_dt` and `gpm_first_sample_s` (the child's startup lag). The
# per-tick series is the `gpmtrace_<tag>.tsv` beside the gputrace. Same contract as the other
# helpers: a failure logs and leaves the manifest without the fields.
function gpm_manifest_section!(gpu, telem)
    gpu === nothing && return nothing
    try
        telem.ticks > 0 || return nothing
        gpu["gpm_samples"] = telem.ticks
        gpu["gpm_sample_dt"] = telem.dt
        gpu["gpm_first_sample_s"] = telem.first_sample_s
        gpu["gpm_sampler_starved"] = telem.starved
        st = gpm_stats(telem)
        for (k, v) in st
            gpu["gpm_" * k] = k == "busy_samples" ? Int(v) : v
        end
        @info "GPM counters (field window)" samples = telem.ticks busy_samples = get(st, "busy_samples", 0) sm_occupancy_busy_mean = get(st, "sm_occupancy_busy_mean", NaN) fp64_util_busy_mean = get(st, "fp64_util_busy_mean", NaN) dram_bw_util_busy_mean = get(st, "dram_bw_util_busy_mean", NaN) sm_util_mean = get(st, "sm_util_mean", NaN)
        return st
    catch err
        @warn "GPM reduction failed — omitting [gpu].gpm_* from the manifest" exception = err
        return nothing
    end
end

# ── Device-event kernel timing → [timing].kernel + [gpu].kernel_* + the kerneltimes TSV ──────────
#
# `LaunchTimer` (lib/GPUDiagnostics) records a device-event pair per kernel launch. It is the only
# kernel clock that works with the production `sync_per_electron = false`: [timing].field is the
# whole phase (first-launch JIT, per-electron spline conversion + upload, the drain, the sampler's
# stop wait, and the slowest shard), while [timing].kernel is the BUSIEST device's summed kernel
# seconds — the denominator for achieved FLOP/s and the clean multi-device scaling metric
# (scaling_report.jl prefers it over the ≥ 99 %-utilization gputrace proxy). Per-device sums,
# launch counts, first-launch (JIT) and median/max per-launch seconds go to [gpu]; the full
# per-launch series to `kerneltimes_<tag>.tsv` (device, launch, seconds). Same contract as the
# other helpers: a failure logs and leaves the manifest without the fields.
function record_kernel_timing!(timing::AbstractDict, gpu, timer; tracefile = nothing)
    try
        lt = launch_times(timer)
        isempty(lt) && return nothing
        devs = sort!(collect(keys(lt)))
        per = [lt[d] for d in devs]
        med(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
        timing["kernel"] = maximum(sum, per)
        if gpu !== nothing
            gpu["kernel_devices"] = devs
            gpu["kernel_s"] = map(sum, per)
            gpu["kernel_launches"] = map(length, per)
            gpu["kernel_first_s"] = map(first, per)
            gpu["kernel_median_s"] = map(med, per)
            gpu["kernel_max_s"] = map(maximum, per)
        end
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
# `compiled_kernels` + `kernel_resources` (lib/GPUDiagnostics) read it back from there — the exact
# code that ran, not a re-creation of it, and no kernel change — and report what the compiler gave
# it (registers per thread, spill/stack bytes, the shared/LDS bytes the descriptor reserves) with
# the theoretical occupancy the runtime derives at the launched block size (the kernel's static
# 256-thread AcceleratedKernels workgroup, read from its signature). The kernels are closures inside
# the `accumulate_*` drivers; on Julia ≥ 1.12 their types carry the driver's name, which `pattern`
# selects (`kernel_driver` records the match). Multi-device runs compile one instance per device —
# the first is reported, `kernel_compiled_matches` keeps the count. AMD ISA figures (SGPR/VGPR/
# spill counts, the compiler's own waves-per-SIMD estimate) flatten to `kernel_isa_*`. Same
# contract as the other helpers: a failure logs and leaves the manifest without the fields.
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
        for k in (:block_size, :registers, :local_mem_bytes, :shared_mem_bytes, :const_mem_bytes,
                :max_threads_per_block, :active_blocks_per_sm, :active_warps_per_sm, :max_warps_per_sm,
                :warp_size, :max_threads_per_sm, :shared_mem_per_sm)
            gpu["kernel_" * String(k)] = Int(getfield(r, k))
        end
        gpu["kernel_occupancy"] = Float64(r.occupancy)
        for (k, v) in r.isa
            gpu["kernel_isa_" * k] = v
        end
        @info "kernel resources (compile time)" kernel = gpu["kernel_driver"] registers = r.registers local_mem_bytes = r.local_mem_bytes shared_mem_bytes = r.shared_mem_bytes block_size = r.block_size active_blocks_per_sm = r.active_blocks_per_sm occupancy = round(r.occupancy; digits = 3) isa_info = r.isa
        return r
    catch err
        @warn "kernel resource report unavailable — omitting [gpu].kernel_registers/…" exception = (err, catch_backtrace())
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
            Float64(gpu_peak_fp64_flops(backend))
        catch err
            @warn "FP64 peak measurement failed — omitting peak_fp64_flops / peak_fraction_field" exception = (err, catch_backtrace())
            NaN
        end
        if isfinite(peak) && peak > 0
            f["peak_fp64_flops"] = peak
            f["peak_fp64_method"] = backend isa ElectronDynamicsModels.KernelAbstractions.CPU ? "blas-peakflops" : "fma-chain-measured"
            f["peak_fraction_field"] = rate / peak
        end
        return f
    catch err
        @warn "FLOP accounting unavailable — omitting [flops] from the manifest" exception = err
        return nothing
    end
end
