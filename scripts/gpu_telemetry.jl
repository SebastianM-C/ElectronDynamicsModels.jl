# GPU telemetry for the solver manifests' [gpu] section — shared by thomson_scattering.jl,
# lpwa.jl and occupancy_bench.jl. A static device snapshot (name, SMs, capacity, VRAM,
# thread-fill occupancy) plus a sampler that records power / compute-util / mem-util / VRAM
# per device over the accumulate_field window and streams the time series to a gputrace TSV.
# Everything is wrapped so a telemetry hiccup NEVER breaks a run — it just omits the section.
#
# The sampler is a CHILD PROCESS (gpu_telemetry_child_cmd → scripts/gputrace{,_cuda}.sh), not
# a Julia task. Two in-process designs failed on the production W7900 host:
#   * ticking through the vendor runtime (hipGetDeviceProperties/hipMemGetInfo) wedges behind
#     a kernel stream backed up with queued launches — hour-long runs recorded samples=2
#     (spawn + teardown) and all-zero utilization stats;
#   * even a runtime-free sysfs tick is suspended wholesale: Julia's GC/libuv-timer coupling
#     stops sleeping tasks while the host thread allocates (0 ticks/15 s under pure-CPU alloc
#     churn regardless of thread count; 1 tick/98.5 s over a real accumulate_field window).
# The child shares nothing with this process, so neither failure mode applies. It appends
# rows to the TSV as it samples (the trace survives a mid-run crash) and stops cooperatively
# when a stopfile appears — or on its own if this process dies, so it cannot be orphaned.

# Run `f()` while a child process samples `devices` (1-based vendor ids) every `dt` seconds.
# `f` is FIRST so the do-block form works: `with_gpu_sampler(backend, dt; kw...) do … end`.
# Without `tracefile`, samples go to a temp file that is deleted after parsing.
# Returns (result_of_f, telemetry) with telemetry a NamedTuple:
#   samples :: Vector{NTuple{6,Float64}} — rows (t_rel_s, device, power_W, compute_util, mem_util, vram_used_B)
#   ticks   :: Int                       — sample rounds (= rows of the first device)
#   dt, window :: Float64                — requested cadence / sampled window, seconds
#   starved :: Bool                      — ticks ≪ window/dt ⇒ stats unreliable (also @warn'ed)
#   trace   :: Union{String,Nothing}     — the TSV (absolute epoch timestamps), or nothing
function with_gpu_sampler(f, backend, dt::Real;
        devices::AbstractVector{<:Integer} = 1:1, tracefile::Union{String, Nothing} = nothing)
    trace = something(tracefile, tempname() * ".tsv")
    stopfile = trace * ".stop"
    t0 = time()
    child = try
        cmd = gpu_telemetry_child_cmd(backend, devices, dt, stopfile)
        open(io -> println(io, "# epoch_s\tdevice\tpower_W\tcompute_util\tmem_util\tvram_used_B"),
            trace, "w")
        # append applies to every file redirect in one pipeline() call, so silence stderr in an
        # inner pipeline and append stdout to the trace in the outer one.
        run(pipeline(pipeline(cmd; stderr = devnull); stdout = trace, append = true); wait = false)
    catch err
        @warn "GPU telemetry unavailable — running without the sampler" exception = err
        nothing
    end
    if child === nothing
        return f(), (samples = NTuple{6, Float64}[], ticks = 0, dt = Float64(dt),
            window = 0.0, starved = false, trace = nothing)
    end

    local result
    try
        result = f()
    finally
        touch(stopfile)
        deadline = time() + 2 + 2dt   # child polls the stopfile once per tick
        while process_running(child) && time() < deadline
            sleep(0.1)
        end
        process_running(child) && kill(child)
        wait(child)
        rm(stopfile; force = true)
    end
    window = time() - t0

    samples = NTuple{6, Float64}[]
    for line in eachline(trace)
        startswith(line, '#') && continue
        parts = split(line, '\t')
        length(parts) == 6 || continue
        vals = map(x -> tryparse(Float64, x), parts)
        any(isnothing, vals) && continue
        # Plausibility gate: a torn row (two writers interleaving on the trace) can still parse as
        # six numbers — a VRAM value glued to the next row's epoch once reached the manifest as
        # a 3e20 B peak. Epoch inside the sampled window (±5 s), utilizations in [0, 1], power
        # and VRAM non-negative and below any real board (5 kW, 1 TB).
        (t0 - 5 <= vals[1] <= time() + 5) || continue
        (0 <= vals[4] <= 1 && 0 <= vals[5] <= 1) || continue
        (0 <= vals[3] <= 5000 && 0 <= vals[6] <= 1.0e12) || continue
        push!(samples, (vals[1] - t0, vals[2], vals[3], vals[4], vals[5], vals[6]))
    end
    tracefile === nothing && rm(trace; force = true)

    ticks = isempty(samples) ? 0 : count(s -> s[2] == samples[1][2], samples)
    # Watchdog: the out-of-process child should make starvation impossible — but if it ever
    # recurs, say so loudly and mark the manifest instead of shipping silent zeros again.
    starved = window > 10 * dt && ticks < 0.5 * window / dt
    starved &&
        @warn "GPU telemetry sampler starved: $ticks ticks over $(round(window; digits = 1)) s at dt=$(dt) s — [gpu] sample stats are unreliable"
    return result, (samples = samples, ticks = ticks, dt = Float64(dt), window = window,
        starved = starved, trace = tracefile)
end

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
