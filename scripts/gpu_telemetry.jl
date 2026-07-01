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
