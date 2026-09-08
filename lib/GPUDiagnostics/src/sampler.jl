# Out-of-process GPU telemetry: `with_gpu_sampler` records power / compute-util / mem-util /
# VRAM per device while a function runs and streams the time series to a TSV. A telemetry
# hiccup never breaks the caller — the function still runs, the telemetry is just empty.
#
# The sampler is a CHILD PROCESS (gpu_telemetry_child_cmd → bin/gputrace{,_cuda}.sh), not
# a Julia task. Two in-process designs failed on a W7900 host:
#   * ticking through the vendor runtime (hipGetDeviceProperties/hipMemGetInfo) wedges behind
#     a kernel stream backed up with queued launches — hour-long runs recorded samples=2
#     (spawn + teardown) and all-zero utilization stats;
#   * even a runtime-free sysfs tick is suspended wholesale: Julia's GC/libuv-timer coupling
#     stops sleeping tasks while the host thread allocates (0 ticks/15 s under pure-CPU alloc
#     churn regardless of thread count; 1 tick/98.5 s over a real hour-scale kernel loop).
# The child shares nothing with this process, so neither failure mode applies. It appends
# rows to the TSV as it samples (the trace survives a mid-run crash) and stops cooperatively
# when a stopfile appears — or on its own if this process dies, so it cannot be orphaned.

"""
    with_gpu_sampler(f, backend, dt; devices = 1:1, tracefile = nothing) -> (f(), telemetry)

Run `f()` while a child process samples `devices` (1-based vendor ids) every `dt` seconds.
`f` is first so the do-block form works. Without `tracefile`, samples go to a temp file that is
deleted after parsing. `telemetry` is a NamedTuple:

- `samples :: Vector{NTuple{6,Float64}}` — rows `(t_rel_s, device, power_W, compute_util, mem_util, vram_used_B)`
- `ticks :: Int` — sample rounds (rows of the first device)
- `dt, window :: Float64` — requested cadence / sampled window, seconds
- `starved :: Bool` — ticks ≪ window/dt ⇒ stats unreliable (also warned)
- `trace :: Union{String,Nothing}` — the TSV (absolute epoch timestamps), or `nothing`

If no sampler child can be built for `backend` (no vendor extension, e.g. the CPU backend),
`f` runs without one and the telemetry is empty.
"""
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
        # `nan` marks a counter the device does not expose (e.g. no mem_busy_percent): keep the row.
        (0 <= vals[4] <= 1 && (isnan(vals[5]) || 0 <= vals[5] <= 1)) || continue
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

