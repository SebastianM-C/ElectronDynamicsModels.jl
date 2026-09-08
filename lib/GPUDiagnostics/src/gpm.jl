# GPU Performance Monitoring (GPM) sampler: `with_gpm_sampler` records ACHIEVED SM occupancy and
# per-pipe utilization (FP64 / FP32 / FP16 / tensor / integer), DRAM-bandwidth utilization and
# PCIe / NVLink traffic per device while a function runs, out of process, into a TSV — the
# hardware-counter complement of `with_gpu_sampler` (which sees only coarse busy percentages).
#
# GPM is NVIDIA's counter aggregation for Hopper and newer GPUs (H100 / H200 / GH200 / B200; with
# recent drivers also consumer Blackwell, e.g. an RTX 5090 on driver 580 — no admin/profiling
# privileges needed). NVML exposes it as "take two samples, ask for the metrics over the interval".
# Pre-Hopper boards and every AMD GPU have no GPM: there `gpu_gpm_supported` is `false` and the
# function simply runs without a sampler.
#
# Like the telemetry sampler, GPM sampling runs in a CHILD PROCESS (`gpu_gpm_child_cmd`, a Julia
# child, bin/gpmtrace.jl): an in-process Julia task is suspended by the GC/timer coupling while the
# solver allocates, and `nvidia-smi` has no GPM query at all — so the child loads the vendor
# package for its NVML bindings only (never a CUDA context) and shares nothing with the parent.
# The child writes its own column header (`# epoch_s  device  <metric columns…>`) and one row per
# device per tick; the parent parses whatever columns the child declares, so the metric set is
# owned by the child alone. Stop is cooperative (stopfile) or automatic when the parent dies.

"""    gpu_gpm_supported(backend, device_ids = 1:1) -> Bool

`true` when every device in `device_ids` (1-based vendor ids) supports GPU Performance
Monitoring counters and the vendor extension can build a sampler child for them. `false` on
the CPU backend, on backends without a vendor extension, and on devices without GPM (NVIDIA
pre-Hopper boards, all AMD GPUs) — never throws."""
gpu_gpm_supported(::Backend, ::AbstractVector{<:Integer} = 1:1) = false

"""    gpu_gpm_child_cmd(backend, device_ids, dt, stopfile) -> Cmd

Build the out-of-process GPM sampler command for the (1-based) `device_ids`. The child prints
one comment header naming its columns (`# epoch_s  device  sm_util  sm_occupancy  …`) and then
one TSV row per device every `dt` seconds on stdout — utilizations as fractions in [0, 1],
`nan` for a metric the device does not report — and exits when `stopfile` appears or the
parent dies. Vendor extensions implement it; see bin/gpmtrace.jl for the NVIDIA child."""
gpu_gpm_child_cmd(b::Backend, ::AbstractVector{<:Integer}, ::Real, ::AbstractString) = error(
    "gpu_gpm_child_cmd: no GPU vendor extension with GPM support loaded for ", typeof(b),
    " — load CUDA.jl"
)

const _EMPTY_GPM_COLUMNS = [:t_rel_s, :device]

_empty_gpm_telemetry(dt, supported) = (
    columns = copy(_EMPTY_GPM_COLUMNS), samples = zeros(0, 2), ticks = 0, dt = Float64(dt),
    window = 0.0, first_sample_s = NaN, starved = false, trace = nothing, supported = supported,
)

"""
    with_gpm_sampler(f, backend, dt; devices = 1:1, tracefile = nothing) -> (f(), telemetry)

Run `f()` while a child process samples GPU Performance Monitoring counters of `devices`
(1-based vendor ids) every `dt` seconds. `f` is first so the do-block form works. Without
`tracefile`, samples go to a temp file that is deleted after parsing. `telemetry` is a NamedTuple:

- `columns :: Vector{Symbol}` — `[:t_rel_s, :device, metric columns…]` as declared by the child
  (NVIDIA: `sm_util, sm_occupancy, fp64_util, dram_bw_util, fp32_util, fp16_util, tensor_util,
  int_util, pcie_tx_MiBps, pcie_rx_MiBps, nvlink_rx_MiBps, nvlink_tx_MiBps`)
- `samples :: Matrix{Float64}` — one row per device per tick, columns as above (`NaN` where a
  metric is not reported); utilizations and occupancy are fractions in [0, 1]
- `ticks :: Int` — sample rounds (rows of the first device)
- `dt, window :: Float64` — requested cadence / sampled window, seconds
- `first_sample_s :: Float64` — seconds from the call to the first row (the child's startup)
- `starved :: Bool` — ticks ≪ window/dt ⇒ stats unreliable (also warned)
- `trace :: Union{String,Nothing}` — the TSV (absolute epoch timestamps), or `nothing`
- `supported :: Bool` — whether a sampler was attempted at all

When [`gpu_gpm_supported`](@ref) is `false` for `backend`/`devices` (CPU backend, no vendor
extension, a GPU without GPM), `f` runs without a sampler and the telemetry is empty. Reduce
with [`gpm_stats`](@ref) or pick a column with [`gpm_column`](@ref).
"""
function with_gpm_sampler(f, backend, dt::Real;
        devices::AbstractVector{<:Integer} = 1:1, tracefile::Union{String, Nothing} = nothing)
    supported = try
        gpu_gpm_supported(backend, devices)
    catch err
        @warn "GPM support query failed — running without the GPM sampler" exception = err
        false
    end
    if !supported
        backend isa KA.CPU || @info "GPM counters not available on this device — running without the GPM sampler"
        return f(), _empty_gpm_telemetry(dt, false)
    end

    trace = something(tracefile, tempname() * ".tsv")
    stopfile = trace * ".stop"
    t0 = time()
    child = try
        cmd = gpu_gpm_child_cmd(backend, devices, dt, stopfile)
        open(io -> nothing, trace, "w")   # the child appends (its header first); truncate any stale file
        run(pipeline(pipeline(cmd; stderr = devnull); stdout = trace, append = true); wait = false)
    catch err
        @warn "GPM sampler unavailable — running without it" exception = err
        nothing
    end
    if child === nothing
        tracefile === nothing && rm(trace; force = true)
        return f(), _empty_gpm_telemetry(dt, true)
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

    columns, rows = _parse_gpm_trace(trace, t0)
    if isempty(rows)
        # A trace without a single row carries nothing worth keeping beside the run outputs.
        @warn "GPM sampler produced no samples over $(round(window; digits = 1)) s (child startup failure? see the trace's stderr)" trace = tracefile
        rm(trace; force = true)
        t = _empty_gpm_telemetry(dt, true)
        return result, merge(t, (columns = columns, samples = zeros(0, length(columns)), window = window))
    end
    tracefile === nothing && rm(trace; force = true)

    samples = permutedims(reduce(hcat, rows))
    ticks = count(==(samples[1, 2]), @view samples[:, 2])
    first_sample_s = minimum(@view samples[:, 1])
    # Starvation watchdog over the window the child was actually sampling: a Julia child needs a
    # few seconds to load its NVML bindings (first_sample_s), which must not count as missed ticks.
    sampled = window - first_sample_s
    starved = sampled > 10 * dt && ticks < 0.5 * sampled / dt
    starved &&
        @warn "GPM sampler starved: $ticks ticks over $(round(sampled; digits = 1)) s at dt=$(dt) s — GPM stats are unreliable"
    return result, (columns = columns, samples = samples, ticks = ticks, dt = Float64(dt),
        window = window, first_sample_s = first_sample_s, starved = starved,
        trace = tracefile, supported = true)
end

# Parse a child trace: the first `# epoch_s\tdevice\t…` comment line names the columns; every data
# row must have exactly that many fields, an epoch inside the sampled window (±5 s) and fraction
# columns (`*_util`, `sm_occupancy`) in [0, 1] (NaN allowed) — a torn or garbled row is dropped.
function _parse_gpm_trace(trace::AbstractString, t0::Real)
    columns = copy(_EMPTY_GPM_COLUMNS)
    rows = Vector{Float64}[]
    isfile(trace) || return columns, rows
    have_header = false
    for line in eachline(trace)
        if startswith(line, '#')
            have_header && continue
            hdr = split(strip(lstrip(line, '#')), '\t')
            if length(hdr) >= 3 && hdr[1] == "epoch_s" && hdr[2] == "device"
                columns = vcat([:t_rel_s, :device], Symbol.(strip.(hdr[3:end])))
                have_header = true
            end
            continue
        end
        have_header || continue
        parts = split(line, '\t')
        length(parts) == length(columns) || continue
        vals = map(x -> tryparse(Float64, x), parts)
        any(isnothing, vals) && continue
        (t0 - 5 <= vals[1] <= time() + 5) || continue
        vals[2] >= 1 || continue
        ok = true
        for (j, c) in enumerate(columns)
            j <= 2 && continue
            if _is_fraction_column(c)
                v = vals[j]
                (isnan(v) || 0 <= v <= 1) || (ok = false; break)
            end
        end
        ok || continue
        vals[1] -= t0
        push!(rows, vals)
    end
    return columns, rows
end

_is_fraction_column(c::Symbol) = (s = String(c); endswith(s, "_util") || s == "sm_occupancy")

"""    gpm_column(telemetry, name::Symbol) -> Vector{Float64}

The column `name` of a [`with_gpm_sampler`](@ref) telemetry over all devices' rows (`NaN`
entries included); empty when the column is absent."""
function gpm_column(telem, name::Symbol)
    j = findfirst(==(name), telem.columns)
    j === nothing && return Float64[]
    return Float64.(@view telem.samples[:, j])
end

"""    gpm_stats(telemetry; busy_column = :sm_util, busy_threshold = 0.5) -> Dict{String, Float64}

Reduce a [`with_gpm_sampler`](@ref) telemetry over all devices' rows, skipping `NaN` entries
per column: for every metric column `<name>_mean` and `<name>_peak`, plus `<name>_busy_mean`
over the rows whose `busy_column` is ≥ `busy_threshold` (the kernel-active part of the window,
so achieved occupancy and pipe utilization can be compared with a kernel's theoretical
occupancy without the idle JIT/upload/drain phases diluting them) and `busy_samples`. Empty
when the telemetry has no rows."""
function gpm_stats(telem; busy_column::Symbol = :sm_util, busy_threshold::Real = 0.5)
    out = Dict{String, Float64}()
    size(telem.samples, 1) == 0 && return out
    busy = gpm_column(telem, busy_column)
    busy_mask = isempty(busy) ? falses(size(telem.samples, 1)) : map(v -> !isnan(v) && v >= busy_threshold, busy)
    out["busy_samples"] = count(busy_mask)
    for (j, c) in enumerate(telem.columns)
        j <= 2 && continue
        col = @view telem.samples[:, j]
        v = Float64[x for x in col if !isnan(x)]
        isempty(v) && continue
        out[String(c) * "_mean"] = sum(v) / length(v)
        out[String(c) * "_peak"] = maximum(v)
        vb = Float64[x for (x, m) in zip(col, busy_mask) if m && !isnan(x)]
        isempty(vb) || (out[String(c) * "_busy_mean"] = sum(vb) / length(vb))
    end
    return out
end
