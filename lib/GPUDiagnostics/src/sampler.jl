# Out-of-process GPU telemetry: `with_gpu_sampler` records per-device power / utilization /
# VRAM — and, where the device has them, hardware counters (NVIDIA GPM: achieved SM occupancy,
# per-pipe utilization, DRAM bandwidth) — while a function runs, streaming the time series to a
# TSV. A telemetry hiccup never breaks the caller: the function still runs, the telemetry is empty.
#
# One sampling function, `gpu_sample(backend, device)`, returns a NamedTuple with whatever the
# vendor exposes. It is usable in-process for a one-shot snapshot and is exactly what the child
# calls in its loop. Behind it sit small per-device SOURCE objects built from a string spec the
# vendor extension resolves ONCE in the parent (`gpu_sampler_sources`): NVIDIA hands the child an
# NVML uuid (the child loads CUDA.jl for its NVML bindings only — no CUDA context — because
# `nvidia-smi` has no GPM query); AMD hands it the amdgpu driver's sysfs paths, so the child reads
# them directly and never loads AMDGPU.jl.
#
# The sampler is a CHILD PROCESS (`telemetry_child_main`, a Julia process started with this
# process's julia binary and load path), not a Julia task. Two in-process designs failed on a
# W7900 host:
#   * ticking through the vendor runtime (hipGetDeviceProperties/hipMemGetInfo) wedges behind
#     a kernel stream backed up with queued launches — hour-long runs recorded samples=2
#     (spawn + teardown) and all-zero utilization stats;
#   * even a runtime-free sysfs tick is suspended wholesale: Julia's GC/libuv-timer coupling
#     stops sleeping tasks while the host thread allocates (0 ticks/15 s under pure-CPU alloc
#     churn regardless of thread count; 1 tick/98.5 s over a real hour-scale kernel loop).
# The child shares nothing with this process, so neither failure mode applies. It appends rows
# to the TSV as it samples (the trace survives a mid-run crash), declares its own column header
# (the parent parses whatever columns the child emits), and stops cooperatively when a stopfile
# appears — or on its own if this process dies, so it cannot be orphaned.

# ── Sources ──────────────────────────────────────────────────────────────────────────────────

"""    SamplerSource

Per-device sampling state. Subtypes implement `sample!(source) -> NamedTuple` (metric name ⇒
`Float64`, `NaN` for a metric the device does not report) and carry a `device::Int` field (the
1-based vendor ordinal); `close!(source)` releases resources (default no-op). Sources are built
from string specs by [`sampler_source`](@ref): `<kind>:<payload>`, dispatched on `Val(kind)`."""
abstract type SamplerSource end

function sample! end
close!(::SamplerSource) = nothing
device_id(s::SamplerSource) = s.device

"""    sampler_source(spec::AbstractString, counters::Symbol) -> SamplerSource

Build the source described by `spec` (`"<kind>:<payload>"`). Kinds shipped here: `sysfs`
(amdgpu driver files: `sysfs:<device>:<power_file>:<busy_file>:<membusy_file|->:<vram_file>`)
and `synthetic` (`synthetic:<device>`, deterministic fake values for tests); the CUDA extension
adds `nvml` (`nvml:GPU-<uuid>=<device>`). `counters` is `:auto` (hardware counters when the
device has them) or `:none`. Unknown kinds throw."""
function sampler_source(spec::AbstractString, counters::Symbol)
    kind, rest = split(spec, ':'; limit = 2)
    return _sampler_source(Val(Symbol(kind)), String(rest), counters)
end
_sampler_source(::Val{K}, rest, counters) where {K} =
    error("sampler_source: unknown source kind '$K' — is the vendor extension loaded in this process?")

# amdgpu driver sysfs. Values: power in µW, busy percentages, VRAM in bytes.
struct SysfsSource <: SamplerSource
    device::Int
    power::String
    busy::String
    membusy::Union{String, Nothing}
    vram::String
end
function _sampler_source(::Val{:sysfs}, rest::AbstractString, ::Symbol)
    f = split(rest, ':')
    length(f) == 5 || throw(ArgumentError("sysfs source spec needs 5 fields, got $(length(f)): $rest"))
    return SysfsSource(parse(Int, f[1]), String(f[2]), String(f[3]), f[4] == "-" ? nothing : String(f[4]), String(f[5]))
end
function _read_number(path)
    try
        v = tryparse(Float64, strip(read(path, String)))
        return v === nothing ? NaN : v
    catch
        return NaN   # transient read failure → nan for this tick
    end
end
sample!(s::SysfsSource) = (
    power_W = _read_number(s.power) / 1.0e6,
    compute_util = _read_number(s.busy) / 100,
    mem_util = s.membusy === nothing ? NaN : _read_number(s.membusy) / 100,
    vram_used_B = _read_number(s.vram),
)

# Deterministic fake device for tests of the child protocol (and a handy `gpu_sample` stand-in on
# the CPU backend): device 1 is "busy", device 2 idle with a counter it does not expose.
struct SyntheticSource <: SamplerSource
    device::Int
    counters::Bool
end
_sampler_source(::Val{:synthetic}, rest::AbstractString, counters::Symbol) =
    SyntheticSource(parse(Int, rest), counters != :none)
function sample!(s::SyntheticSource)
    busy = s.device == 1
    base = (power_W = 100.0 + 50 * (s.device - 1), compute_util = busy ? 0.9 : 0.1,
        mem_util = s.device == 2 ? NaN : 0.5, vram_used_B = 1000.0 * s.device)
    s.counters || return base
    return merge(base, (sm_util = busy ? 0.9 : 0.1, sm_occupancy = busy ? 0.3 : 0.1, fp64_util = busy ? 0.8 : NaN))
end

"""    gpu_sampler_sources(backend, device_ids, counters) -> (specs, packages)

Vendor hook (implemented by the CUDA.jl / AMDGPU.jl extensions): resolve the (1-based)
`device_ids` into source specs a child process can open WITHOUT the vendor runtime — NVML uuids
on NVIDIA (`nvml:GPU-<uuid>=<device>`; the child loads the listed `packages`, i.e. CUDA.jl, for
the NVML bindings only), amdgpu sysfs paths on AMD (`sysfs:…`, no packages). `counters` is
`:auto` or `:none`. The vendor runtime is touched only here, in the parent."""
gpu_sampler_sources(b::Backend, ::AbstractVector{<:Integer}, ::Symbol) = error(
    "gpu_sampler_sources: no GPU vendor extension loaded for ", typeof(b), " — load CUDA.jl or AMDGPU.jl"
)

const _SOURCE_CACHE = Dict{Tuple{DataType, Int, Symbol}, SamplerSource}()
const _SOURCE_LOCK = ReentrantLock()

"""    gpu_sample(backend, device = gpu_device(backend); counters = :auto) -> NamedTuple

One telemetry sample of `device` (1-based vendor ordinal), in-process: `power_W`,
`compute_util`, `mem_util`, `vram_used_B` (fractions in [0, 1]; `NaN` when the device does not
expose a counter) and, on NVIDIA GPUs with GPM (Hopper and newer) when `counters = :auto`,
`sm_util`, `sm_occupancy` (ACHIEVED), `fp64_util`, `dram_bw_util`, `fp32_util`, `fp16_util`,
`tensor_util`, `int_util`, `pcie_tx_MiBps`, `pcie_rx_MiBps`, `nvlink_rx_MiBps`, `nvlink_tx_MiBps`.
Interval metrics (GPM) cover the time since the previous `gpu_sample` of that device (the first
call takes a 100 ms interval). This is exactly what the sampler child calls per tick; for a time
series while a kernel runs use [`with_gpu_sampler`](@ref) — an in-process loop is suspended by
Julia's GC/timer coupling while the host allocates."""
function gpu_sample(backend::Backend, device::Integer = gpu_device(backend); counters::Symbol = :auto)
    _check_counters(counters)
    key = (typeof(backend), Int(device), counters)
    src = lock(_SOURCE_LOCK) do
        get!(_SOURCE_CACHE, key) do
            s = gpu_sampler_sources(backend, [Int(device)], counters)
            sampler_source(only(s.specs), counters)
        end
    end
    return sample!(src)
end

_check_counters(c::Symbol) = c in (:auto, :none) || throw(ArgumentError("counters must be :auto or :none, got :$c"))

# ── The child ────────────────────────────────────────────────────────────────────────────────

"""    telemetry_child_main(args) -> Int

Entry point of the sampler child (a separate Julia process started by [`with_gpu_sampler`](@ref)).
`args`: `--dt=<s> --ppid=<pid> --stop=<stopfile> --counters=<auto|none> <spec>...`. Opens one
[`sampler_source`](@ref) per spec, prints a header comment naming the columns
(`# epoch_s  device  <metric…>` — the union of the sources' first samples, base columns first),
then one TSV row per device every `dt` seconds on stdout (`nan` for a missing metric), and returns
when the stopfile appears or the parent dies. Never loads a vendor runtime beyond what the parent
asked for (`packages`), so it starts in about a second."""
function telemetry_child_main(args::AbstractVector{<:AbstractString})
    o = _parse_child_args(args)
    sources = SamplerSource[]
    for spec in o.specs
        try
            push!(sources, sampler_source(spec, o.counters))
        catch err
            println(stderr, "telemetry child: cannot open source '$spec': ", sprint(showerror, err))
        end
    end
    if isempty(sources)
        println(stderr, "telemetry child: no source could be opened — exiting")
        return 1
    end
    columns = Symbol[]
    for s in sources, k in keys(sample!(s))   # first samples: declares the columns, primes interval counters
        k in columns || push!(columns, k)
    end
    println(stdout, "# epoch_s\tdevice\t", join(String.(columns), '\t'))
    flush(stdout)
    io = IOBuffer()
    while !(isfile(o.stopfile) || !_parent_alive(o.ppid))
        sleep(o.dt)
        now = time()
        for s in sources
            nt = try
                sample!(s)
            catch err
                println(stderr, "telemetry child: sample of device $(device_id(s)) failed: ", sprint(showerror, err))
                continue
            end
            print(io, _fixed(now, 3), '\t', device_id(s))
            for c in columns
                v = Float64(get(nt, c, NaN))
                print(io, '\t', _fmt_value(v))
            end
            println(io)
        end
        write(stdout, take!(io))
        flush(stdout)
    end
    foreach(close!, sources)
    return 0
end

function _parse_child_args(args)
    dt = 1.0; ppid = 0; stopfile = ""; counters = :auto; specs = String[]
    for a in args
        if startswith(a, "--dt=")
            dt = parse(Float64, a[6:end])
        elseif startswith(a, "--ppid=")
            ppid = parse(Int, a[8:end])
        elseif startswith(a, "--stop=")
            stopfile = String(a[8:end])
        elseif startswith(a, "--counters=")
            counters = Symbol(a[12:end])
        else
            push!(specs, String(a))
        end
    end
    dt > 0 || throw(ArgumentError("telemetry child: --dt must be positive"))
    isempty(stopfile) && throw(ArgumentError("telemetry child: --stop=<file> is required"))
    _check_counters(counters)
    return (; dt, ppid, stopfile, counters, specs)
end

# `kill -0`: ESRCH ⇒ gone; EPERM ⇒ alive but not ours (still alive). ppid 0 ⇒ no watchdog.
function _parent_alive(ppid::Integer)
    ppid <= 0 && return true
    Sys.iswindows() && return true
    return ccall(:kill, Cint, (Cint, Cint), ppid, 0) == 0 || Libc.errno() == Libc.EPERM
end

# Row formatting without Printf (the lib must stay loadable from the tracked manifests): fixed
# decimals for the epoch, plain integers where the value is one (VRAM bytes), 6 significant
# digits otherwise.
_fmt_value(v::Float64) = isnan(v) ? "nan" :
    (isinteger(v) && abs(v) < 1.0e15) ? string(Int(v)) : repr(round(v; sigdigits = 6))
function _fixed(x::Float64, digits::Int)
    scale = 10^digits
    n = round(Int, x * scale)
    return string(n ÷ scale, '.', lpad(n % scale, digits, '0'))
end

# The child command: this process's julia, `--threads=1`, the same expanded LOAD_PATH (project
# stack incl. the default environment — so the child resolves GPUDiagnostics and the vendor
# package exactly as the parent did, even when they are indirect dependencies: `Base.require` by
# PkgId walks every manifest on the stack).
function telemetry_child_cmd(packages::AbstractVector{Base.PkgId}, dt::Real, stopfile::AbstractString,
        counters::Symbol, specs::AbstractVector{<:AbstractString})
    req(p) = "Base.require(Base.PkgId(Base.UUID(\"$(p.uuid)\"), \"$(p.name)\"))"
    code = join(vcat([req(p) * ";" for p in packages],
        ["GD = " * req(Base.PkgId(@__MODULE__)) * ";", "exit(GD.telemetry_child_main(ARGS))"]), " ")
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1 -e $code -- --dt=$(Float64(dt)) --ppid=$(getpid()) --stop=$stopfile --counters=$counters $specs`
    return addenv(cmd, "JULIA_LOAD_PATH" => join(Base.load_path(), Sys.iswindows() ? ';' : ':'))
end

# ── The parent: with_gpu_sampler + telemetry table ───────────────────────────────────────────

"""    GPUTelemetry

Column table returned by [`with_gpu_sampler`](@ref): `columns` (`[:t_rel_s, :device, metric
columns…]` as declared by the child), `samples::Matrix{Float64}` (one row per device per tick,
`NaN` where a metric is not reported), `ticks` (sample rounds = rows of the first device), `dt`
and `window` (requested cadence / sampled window, s), `first_sample_s` (child startup lag),
`starved` (ticks ≪ window/dt ⇒ stats unreliable), `trace` (the TSV, or `nothing`) and
`counters`. Index a column by symbol: `telem[:compute_util]`; `haskey(telem, :fp64_util)`."""
struct GPUTelemetry
    columns::Vector{Symbol}
    samples::Matrix{Float64}
    ticks::Int
    dt::Float64
    window::Float64
    first_sample_s::Float64
    starved::Bool
    trace::Union{String, Nothing}
    counters::Symbol
end
const _BASE_COLUMNS = [:t_rel_s, :device]
_empty_telemetry(dt, counters; columns = copy(_BASE_COLUMNS), window = 0.0) =
    GPUTelemetry(columns, zeros(0, length(columns)), 0, Float64(dt), window, NaN, false, nothing, counters)

function Base.getindex(t::GPUTelemetry, c::Symbol)
    j = findfirst(==(c), t.columns)
    j === nothing && throw(KeyError(c))
    return t.samples[:, j]
end
Base.haskey(t::GPUTelemetry, c::Symbol) = c in t.columns
Base.keys(t::GPUTelemetry) = t.columns
Base.length(t::GPUTelemetry) = size(t.samples, 1)
Base.show(io::IO, t::GPUTelemetry) = print(io, "GPUTelemetry(", length(t), " rows × ", length(t.columns),
    " columns, ticks = ", t.ticks, ", dt = ", t.dt, ", window = ", round(t.window; digits = 1), " s",
    t.starved ? ", STARVED" : "", ")")

"""
    with_gpu_sampler(f, backend, dt; devices = 1:1, tracefile = nothing, counters = :auto) -> (f(), telem)

Run `f()` while a child process samples `devices` (1-based vendor ids) every `dt` seconds:
power / compute / memory utilization / VRAM everywhere, plus hardware counters where the device
has them (`counters = :auto`; NVIDIA GPM on Hopper and newer — achieved SM occupancy, FP64 / FP32
/ tensor pipe and DRAM-bandwidth utilization, PCIe / NVLink traffic; `:none` skips them). `f` is
first so the do-block form works. Without `tracefile`, samples go to a temp file that is deleted
after parsing. `telem` is a [`GPUTelemetry`](@ref) column table; reduce it with
[`gpu_telemetry_stats`](@ref).

If no sampler child can be built for `backend` (no vendor extension, e.g. the CPU backend) or
the child fails to start, a warning is logged and `f` runs without one — telemetry is never
allowed to break the caller.
"""
function with_gpu_sampler(f, backend, dt::Real; devices::AbstractVector{<:Integer} = 1:1,
        tracefile::Union{String, Nothing} = nothing, counters::Symbol = :auto)
    _check_counters(counters)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    trace = something(tracefile, tempname() * ".tsv")
    stopfile = trace * ".stop"
    errfile = trace * ".stderr"
    t0 = time()
    child = try
        src = gpu_sampler_sources(backend, devices, counters)
        cmd = telemetry_child_cmd(src.packages, dt, stopfile, counters, src.specs)
        open(io -> nothing, trace, "w")   # the child appends (its header first); truncate any stale file
        run(pipeline(cmd; stdout = trace, stderr = errfile, append = true); wait = false)
    catch err
        @warn "GPU telemetry unavailable — running without the sampler" exception = err
        nothing
    end
    if child === nothing
        tracefile === nothing && rm(trace; force = true)
        rm(errfile; force = true)
        return f(), _empty_telemetry(dt, counters)
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

    columns, rows = _parse_trace(trace, t0)
    child_err = isfile(errfile) ? strip(read(errfile, String)) : ""
    rm(errfile; force = true)
    if isempty(rows)
        @warn "GPU telemetry sampler produced no samples over $(round(window; digits = 1)) s (child exit code $(child.exitcode)) — running stats unavailable" child_stderr = first(child_err, 2000)
        rm(trace; force = true)   # nothing worth keeping beside the run outputs
        return result, _empty_telemetry(dt, counters; columns, window)
    end
    isempty(child_err) || @debug "GPU telemetry child stderr" child_stderr = child_err
    tracefile === nothing && rm(trace; force = true)

    samples = permutedims(reduce(hcat, rows))
    ticks = count(==(samples[1, 2]), @view samples[:, 2])
    first_sample_s = minimum(@view samples[:, 1])
    # Starvation watchdog over the window the child was actually sampling: the child needs a
    # moment to start (first_sample_s; a few seconds when it loads CUDA.jl for NVML), which must
    # not count as missed ticks. The out-of-process child should make starvation impossible — if
    # it ever recurs, say so loudly and mark the manifest instead of shipping silent zeros.
    starved = _starved(window, first_sample_s, ticks, dt)
    starved &&
        @warn "GPU telemetry sampler starved: $ticks ticks over $(round(window - first_sample_s; digits = 1)) s at dt=$(dt) s — [gpu] sample stats are unreliable"
    return result, GPUTelemetry(columns, samples, ticks, Float64(dt), window, first_sample_s, starved, tracefile, counters)
end

_starved(window, first_sample_s, ticks, dt) = (s = window - first_sample_s; s > 10 * dt && ticks < 0.5 * s / dt)

# Parse a child trace: the first `# epoch_s\tdevice\t…` comment names the columns; every data row
# must have exactly that many fields and pass a plausibility gate — a torn row (two writers
# interleaving) can still parse as numbers; a VRAM value glued to the next row's epoch once reached
# a manifest as a 3e20 B peak. Epoch inside the sampled window (±5 s), device ≥ 1, fractions
# (`*_util`, `sm_occupancy`) in [0, 1], power below 5 kW, VRAM below 1 TB; `nan` (a counter the
# device does not expose) is kept anywhere but epoch/device.
function _parse_trace(trace::AbstractString, t0::Real)
    columns = copy(_BASE_COLUMNS)
    rows = Vector{Float64}[]
    isfile(trace) || return columns, rows
    have_header = false
    for line in eachline(trace)
        if startswith(line, '#')
            have_header && continue
            hdr = split(strip(lstrip(line, '#')), '\t')
            if length(hdr) >= 3 && hdr[1] == "epoch_s" && hdr[2] == "device"
                columns = vcat(copy(_BASE_COLUMNS), Symbol.(strip.(hdr[3:end])))
                have_header = true
            end
            continue
        end
        have_header || continue
        parts = split(line, '\t')
        length(parts) == length(columns) || continue
        vals = map(x -> tryparse(Float64, x), parts)
        any(isnothing, vals) && continue
        vals = Float64[v for v in vals]
        (t0 - 5 <= vals[1] <= time() + 5) || continue
        vals[2] >= 1 || continue
        _plausible_row(columns, vals) || continue
        vals[1] -= t0
        push!(rows, vals)
    end
    return columns, rows
end

function _plausible_row(columns, vals)
    for j in 3:length(columns)
        v = vals[j]
        isnan(v) && continue
        c = String(columns[j])
        if endswith(c, "_util") || c == "sm_occupancy"
            0 <= v <= 1 || return false
        elseif c == "power_W"
            0 <= v <= 5000 || return false
        elseif c == "vram_used_B"
            0 <= v <= 1.0e12 || return false
        end
    end
    return true
end

"""    gpu_telemetry_stats(telem; busy_column = :compute_util, busy_threshold = 0.5) -> Dict{String, Any}

Reduce a [`GPUTelemetry`](@ref) over all devices' rows, skipping `NaN` entries per column: for
every metric column `<col>_mean`, `<col>_peak` and `<col>_busy_mean` — the mean over rows whose
`busy_column` is ≥ `busy_threshold`, i.e. the kernel-active part of the window without the idle
JIT / upload / drain phases diluting it (the number to hold against a kernel's theoretical
occupancy) — plus `samples` (ticks) and `busy_samples` (rows). Empty when there are no rows."""
function gpu_telemetry_stats(t::GPUTelemetry; busy_column::Symbol = :compute_util, busy_threshold::Real = 0.5)
    out = Dict{String, Any}()
    n = size(t.samples, 1)
    n == 0 && return out
    out["samples"] = t.ticks
    busy = haskey(t, busy_column) ? map(v -> !isnan(v) && v >= busy_threshold, t[busy_column]) : falses(n)
    out["busy_samples"] = count(busy)
    for (j, c) in enumerate(t.columns)
        j <= 2 && continue
        col = @view t.samples[:, j]
        v = Float64[x for x in col if !isnan(x)]
        isempty(v) && continue
        out[String(c) * "_mean"] = sum(v) / length(v)
        out[String(c) * "_peak"] = maximum(v)
        vb = Float64[x for (x, m) in zip(col, busy) if m && !isnan(x)]
        isempty(vb) || (out[String(c) * "_busy_mean"] = sum(vb) / length(vb))
    end
    return out
end
