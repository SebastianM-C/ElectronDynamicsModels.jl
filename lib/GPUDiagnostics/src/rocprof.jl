# AMD hardware counters through rocprofv3 (ROCm 6.2+ / 7.x). There is no in-process counter API
# AMDGPU.jl could call (nothing like NVIDIA's GPM), so the model is the profiler's: rocprofv3 wraps
# a WHOLE PROCESS —
#     rocprofv3 --kernel-trace --pmc <counters> --output-format csv -d <dir> -o <name> -- <command>
# — and writes `<dir>/<name>_counter_collection.csv` (one row per dispatch × counter, with the
# dispatch's timestamps and register/LDS/scratch footprint), `<name>_kernel_trace.csv` (every
# dispatch) and `<name>_agent_info.csv` (the device: CU / XCD / SE / SIMD counts, wavefront size).
# `rocprof_command` builds that wrapper around a `Cmd` (under `timeout -k`, because a counter
# request the hardware refuses aborts rocprofv3 with SIGABRT and leaves the profiled child hung),
# `rocprof_counters` parses the output for one kernel into a `RocprofCounters` table, and
# `rocprof_derived` / `rocprof_summary` / `rocprof_manifest_section` reduce it: medians with
# spreads across dispatches plus derived metrics with the normalisation rocprofv3's OWN
# derived-metric definitions use on gfx942 (`rocprofv3 --list-avail`, rocprofv3 1.1.0):
#
#     OccupancyPercent = 400·Σ SQ_WAVE_CYCLES / max_xcc(GRBM_GUI_ACTIVE) / CU_NUM / 32
#     SALUBusy         = 100·Σ SQ_INST_CYCLES_SALU / CU_NUM / max_xcc(GRBM_GUI_ACTIVE)
#
# i.e. (1) the CSV reports every counter SUMMED over its hardware dimensions — on the MI300X
# `GRBM_GUI_ACTIVE` has DIMENSION_XCC[0:7], so the reported value is 8× the cycles one die saw
# (a 40 ms dispatch shows 5.7e8 "cycles" = 14 GHz); the cycle count of the dispatch is
# `GRBM_GUI_ACTIVE / n_xcd` (`Num_Xcc` in the agent info; 1 on single-die parts). (2) `*_sum`
# unit counters are summed over every instance on the device (one TA/TD/TCP per CU), so a unit's
# busy fraction is `X_BUSY_sum / (cycles × n_cu)` with the device's CU count (304 on the MI300X).
# (3) The SQ wave-cycle counters (`SQ_WAVE_CYCLES`, `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`,
# `SQ_ACTIVE_INST_*`) are in QUAD-cycles (4 cycles) — ratios between them are unit-free, but
# resident waves = 4·SQ_WAVE_CYCLES / cycles; `SQ_BUSY_CYCLES` is plain cycles per shader engine.
# (4) `SQ_INSTS_*` count per-WAVE instruction issues; × wavefront size (64 on CDNA) / slots gives
# the per-slot (per work-item iteration) count a static instruction mix can be held against.
# Nothing here needs AMDGPU.jl: the parser is plain Julia and runs (and is tested) on any host.

"""    ROCPROF_COUNTER_SETS

Named `rocprofv3 --pmc` counter sets for [`rocprof_command`](@ref)`(…; counters = :name)`. Each
set is ONE pass (rocprofv3 collects a set per dispatch; the hardware's counter capacity is the
limit — a set that exceeds it aborts the profiled process, see below). What each answers, and
what [`rocprof_derived`](@ref) computes from it:

- `:sq_issue` — instruction issue per wave, by class: `SQ_INSTS_VMEM_RD/_WR`, `_SMEM`, `_SALU`,
  `_BRANCH`, `_VALU_INT64`, `_VALU` (+ `GRBM_GUI_ACTIVE`). → `insts_per_slot_<class>` (× wave
  size / slots): the dynamic instruction mix per slot to hold against the static
  `kernel_instruction_mix` of the hot loop; is the kernel load-, integer- or branch-heavy.
- `:sq_waves` — wave residency and what waves do: `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_WAVE_CYCLES`,
  `SQ_ACTIVE_INST_ANY/_VALU`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_ANY` (+ `GRBM_GUI_ACTIVE`). →
  `resident_waves`, `waves_per_cu`, `occupancy` (achieved, vs `kernel_occupancy` at compile
  time), `wave_wait_frac` (waiting for anything), `wave_wait_inst_frac` (waiting for an
  instruction to ISSUE — the difference to `wave_wait_frac` is the dependency / memory-latency
  wait), `wave_active_inst_frac`, `wave_active_valu_frac`, `sq_busy`: latency-bound (waves
  mostly waiting on data) vs issue-bound (waiting on the arbiter).
- `:l1_pipe` — the vector-memory pipe: `TA_TA_BUSY_sum`, `TD_TD_BUSY_sum`,
  `TCP_PENDING_STALL_CYCLES_sum`, `TCP_TOTAL_READ_sum`, `TCP_TCC_READ_REQ_sum`,
  `TCP_TOTAL_CACHE_ACCESSES_sum` (+ `GRBM_GUI_ACTIVE`). → `ta_busy`, `td_busy`,
  `tcp_pending_stall` (fractions of the dispatch, per CU), `l1_miss` (L1 → L2 read requests per
  cache access): is the address/data path saturated, does the working set fit L1.
- `:fp64` — FP64 VALU issue: `SQ_INSTS_VALU_FMA_F64`, `_ADD_F64`, `_MUL_F64`, `_TRANS_F64`
  (+ `GRBM_GUI_ACTIVE`). → `insts_per_slot_valu_fma_f64` etc. and `fp64_flop_per_slot`
  (2·FMA + ADD + MUL + TRANS, per slot): the hardware's own FLOP count per slot to hold against
  the algorithmic `[flops].flop_per_slot`.
- `:l2` — the L2 (TCC) and what leaves the die: `TCC_HIT_sum`, `TCC_MISS_sum`,
  `TCC_EA0_RDREQ_sum` (L2 → fabric read requests), `TCC_EA0_RDREQ_DRAM_sum` (of those, to HBM)
  (+ `GRBM_GUI_ACTIVE`). → `l2_hit`, `l2_dram_read_frac`; the `_median` values / slots give the
  L2 requests and HBM reads per slot: does the working set live in L2 or in HBM.

Verified on gfx942 (MI300X VF, ROCm 7.2.4, rocprofv3 1.1.0): `:sq_issue`, `:sq_waves` and
`:l1_pipe` collect in one pass on the production field kernel; `:fp64` and `:l2` on a test kernel
(`c = a + b * 1.5` over 2^22 doubles = 65 536 waves: exactly 1.00 `ADD_F64` + 1.00 `MUL_F64` per
element from `× 64 / slots`, and TCC hits / misses / fabric / HBM reads all populated). The TCC
block's capacity is FOUR counters per pass: adding `TCC_REQ_sum` as a fifth — with or without
`TCC_READ_sum` as a sixth, the original six-counter L2 set — makes rocprofv3 log "Request exceeds
the capabilities of the hardware to collect", abort with signal 6 and leave the profiled child
hung until `timeout -k` kills it (exit 137; hence the timeout in [`rocprof_command`](@ref)).
Sets are `Vector{String}` so a caller can pass its own list instead of a name."""
const ROCPROF_COUNTER_SETS = Dict{Symbol, Vector{String}}(
    :sq_issue => ["GRBM_GUI_ACTIVE", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_SMEM",
        "SQ_INSTS_SALU", "SQ_INSTS_BRANCH", "SQ_INSTS_VALU_INT64", "SQ_INSTS_VALU"],
    :sq_waves => ["SQ_WAVES", "SQ_BUSY_CYCLES", "GRBM_GUI_ACTIVE", "SQ_WAVE_CYCLES",
        "SQ_ACTIVE_INST_ANY", "SQ_ACTIVE_INST_VALU", "SQ_WAIT_INST_ANY", "SQ_WAIT_ANY"],
    :l1_pipe => ["GRBM_GUI_ACTIVE", "TA_TA_BUSY_sum", "TD_TD_BUSY_sum", "TCP_PENDING_STALL_CYCLES_sum",
        "TCP_TOTAL_READ_sum", "TCP_TCC_READ_REQ_sum", "TCP_TOTAL_CACHE_ACCESSES_sum"],
    :fp64 => ["GRBM_GUI_ACTIVE", "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_ADD_F64",
        "SQ_INSTS_VALU_MUL_F64", "SQ_INSTS_VALU_TRANS_F64"],
    :l2 => ["GRBM_GUI_ACTIVE", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_DRAM_sum"],
)

# The SQ counters rocprofv3 reports in quad-cycles (4 clock cycles) on gfx9 — see `--list-avail`.
const _ROCPROF_QUAD_CYCLE = 4

"""    rocprof_available() -> Bool

Whether `rocprofv3` is on the PATH (ROCm 6.2+; the counter path of [`rocprof_command`](@ref))."""
rocprof_available() = Sys.which("rocprofv3") !== nothing

"""
    rocprof_command(cmd::Cmd; counters, dir, name, timeout_s = 600, kernel_trace = true,
                    kill_after_s = 20, rocprofv3 = "rocprofv3") -> Cmd

Wrap `cmd` (environment and working directory preserved) in a rocprofv3 counter collection:

    timeout -k <kill_after_s> <timeout_s> rocprofv3 [--kernel-trace] --pmc <counters…>
        --output-format csv -d <dir> -o <name> -- <cmd>

`counters` is a name from [`ROCPROF_COUNTER_SETS`](@ref) or a vector of counter names. The
outputs land in `dir` as `<name>_counter_collection.csv` / `_kernel_trace.csv` / `_agent_info.csv`
for [`rocprof_counters`](@ref)`(dir; name)`. rocprofv3 multiplies each dispatch's time by
roughly its counter-read overhead only (kernels run at speed), but a counter request the hardware
cannot serve aborts rocprofv3 (signal 6) and leaves the profiled child hung — `timeout` signals
the whole process group when it expires and SIGKILLs it `kill_after_s` later, so the child dies
too. `run` the result, or `pipeline` it into a log."""
function rocprof_command(cmd::Cmd; counters, dir::AbstractString, name::AbstractString,
        timeout_s::Real = 600, kernel_trace::Bool = true, kill_after_s::Real = 20,
        rocprofv3::AbstractString = "rocprofv3")
    pmc = counters isa Symbol ? _rocprof_counter_set(counters) : String[String(c) for c in counters]
    isempty(pmc) && throw(ArgumentError("rocprof_command: no counters requested"))
    isempty(name) && throw(ArgumentError("rocprof_command: `name` (the output prefix) must not be empty"))
    timeout_s > 0 || throw(ArgumentError("rocprof_command: timeout_s must be positive"))
    trace = kernel_trace ? ["--kernel-trace"] : String[]
    wrapped = `timeout -k $(string(kill_after_s)) $(string(timeout_s)) $rocprofv3 $trace --pmc $pmc --output-format csv -d $dir -o $name -- $(cmd.exec)`
    return Cmd(wrapped; env = cmd.env, dir = cmd.dir)
end
function _rocprof_counter_set(s::Symbol)
    haskey(ROCPROF_COUNTER_SETS, s) ||
        throw(ArgumentError("rocprof_command: unknown counter set :$s — one of $(sort!(collect(keys(ROCPROF_COUNTER_SETS))))"))
    return ROCPROF_COUNTER_SETS[s]
end

# ── CSV ──────────────────────────────────────────────────────────────────────────────────────
# rocprofv3's CSV quotes strings (kernel names contain commas and angle brackets) and leaves
# numbers bare; RFC 4180 doubled quotes inside a quoted field are honoured.
function _csv_fields(line::AbstractString)
    out = String[]
    buf = IOBuffer()
    inq = false
    i = firstindex(line)
    n = lastindex(line)
    while i <= n
        c = line[i]
        if inq
            if c == '"'
                j = nextind(line, i)
                if j <= n && line[j] == '"'
                    write(buf, '"')
                    i = j
                else
                    inq = false
                end
            else
                write(buf, c)
            end
        elseif c == '"'
            inq = true
        elseif c == ','
            push!(out, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    push!(out, String(take!(buf)))
    return out
end

function _read_csv(path::AbstractString)
    header = String[]
    rows = Vector{String}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        f = _csv_fields(rstrip(line, '\r'))
        if isempty(header)
            header = f
        else
            length(f) == length(header) || throw(ArgumentError("$(basename(path)): row with $(length(f)) fields, header has $(length(header))"))
            push!(rows, f)
        end
    end
    isempty(header) && throw(ArgumentError("$(basename(path)) is empty"))
    return header, rows
end

function _csv_column(header::Vector{String}, name::AbstractString, path)
    j = findfirst(==(name), header)
    j === nothing && throw(ArgumentError("$(basename(path)): no column '$name' (columns: $(join(header, ", ")))"))
    return j
end

_int(s) = (v = tryparse(Int, s); v === nothing ? round(Int, parse(Float64, s)) : v)

# ── The table ────────────────────────────────────────────────────────────────────────────────

"""    RocprofCounters

Per-dispatch hardware counters of ONE kernel from a rocprofv3 run, built by
[`rocprof_counters`](@ref): `counters` (names), `dispatch_ids`, `values` (dispatches × counters),
`duration_s` (from the dispatch timestamps), `resources` (`grid_size`, `workgroup_size`,
`vgpr_count`, `agpr_count`, `sgpr_count`, `lds_bytes`, `scratch_bytes` of the kernel),
`device` (`name`, `product`, `n_cu`, `n_xcd`, `n_se`, `n_simd`, `wave_size`, `max_waves_per_cu`
from the agent info, 0 = unknown), `slots` (per dispatch, as given by the caller, or `nothing`),
`kernel` (the matched name) and `dispatches` (every dispatch of the run: `id`, `kernel`,
`duration_s`). Index a counter by name: `rc["SQ_WAVES"]` (a vector over dispatches);
`haskey(rc, "SQ_WAVES")`; `median(rc, "SQ_WAVES")` via [`rocprof_median`](@ref)."""
struct RocprofCounters
    dir::String
    name::String
    kernel::String
    counters::Vector{String}
    dispatch_ids::Vector{Int}
    values::Matrix{Float64}
    duration_s::Vector{Float64}
    resources::NamedTuple{(:grid_size, :workgroup_size, :vgpr_count, :agpr_count, :sgpr_count, :lds_bytes, :scratch_bytes), NTuple{7, Int}}
    device::NamedTuple{(:name, :product, :n_cu, :n_xcd, :n_se, :n_simd, :wave_size, :max_waves_per_cu), Tuple{String, String, Int, Int, Int, Int, Int, Int}}
    slots::Union{Nothing, Int}
    dispatches::Vector{NamedTuple{(:id, :kernel, :duration_s), Tuple{Int, String, Float64}}}
end

function Base.getindex(rc::RocprofCounters, c::AbstractString)
    j = findfirst(==(c), rc.counters)
    j === nothing && throw(KeyError(c))
    return rc.values[:, j]
end
Base.haskey(rc::RocprofCounters, c::AbstractString) = c in rc.counters
Base.keys(rc::RocprofCounters) = rc.counters
Base.length(rc::RocprofCounters) = length(rc.dispatch_ids)
Base.show(io::IO, rc::RocprofCounters) = print(io, "RocprofCounters(", rc.name, ": ", length(rc), " dispatches of ",
    _kernel_head(rc.kernel), " × ", length(rc.counters), " counters, median ",
    round(_median(rc.duration_s) * 1e3; digits = 2), " ms, device ", rc.device.name, ")")

_kernel_head(name::AbstractString) = String(first(split(name, '('; limit = 2)))

function _median(v::AbstractVector{<:Real})
    isempty(v) && return NaN
    s = sort(Float64[x for x in v])
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end
# Spread across dispatches relative to the median: (max − min) / |median|, 0 for a constant.
function _rel_spread(v::AbstractVector{<:Real})
    m = _median(v)
    (isempty(v) || m == 0) && return 0.0
    return (maximum(v) - minimum(v)) / abs(m)
end

"""    rocprof_median(rc, counter) -> Float64

Median of `counter` over the dispatches of a [`RocprofCounters`](@ref)."""
rocprof_median(rc::RocprofCounters, c::AbstractString) = _median(rc[c])

# `<name>_agent_info.csv`: one row per HSA agent; the GPU that ran the dispatches is matched by
# node id ("Agent 1" in the counter file ⇒ Node_Id 1), else the first GPU row.
const _NO_DEVICE = (name = "", product = "", n_cu = 0, n_xcd = 0, n_se = 0, n_simd = 0, wave_size = 0, max_waves_per_cu = 0)
function _rocprof_agent(path::AbstractString, agent_id::Union{Int, Nothing})
    isfile(path) || return _NO_DEVICE
    header, rows = _read_csv(path)
    col(n) = _csv_column(header, n, path)
    gpus = filter(r -> r[col("Agent_Type")] == "GPU", rows)
    isempty(gpus) && return _NO_DEVICE
    r = agent_id === nothing ? nothing : findfirst(r -> _int(r[col("Node_Id")]) == agent_id, gpus)
    row = gpus[something(r, 1)]
    get_int(n) = (j = findfirst(==(n), header); j === nothing ? 0 : _int(row[j]))
    return (name = row[col("Name")], product = row[col("Product_Name")], n_cu = get_int("Cu_Count"),
        n_xcd = get_int("Num_Xcc"), n_se = get_int("Num_Shader_Banks"), n_simd = get_int("Simd_Count"),
        wave_size = get_int("Wave_Front_Size"), max_waves_per_cu = get_int("Max_Waves_Per_Cu"))
end

function _rocprof_name(dir::AbstractString, name)
    name === nothing || return String(name)
    files = filter(f -> endswith(f, "_counter_collection.csv"), readdir(dir))
    length(files) == 1 && return String(chopsuffix(first(files), "_counter_collection.csv"))
    isempty(files) && throw(ArgumentError("rocprof_counters: no *_counter_collection.csv in $dir"))
    throw(ArgumentError("rocprof_counters: several counter collections in $dir — pass `name`: $(join(chopsuffix.(files, "_counter_collection.csv"), ", "))"))
end

_kernel_matches(kernel::Regex, name) = occursin(kernel, name)
_kernel_matches(kernel::AbstractString, name) = occursin(kernel, name)

"""
    rocprof_counters(dir; name = nothing, kernel = r"forindices", slots = nothing,
                     n_cu = nothing, n_xcd = nothing, wave_size = nothing) -> RocprofCounters

Parse the rocprofv3 outputs `<dir>/<name>_counter_collection.csv` (+ `_kernel_trace.csv` and
`_agent_info.csv` when present; `name` may be omitted when `dir` holds exactly one collection)
into the per-dispatch counter table of the kernel whose name matches `kernel` (a `Regex` or
substring; every KernelAbstractions kernel is a `gpu__forindices_global_(…)` on AMDGPU.jl, so
the default selects the KA kernel of a run with one — pass something more specific when several
KA kernels ran: the match must resolve to ONE distinct kernel name). `slots` is the number of
inner-loop iterations (work-items × per-item iterations) of ONE dispatch, the caller's knowledge,
for the per-slot metrics. `n_cu`, `n_xcd` and `wave_size` override the agent info — required when
the run has no `_agent_info.csv` (the KNOWN device values; nothing is defaulted). `n_xcd` is the
trap: every die (XCD) has its own GRBM block and the CSV reports `GRBM_GUI_ACTIVE` SUMMED over
them (`DIMENSION_XCC[0:7]` on gfx942), so a 40 ms dispatch shows 5.7e8 "cycles" — 14 GHz — until
divided by the 8 dies; the agent info's `Num_Xcc` is 8 on the MI300X and 1 on single-die parts,
and every per-cycle rate below (unit busy, resident waves, clock) uses `GRBM_GUI_ACTIVE / n_xcd`.
See [`rocprof_derived`](@ref) for what is computed from the table."""
function rocprof_counters(dir::AbstractString; name = nothing, kernel = r"forindices", slots = nothing,
        n_cu = nothing, n_xcd = nothing, wave_size = nothing)
    isdir(dir) || throw(ArgumentError("rocprof_counters: no such directory $dir"))
    nm = _rocprof_name(dir, name)
    path = joinpath(dir, nm * "_counter_collection.csv")
    isfile(path) || throw(ArgumentError("rocprof_counters: $path not found"))
    header, rows = _read_csv(path)
    col(n) = _csv_column(header, n, path)
    jid, jk, jc, jv, jt0, jt1 = col("Dispatch_Id"), col("Kernel_Name"), col("Counter_Name"), col("Counter_Value"),
        col("Start_Timestamp"), col("End_Timestamp")

    # group rows by dispatch, in dispatch order
    by_id = Dict{Int, Vector{Vector{String}}}()
    order = Int[]
    for r in rows
        id = _int(r[jid])
        haskey(by_id, id) || push!(order, id)
        push!(get!(() -> Vector{String}[], by_id, id), r)
    end
    sort!(order)
    isempty(order) && throw(ArgumentError("rocprof_counters: $path has no dispatches"))

    ids = [id for id in order if _kernel_matches(kernel, first(by_id[id])[jk])]
    isempty(ids) && throw(ArgumentError("rocprof_counters: no dispatch of $path matches kernel $kernel — kernels: " *
        join(unique(_kernel_head(first(by_id[id])[jk]) for id in order), ", ")))
    names = unique(first(by_id[id])[jk] for id in ids)
    length(names) == 1 || throw(ArgumentError("rocprof_counters: kernel $kernel matches $(length(names)) distinct kernels — be more specific: " *
        join(_kernel_head.(names), ", ")))
    kname = only(names)

    counters = String[]
    for id in ids, r in by_id[id]
        r[jc] in counters || push!(counters, r[jc])
    end
    values = fill(NaN, length(ids), length(counters))
    duration = zeros(length(ids))
    for (i, id) in enumerate(ids)
        rs = by_id[id]
        for r in rs
            values[i, findfirst(==(r[jc]), counters)] = parse(Float64, r[jv])
        end
        duration[i] = (_int(first(rs)[jt1]) - _int(first(rs)[jt0])) / 1.0e9
    end

    r1 = first(by_id[first(ids)])
    res_int(n) = _int(r1[col(n)])
    resources = (grid_size = res_int("Grid_Size"), workgroup_size = res_int("Workgroup_Size"),
        vgpr_count = res_int("VGPR_Count"), agpr_count = res_int("Accum_VGPR_Count"), sgpr_count = res_int("SGPR_Count"),
        lds_bytes = res_int("LDS_Block_Size"), scratch_bytes = res_int("Scratch_Size"))

    agent_id = (ja = findfirst(==("Agent_Id"), header); ja === nothing ? nothing :
        (m = match(r"(\d+)", r1[ja]); m === nothing ? nothing : parse(Int, m[1])))
    dev = _rocprof_agent(joinpath(dir, nm * "_agent_info.csv"), agent_id)
    dev = merge(dev, (n_cu = something(n_cu, dev.n_cu), n_xcd = something(n_xcd, dev.n_xcd),
        wave_size = something(wave_size, dev.wave_size)))

    trace = joinpath(dir, nm * "_kernel_trace.csv")
    dispatches = if isfile(trace)
        th, trows = _read_csv(trace)
        tc(n) = _csv_column(th, n, trace)
        [(id = _int(r[tc("Dispatch_Id")]), kernel = r[tc("Kernel_Name")],
            duration_s = (_int(r[tc("End_Timestamp")]) - _int(r[tc("Start_Timestamp")])) / 1.0e9) for r in trows]
    else
        [(id = id, kernel = first(by_id[id])[jk],
            duration_s = (_int(first(by_id[id])[jt1]) - _int(first(by_id[id])[jt0])) / 1.0e9) for id in order]
    end
    sort!(dispatches; by = d -> d.id)

    slots === nothing || slots > 0 || throw(ArgumentError("rocprof_counters: slots must be positive"))
    return RocprofCounters(String(dir), nm, kname, counters, ids, values, duration, resources, dev,
        slots === nothing ? nothing : Int(slots), dispatches)
end

# ── Derived metrics ──────────────────────────────────────────────────────────────────────────

"""
    rocprof_derived(rc::RocprofCounters) -> Dict{String, Float64}

Derived metrics of the kernel, each computed PER DISPATCH and reduced by the median — only
those whose counters the run collected (a set from [`ROCPROF_COUNTER_SETS`](@ref) yields the
metrics its docstring lists). With `cycles = GRBM_GUI_ACTIVE / n_xcd` (the dispatch's clock
cycles on one die; the CSV sums the counter over the dies), `n_cu` / `n_se` / `wave_size` /
`max_waves_per_cu` from `rc.device` and `slots` from the caller:

- `clock_GHz = cycles / duration` — the MEAN ACTIVE clock over the dispatch: `GRBM_GUI_ACTIVE`
  counts cycles while a die's GUI is active, so this equals the engine clock only while every
  die is busy for the whole dispatch and is a lower bound otherwise (idle tails count no cycles;
  it is meaningless for sub-ms dispatches, whose counter window exceeds the kernel). It is also
  the `n_xcd` sanity check (14 GHz means the die sum was not divided). Validated against the
  amdgpu sysfs engine clock (`freq1_input` sampled at 65 Hz through the same launches): 1.70 vs
  1.70 GHz and 1.25 vs 1.26 GHz median. The clock is power-managed on the MI300X (1.7 GHz at
  7 waves/CU, 1.25 GHz at 14 waves/CU on the same kernel — at the 750 W board cap), so compare
  two runs of the SAME work in CYCLES (`GRBM_GUI_ACTIVE / n_xcd`, `SQ_BUSY_CYCLES`), not in
  seconds: fewer cycles = more work per cycle; wall time falling less than the cycles = the clock
  dropped.
- `insts_per_slot_<class> = SQ_INSTS_<CLASS> × wave_size / slots` for every `SQ_INSTS_*`
  counter (`vmem_rd`, `valu`, `valu_fma_f64`, …); `fp64_flop_per_slot = (2·FMA + ADD + MUL +
  TRANS) × wave_size / slots` when the four FP64 classes are present.
- `<unit>_busy = <UNIT>_<UNIT>_BUSY_sum / (cycles × n_cu)` for every `*_BUSY_sum` counter
  (`ta_busy`, `td_busy`, …); `tcp_pending_stall = TCP_PENDING_STALL_CYCLES_sum / (cycles × n_cu)`.
- `l1_miss = TCP_TCC_READ_REQ_sum / TCP_TOTAL_CACHE_ACCESSES_sum`; `l2_hit = TCC_HIT_sum /
  (TCC_HIT_sum + TCC_MISS_sum)`; `l2_dram_read_frac = TCC_EA0_RDREQ_DRAM_sum / TCC_EA0_RDREQ_sum`.
- `wave_wait_frac = SQ_WAIT_ANY / SQ_WAVE_CYCLES` (waiting for anything), `wave_wait_inst_frac
  = SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` (waiting for an instruction to issue; the difference is
  the dependency / memory-latency wait), `wave_active_inst_frac = SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES`,
  `wave_active_valu_frac = SQ_ACTIVE_INST_VALU / SQ_WAVE_CYCLES` (all quad-cycle counters, so
  unit-free); `resident_waves = 4·SQ_WAVE_CYCLES / cycles`, `waves_per_cu = resident_waves /
  n_cu`, `occupancy = waves_per_cu / max_waves_per_cu` (rocprofv3's `OccupancyPercent` / 100);
  `wave_cycles = 4·SQ_WAVE_CYCLES / SQ_WAVES` (mean wave lifetime in cycles); `sq_busy =
  SQ_BUSY_CYCLES / (cycles × n_se)`.

Throws an `ArgumentError` naming the missing device value when a metric needs `n_xcd`, `n_cu`
or `wave_size` and neither the agent info nor the caller supplied it; per-slot metrics need
`slots` and are skipped without it."""
function rocprof_derived(rc::RocprofCounters)
    out = Dict{String, Float64}()
    dev = rc.device
    has(c) = haskey(rc, c)
    need(field, what) = (v = getfield(dev, field); v > 0 ? v :
        throw(ArgumentError("rocprof_derived: $what needs `$field` — not in the agent info; pass it to rocprof_counters (n_xcd = 8 on gfx942 / MI300X, 1 on single-die parts; n_cu = 304 on the MI300X)")))
    put!(key, v::AbstractVector) = (out[key] = _median(v); nothing)

    cycles = has("GRBM_GUI_ACTIVE") ? rc["GRBM_GUI_ACTIVE"] ./ need(:n_xcd, "the dispatch cycle count") : nothing
    cycles === nothing || put!("clock_GHz", cycles ./ rc.duration_s ./ 1.0e9)

    if rc.slots !== nothing
        for c in rc.counters
            startswith(c, "SQ_INSTS_") || continue
            put!("insts_per_slot_" * lowercase(c[10:end]), rc[c] .* need(:wave_size, "per-slot instruction counts") ./ rc.slots)
        end
        f64 = ("SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64", "SQ_INSTS_VALU_TRANS_F64")
        if all(has, f64)
            flop = 2 .* rc[f64[1]] .+ rc[f64[2]] .+ rc[f64[3]] .+ rc[f64[4]]
            put!("fp64_flop_per_slot", flop .* need(:wave_size, "fp64_flop_per_slot") ./ rc.slots)
        end
    end

    if cycles !== nothing
        for c in rc.counters
            m = match(r"^([A-Z]+)_[A-Z]+_BUSY_sum$", c)
            m === nothing && continue
            put!(lowercase(m[1]) * "_busy", rc[c] ./ (cycles .* need(:n_cu, "$c unit-busy fraction")))
        end
        has("TCP_PENDING_STALL_CYCLES_sum") &&
            put!("tcp_pending_stall", rc["TCP_PENDING_STALL_CYCLES_sum"] ./ (cycles .* need(:n_cu, "tcp_pending_stall")))
        if has("SQ_WAVE_CYCLES")
            resident = _ROCPROF_QUAD_CYCLE .* rc["SQ_WAVE_CYCLES"] ./ cycles
            put!("resident_waves", resident)
            if dev.n_cu > 0
                put!("waves_per_cu", resident ./ dev.n_cu)
                dev.max_waves_per_cu > 0 && put!("occupancy", resident ./ (dev.n_cu * dev.max_waves_per_cu))
            end
        end
        has("SQ_BUSY_CYCLES") && dev.n_se > 0 && put!("sq_busy", rc["SQ_BUSY_CYCLES"] ./ (cycles .* dev.n_se))
    end
    ratio(key, num, den) = has(num) && has(den) && put!(key, rc[num] ./ rc[den])
    ratio("l1_miss", "TCP_TCC_READ_REQ_sum", "TCP_TOTAL_CACHE_ACCESSES_sum")
    ratio("l2_dram_read_frac", "TCC_EA0_RDREQ_DRAM_sum", "TCC_EA0_RDREQ_sum")
    has("TCC_HIT_sum") && has("TCC_MISS_sum") && put!("l2_hit", rc["TCC_HIT_sum"] ./ (rc["TCC_HIT_sum"] .+ rc["TCC_MISS_sum"]))
    ratio("wave_wait_frac", "SQ_WAIT_ANY", "SQ_WAVE_CYCLES")
    ratio("wave_wait_inst_frac", "SQ_WAIT_INST_ANY", "SQ_WAVE_CYCLES")
    ratio("wave_active_inst_frac", "SQ_ACTIVE_INST_ANY", "SQ_WAVE_CYCLES")
    ratio("wave_active_valu_frac", "SQ_ACTIVE_INST_VALU", "SQ_WAVE_CYCLES")
    has("SQ_WAVE_CYCLES") && has("SQ_WAVES") && put!("wave_cycles", _ROCPROF_QUAD_CYCLE .* rc["SQ_WAVE_CYCLES"] ./ rc["SQ_WAVES"])
    return out
end

"""
    rocprof_summary(rc::RocprofCounters) -> Dict{String, Any}

Flat, manifest-ready reduction of a [`RocprofCounters`](@ref): `kernel` (name up to its
signature), `dispatches`, `dispatch_median_s` / `_min_s` / `_max_s` / `_total_s`, the kernel's
`grid_size`, `workgroup_size`, `vgpr_count`, `agpr_count`, `sgpr_count`, `lds_bytes`,
`scratch_bytes`, the device's `device`, `n_cu`, `n_xcd`, `n_se`, `wave_size`,
`max_waves_per_cu`, `slots` (when given), `counters` (the names), `<COUNTER>_median` and
`<COUNTER>_rel_spread` ((max − min) / |median| across dispatches) for every counter, and the
[`rocprof_derived`](@ref) metrics."""
function rocprof_summary(rc::RocprofCounters)
    out = Dict{String, Any}(
        "kernel" => _kernel_head(rc.kernel),
        "dispatches" => length(rc),
        "dispatch_median_s" => _median(rc.duration_s),
        "dispatch_min_s" => minimum(rc.duration_s),
        "dispatch_max_s" => maximum(rc.duration_s),
        "dispatch_total_s" => sum(rc.duration_s),
        "counters" => copy(rc.counters),
    )
    for k in keys(rc.resources)
        out[String(k)] = getfield(rc.resources, k)
    end
    out["device"] = rc.device.product
    for k in (:n_cu, :n_xcd, :n_se, :wave_size, :max_waves_per_cu)
        v = getfield(rc.device, k)
        v > 0 && (out[String(k)] = v)
    end
    rc.slots === nothing || (out["slots"] = rc.slots)
    for c in rc.counters
        out[c * "_median"] = _median(rc[c])
        out[c * "_rel_spread"] = _rel_spread(rc[c])
    end
    merge!(out, rocprof_derived(rc))
    return out
end

"""
    rocprof_manifest_section(dir; name = nothing, kernel = r"forindices", slots = nothing,
                             n_cu = nothing, n_xcd = nothing, wave_size = nothing, prefix = "rocprof_")
        -> Dict{String, Any}

[`rocprof_summary`](@ref) of [`rocprof_counters`](@ref)`(dir; …)` with every key prefixed
(`rocprof_td_busy`, `rocprof_SQ_WAVES_median`, …) — ready to `merge!` into a run manifest's
`[gpu]` table next to the sampler's `gpm_*` keys."""
function rocprof_manifest_section(dir::AbstractString; prefix::AbstractString = "rocprof_", kwargs...)
    rc = rocprof_counters(dir; kwargs...)
    return Dict{String, Any}(prefix * k => v for (k, v) in rocprof_summary(rc))
end
