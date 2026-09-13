# hw_counter_merge.jl — merge a hardware-counter collection into a run manifest's [gpu] table as
# hw_* keys (GPUDiagnostics.jl `diagnostics_dict(hc; prefix = "hw_")`; orchestration/profile_cell.sh
# runs this after the profiled cell returns). Prints the summary either way.
#
#   julia --project=scripts scripts/hw_counter_merge.jl <collection dir> <name> [--manifest=<toml>]
#         [--kernel=<regex>] [--slots=<n>] [--n-cu=<n>] [--n-xcd=<n>] [--no-write]
#
# <dir>/<name>_counter_collection.csv (+ _kernel_trace / _agent_info) is the rocprofv3 output;
# the manifest defaults to <dir>/run_<name>.toml (EDM_OUTDIR=<dir> EDM_RUN_TAG=<name>). What the
# per-slot metrics need — the slots of ONE dispatch — comes from the manifest unless --slots is
# given: [flops].slots_executed over the run's kernel launches ([gpu].kernel_launches summed over
# devices; the profiled process sees one device). The device properties every per-cycle rate needs
# (n_cu / n_xcd = 304 and 8 on the MI300X) come from the collection's agent info; --n-cu / --n-xcd
# override them per device id, and without agent info n_cu falls back to [gpu].sm_count. The hw_*
# keys already in [gpu] are dropped before the merge, so a re-run replaces them; legacy rocprof_*
# keys of pre-0.3 manifests stay (schema 2 renamed the family, it does not migrate old reports).
using GPUDiagnostics
using TOML

function parse_args(args)
    pos = String[]
    opt = Dict{String, String}()
    for a in args
        if startswith(a, "--")
            k, v = occursin('=', a) ? split(a[3:end], '='; limit = 2) : (a[3:end], "1")
            opt[String(k)] = String(v)
        else
            push!(pos, a)
        end
    end
    length(pos) == 2 || (println(stderr, "usage: hw_counter_merge.jl <collection dir> <name> [--manifest=…] [--kernel=…] [--slots=…] [--n-cu=…] [--n-xcd=…] [--no-write]"); exit(64))
    return pos[1], pos[2], opt
end

dir, name, opt = parse_args(ARGS)
manifest = get(opt, "manifest", joinpath(dir, "run_$name.toml"))
m = isfile(manifest) ? TOML.parsefile(manifest) : nothing
m === nothing && @warn "no run manifest at $manifest — printing the summary only"
gpu = m === nothing ? Dict{String, Any}() : get!(m, "gpu", Dict{String, Any}())

intopt(k) = haskey(opt, k) ? parse(Int, opt[k]) : nothing
slots = intopt("slots")
if slots === nothing && m !== nothing
    se = get(get(m, "flops", Dict()), "slots_executed", nothing)
    launches = get(gpu, "kernel_launches", nothing)
    if se !== nothing && launches !== nothing && sum(launches) > 0
        slots = round(Int, se / sum(launches))
    else
        @warn "slots per dispatch unknown ([flops].slots_executed / [gpu].kernel_launches missing) — per-slot metrics skipped; pass --slots"
    end
end
kernel = Regex(get(opt, "kernel", "forindices"))

hc = hw_counters(dir; name, kernel, slots)
d = first(hc.dispatches)
over = Dict{String, Any}()
for (flag, key) in (("n-cu", "n_cu"), ("n-xcd", "n_xcd"))
    n = intopt(flag)
    n === nothing || (over[key] = n)
end
if !haskey(over, "n_cu") && ismissing(get(d.device, "n_cu", missing)) && haskey(gpu, "sm_count")
    over["n_cu"] = Int(gpu["sm_count"])   # no agent info: the manifest's device snapshot
end
if !isempty(over)
    # device_overrides is keyed by the dispatch's own device id, so a first parse has to supply it
    if ismissing(d.device_id)
        @warn "the collection reports no device id — $(join(sort!(collect(keys(over))), ", ")) not applied"
    else
        hc = hw_counters(dir; name, kernel, slots, device_overrides = Dict(d.device_id => over))
    end
end

println(hc)
section = diagnostics_dict(hc; prefix = "hw_")
for k in sort!(collect(keys(section)))
    v = section[k]
    println(rpad(k, 52), v isa AbstractFloat ? round(v; sigdigits = 5) : v)
end

if m !== nothing && !haskey(opt, "no-write")
    filter!(kv -> !startswith(first(kv), "hw_"), gpu)
    merge!(gpu, section)
    m["gpu"] = gpu
    open(io -> TOML.print(io, m; sorted = true), manifest, "w")
    println("merged $(length(section)) keys into [gpu] of $manifest")
end
