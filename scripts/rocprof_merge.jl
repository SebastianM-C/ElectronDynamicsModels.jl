# rocprof_merge.jl — merge a rocprofv3 counter collection into a run manifest's [gpu] table as
# rocprof_* keys (lib/GPUDiagnostics `rocprof_manifest_section`; orchestration/profile_cell.sh
# runs this after the profiled cell returns). Prints the summary either way.
#
#   julia --project=scripts scripts/rocprof_merge.jl <rocprof dir> <name> [--manifest=<toml>]
#         [--kernel=<regex>] [--slots=<n>] [--n-cu=<n>] [--n-xcd=<n>] [--no-write]
#
# <dir>/<name>_counter_collection.csv (+ _kernel_trace / _agent_info) is the rocprofv3 output;
# the manifest defaults to <dir>/run_<name>.toml (EDM_OUTDIR=<dir> EDM_RUN_TAG=<name>). What the
# per-slot metrics need — the slots of ONE dispatch — comes from the manifest unless --slots is
# given: [flops].slots_executed over the run's kernel launches ([gpu].kernel_launches summed over
# devices; the profiled process sees one device). n_cu defaults to the agent info's CU count, then
# [gpu].sm_count; n_xcd to the agent info (8 on the MI300X). Keys already present are replaced.
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
    length(pos) == 2 || (println(stderr, "usage: rocprof_merge.jl <rocprof dir> <name> [--manifest=…] [--kernel=…] [--slots=…] [--n-cu=…] [--n-xcd=…] [--no-write]"); exit(64))
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
n_cu = intopt("n-cu")
n_xcd = intopt("n-xcd")
kernel = Regex(get(opt, "kernel", "forindices"))

rc = rocprof_counters(dir; name, kernel, slots, n_cu, n_xcd)
if rc.device.n_cu == 0 && haskey(gpu, "sm_count")   # no agent info: the manifest's device snapshot
    rc = rocprof_counters(dir; name, kernel, slots, n_cu = Int(gpu["sm_count"]), n_xcd)
end
println(rc)
section = Dict{String, Any}("rocprof_" * k => v for (k, v) in rocprof_summary(rc))
for k in sort!(collect(keys(section)))
    v = section[k]
    println(rpad(k, 44), v isa AbstractFloat ? round(v; sigdigits = 5) : v)
end

if m !== nothing && !haskey(opt, "no-write")
    merge!(gpu, section)
    m["gpu"] = gpu
    open(io -> TOML.print(io, m; sorted = true), manifest, "w")
    println("merged $(length(section)) rocprof_* keys into [gpu] of $manifest")
end
