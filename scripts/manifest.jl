# manifest.jl — shared helpers for emitting research-dashboard metadata from
# post-processing scripts. Usage from a script:
#
#   include(joinpath(@__DIR__, "manifest.jl"))
#   pid = find_parent_run(dirname(abspath(datafile)), basename(datafile))
#   pid === nothing || write_derived(dirname(abspath(datafile)); kind="powspec",
#       label="power spectrum", run_id=pid, plot=basename(out), source=basename(datafile))
#
# The dashboard (research.314159265.dev) reads run_<uuid>.toml (emitted by
# thomson_scattering.jl) and these derived_*.toml sidecars; see EDM_results_dashboard.

using TOML
using Dates

"""
    find_parent_run(dir, datafile) -> run_id | nothing

Find the run a derived plot was computed from: scan `dir` for a `run_*.toml` whose
`[outputs].datafile` equals `datafile` (a basename) and return its `provenance.run_id`.
Binds derived plots to their run without needing the run id encoded in filenames.
"""
function find_parent_run(dir::AbstractString, datafile::AbstractString)
    isdir(dir) || return nothing
    for f in readdir(dir)
        (startswith(f, "run_") && endswith(f, ".toml")) || continue
        m = try
            TOML.parsefile(joinpath(dir, f))
        catch
            continue
        end
        if get(get(m, "outputs", Dict()), "datafile", nothing) == datafile
            return get(get(m, "provenance", Dict()), "run_id", nothing)
        end
    end
    return nothing
end

"""
    write_derived(dir; kind, label, run_id, plot, source=nothing, datafile=nothing, setup=Dict())

Write a `[derived]` sidecar TOML into `dir` binding `plot` (a basename in `dir`) to parent
`run_id`. `setup` keys that vary across same-kind sidecars become a secondary picker axis
in the dashboard; `source` records the input artifact as provenance.
"""
function write_derived(dir::AbstractString; kind, label, run_id, plot,
        source = nothing, datafile = nothing, setup = Dict())
    d = Dict{String,Any}("kind" => kind, "label" => label, "depends_on" => [run_id], "plot" => plot)
    source === nothing || (d["source"] = source)
    datafile === nothing || (d["datafile"] = datafile)
    m = Dict{String,Any}(
        "provenance" => Dict("script" => basename(PROGRAM_FILE), "host" => gethostname(),
            "timestamp" => string(now())),
        "derived" => d,
    )
    isempty(setup) || (m["setup"] = Dict{String,Any}(string(k) => v for (k, v) in setup))
    suffix = isempty(setup) ? "" : "_" * join(string.(values(setup)), "-")
    name = "derived_$(kind)$(suffix)_$(first(string(run_id), 8)).toml"
    open(io -> TOML.print(io, m; sorted = true), joinpath(dir, name), "w")
    return joinpath(dir, name)
end

"""
    write_run_manifest(dir; run_id, script, config=Dict(), laser=Dict(), setup=Dict(),
                       derived_from=nothing, datafile=nothing, plots=String[])

Write a `run_<run_id>.toml` for an analysis node — a run-like entity (e.g. a Lorenz-gauge
verification) with its own parameters but no primary `.jls`. Pass `derived_from` (a parent
run_id) to make the dashboard show it as an "analysis" with a lineage link to that run.
"""
function write_run_manifest(dir::AbstractString; run_id, script, config = Dict(),
        laser = Dict(), setup = Dict(), derived_from = nothing, datafile = nothing, plots = String[])
    prov = Dict{String,Any}("run_id" => run_id, "script" => script,
        "host" => gethostname(), "timestamp" => string(now()))
    derived_from === nothing || (prov["derived_from"] = derived_from)
    outs = Dict{String,Any}("plots" => collect(plots))
    datafile === nothing || (outs["datafile"] = datafile)
    m = Dict{String,Any}("provenance" => prov, "outputs" => outs)
    isempty(config) || (m["config"] = Dict{String,Any}(string(k) => v for (k, v) in config))
    isempty(laser)  || (m["laser"]  = Dict{String,Any}(string(k) => v for (k, v) in laser))
    isempty(setup)  || (m["setup"]  = Dict{String,Any}(string(k) => v for (k, v) in setup))
    open(io -> TOML.print(io, m; sorted = true), joinpath(dir, "run_$(run_id).toml"), "w")
    return joinpath(dir, "run_$(run_id).toml")
end
