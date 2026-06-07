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

# Git commit of the repo holding the post-processing scripts (this file lives in
# scripts/), recorded in derived/analysis provenance so the dashboard can link the
# *plotting* script on GitHub at the exact commit — alongside the run's own link.
function _script_repo_commit()
    return try
        readchomp(Cmd(["git", "-C", @__DIR__, "rev-parse", "HEAD"]))
    catch
        "unknown"
    end
end

"""
    find_parent_manifest(dir, datafile) -> (run_id, manifest::Dict) | nothing

Find the run a derived plot was computed from: scan `dir` for a `run_*.toml` whose
`[outputs].datafile` equals `datafile` (a basename) and return its
`provenance.run_id` together with the parsed manifest. Binds derived plots to
their run — and lets post-processing read run parameters (e.g.
`samples_per_period`) from the manifest rather than from filenames.
"""
function find_parent_manifest(dir::AbstractString, datafile::AbstractString)
    isdir(dir) || return nothing
    for f in readdir(dir)
        (startswith(f, "run_") && endswith(f, ".toml")) || continue
        m = try
            TOML.parsefile(joinpath(dir, f))
        catch
            continue
        end
        if get(get(m, "outputs", Dict()), "datafile", nothing) == datafile
            return (get(get(m, "provenance", Dict()), "run_id", nothing), m)
        end
    end
    return nothing
end

"""
    find_parent_run(dir, datafile) -> run_id | nothing

The `provenance.run_id` of the run that produced `datafile` (see
[`find_parent_manifest`](@ref)), or `nothing` if no run manifest binds it.
"""
function find_parent_run(dir::AbstractString, datafile::AbstractString)
    r = find_parent_manifest(dir, datafile)
    return r === nothing ? nothing : r[1]
end

"""
    spp_from_manifest(manifest; default = nothing) -> Int

Read `samples_per_period` from a parsed run `manifest` (`[config]`, falling back to
`[setup]`). Errors if absent and no `default` is given — post-processing should get
this from the run TOML, not by parsing the data filename.
"""
function spp_from_manifest(manifest::AbstractDict; default = nothing)
    for sec in ("config", "setup")
        v = get(get(manifest, sec, Dict()), "samples_per_period", nothing)
        v === nothing || return Int(v)
    end
    default === nothing && error("samples_per_period not found in run manifest [config]/[setup]")
    return default
end

"""
    write_derived(dir; kind, label, run_id, plot, source=nothing, datafile=nothing, setup=Dict())

Write a `[derived]` sidecar TOML into `dir` binding `plot` (a basename in `dir`) to parent
`run_id`. `setup` keys that vary across same-kind sidecars become a secondary picker axis
in the dashboard; `source` records the input artifact as provenance.
"""
function write_derived(dir::AbstractString; kind, label, run_id, plot,
        source = nothing, datafile = nothing, setup = Dict(), description = nothing)
    d = Dict{String,Any}("kind" => kind, "label" => label, "depends_on" => [run_id], "plot" => plot)
    source === nothing || (d["source"] = source)
    datafile === nothing || (d["datafile"] = datafile)
    # `description`: markdown + $…$ LaTeX, rendered (KaTeX) in the dashboard plot modal.
    description === nothing || (d["description"] = description)
    m = Dict{String,Any}(
        "provenance" => Dict("script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now())),
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
    prov = Dict{String,Any}("run_id" => run_id, "script" => script, "repo_commit" => _script_repo_commit(),
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

# ─────────────────────────────────────────────────────────────────────────────
# Solver-run provenance — shared by thomson_scattering.jl / _A / lpwa.jl, which
# each previously carried their own copy of the git capture + provenance dict.
# The model/laser parameter dict stays in each script (MTK `prob.ps` vs globals);
# only the invariant identity/provenance machinery lives here.
# ─────────────────────────────────────────────────────────────────────────────

"""
    git_state(repo_dir) -> (; commit::String, dirty::Bool)

HEAD commit and working-tree state of the git repo at `repo_dir`. `dirty` is the full
`git status --porcelain` (untracked files count) — matching the `repo_dirty` field the
dashboard already records. Returns `commit="unknown", dirty=false` if git is unavailable.
"""
function git_state(repo_dir::AbstractString)
    git(args...) = try
        readchomp(Cmd(["git", "-C", string(repo_dir), args...]))
    catch
        "unknown"
    end
    status = git("status", "--porcelain")
    return (; commit = git("rev-parse", "HEAD"), dirty = !(status == "" || status == "unknown"))
end

"""
    assert_committed(repo_dir; allow_dirty = get(ENV,"EDM_ALLOW_DIRTY","0")=="1")

Fail fast if the repo at `repo_dir` has **uncommitted tracked changes**, so a run is never
produced from code its `repo_commit` cannot reproduce. Untracked files are ignored (scratch
and outputs don't affect a committed-code run; a missing committed dependency fails loudly on
a fresh clone anyway). Set `EDM_ALLOW_DIRTY=1` to override for throwaway/debug runs.
"""
function assert_committed(repo_dir::AbstractString;
        allow_dirty::Bool = get(ENV, "EDM_ALLOW_DIRTY", "0") == "1")
    tracked = try
        readchomp(Cmd(["git", "-C", string(repo_dir), "status", "--porcelain", "--untracked-files=no"]))
    catch
        ""
    end
    if !isempty(tracked) && !allow_dirty
        error("Refusing to run on a dirty working tree at $repo_dir:\n$tracked\n" *
              "Commit so repo_commit reproduces this run, or set EDM_ALLOW_DIRTY=1 to override.")
    end
    return nothing
end

"""
    run_provenance(; run_id, gpu_backend, repo_dir, script=abspath(PROGRAM_FILE),
                   gpu_device=nothing) -> Dict{String,Any}

The standard solver-run `[provenance]` block. `gpu_device` (e.g. `CUDA.name(CUDA.device())`)
is recorded only when supplied — kept as an argument so this helper stays free of any GPU
backend dependency.
"""
function run_provenance(; run_id, gpu_backend, repo_dir,
        script::AbstractString = abspath(PROGRAM_FILE), gpu_device = nothing)
    gs = git_state(repo_dir)
    prov = Dict{String,Any}(
        "run_id" => run_id,
        "repo_commit" => gs.commit,
        "repo_dirty" => gs.dirty,
        "edm_pkgdir" => string(repo_dir),
        "script" => script,
        "host" => gethostname(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "gpu_backend" => gpu_backend,
        "julia_version" => string(VERSION),
        "timestamp" => string(now()),
    )
    gpu_device === nothing || (prov["gpu_device"] = string(gpu_device))
    return prov
end

"""
    run_spec_from_manifest(manifest) -> (; commit::String, env::Dict{String,String})

The replay seed consumed by the reproduce/sweep launcher: from a parsed run `manifest`
(`TOML.parsefile(run_*.toml)`), return the git `commit` and the `EDM_*` environment that
reproduces the run on a fresh checkout. This is the exact inverse of the way the solver
scripts read `ENV` at the top and write the manifest at the end — keeping both directions
here is what stops a "reproduce" from silently diverging from the original run.
"""
function run_spec_from_manifest(manifest::AbstractDict)
    prov   = get(manifest, "provenance", Dict())
    commit = get(prov, "repo_commit", "unknown")
    cfg    = get(manifest, "config", Dict())
    model  = get(manifest, "model", get(manifest, "laser", Dict()))
    env = Dict{String,String}()
    # TODO(human): populate `env` with the EDM_* knobs that reproduce this run, reading
    # from `cfg` / `model` / `prov`. The solver reads these from ENV at the top of
    # lpwa.jl / thomson_scattering.jl (EDM_NX, EDM_N, EDM_NSAMPLES, EDM_SPP, EDM_NSUBSTEPS,
    # EDM_A0, EDM_INITIAL_PHASE, EDM_GPU_BACKEND, EDM_SYNC_PER_ELECTRON). Map each back to
    # its manifest location, and decide which section is authoritative when a value lives
    # in more than one (e.g. a0 in both config & model; phi0 vs config.initial_phase).
    return (; commit, env)
end
