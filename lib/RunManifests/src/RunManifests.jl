"""
    RunManifests

In-repo support package for ElectronDynamicsModels research runs. It centralises
everything reproducibility-related:

  * **git provenance** — `git_state`, the standard solver `run_provenance` block;
  * **a clean-tree guard** — `assert_committed`, so a run is never produced from
    uncommitted code its `repo_commit` cannot reproduce;
  * **manifest I/O** — `write_run_manifest` / `write_derived` (the `run_*.toml` and
    `derived_*.toml` the results dashboard consumes) and the `find_parent_*` readers;
  * **the replay seed** — `run_spec_from_manifest`, the inverse of how solver scripts
    read `ENV`, used by the reproduce/sweep launcher.

The solver/plot scripts get these via `using RunManifests`; `scripts/manifest.jl` is a
thin back-compat shim that re-exports them for scripts that still `include` it.
"""
module RunManifests

using TOML
using Dates

export git_state, assert_committed, run_provenance, run_spec_from_manifest, expand_sweep
export find_parent_manifest, find_parent_run, spp_from_manifest
export write_derived, write_run_manifest, write_solver_manifest, REQUIRED_CONFIG_KEYS

# Git commit of the repo holding this package (lib/ lives inside the EDM repo, so this is
# the EDM repo's HEAD), recorded in derived/analysis provenance so the dashboard can link
# the *plotting* script on GitHub at the exact commit — alongside the run's own link.
function _script_repo_commit()
    return try
        readchomp(Cmd(["git", "-C", @__DIR__, "rev-parse", "HEAD"]))
    catch
        "unknown"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Git provenance + clean-tree guard (shared by thomson_scattering.jl / _A / lpwa.jl,
# which each previously carried their own copy). The model/laser parameter dict stays
# in each script (MTK `prob.ps` vs script globals); only the invariant machinery is here.
# ─────────────────────────────────────────────────────────────────────────────

"""
    git_state(repo_dir) -> (; commit::String, dirty::Bool)

HEAD commit and working-tree state of the git repo at `repo_dir`. `dirty` reflects the full
`git status --porcelain` (untracked files count) — matching the `repo_dirty` field the
dashboard records. Returns `commit="unknown", dirty=false` if git is unavailable.
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
function assert_committed(
        repo_dir::AbstractString;
        allow_dirty::Bool = get(ENV, "EDM_ALLOW_DIRTY", "0") == "1"
    )
    tracked = try
        readchomp(Cmd(["git", "-C", string(repo_dir), "status", "--porcelain", "--untracked-files=no"]))
    catch
        ""
    end
    if !isempty(tracked) && !allow_dirty
        error(
            "Refusing to run on a dirty working tree at $repo_dir:\n$tracked\n" *
                "Commit so repo_commit reproduces this run, or set EDM_ALLOW_DIRTY=1 to override."
        )
    end
    return nothing
end

"""
    run_provenance(; run_id, gpu_backend, repo_dir, script=abspath(PROGRAM_FILE),
                   gpu_device=nothing) -> Dict{String,Any}

The standard solver-run `[provenance]` block. `gpu_device` (e.g. `CUDA.name(CUDA.device())`)
is recorded only when supplied — kept as an argument so this package stays free of any GPU
backend dependency.
"""
function run_provenance(;
        run_id, gpu_backend, repo_dir,
        script::AbstractString = abspath(PROGRAM_FILE), gpu_device = nothing
    )
    gs = git_state(repo_dir)
    prov = Dict{String, Any}(
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
in one place is what stops a "reproduce" from silently diverging from the original run.
"""
function run_spec_from_manifest(manifest::AbstractDict)
    prov = get(manifest, "provenance", Dict())
    commit = get(prov, "repo_commit", "unknown")
    cfg = get(manifest, "config", Dict())
    # EDM_* knobs come from `config` (the input layer / echo of ENV — authoritative for
    # replay, matching the dashboard's PARAM_SPEC); `gpu_backend` lives in `provenance`.
    env = Dict{String, String}()
    env["EDM_INITIAL_PHASE"] = string(cfg["initial_phase"])
    env["EDM_A0"] = string(cfg["a0"])
    env["EDM_NX"] = string(cfg["Nx"])
    env["EDM_N"] = string(cfg["N"])
    env["EDM_NSAMPLES"] = string(cfg["N_samples"])
    env["EDM_SPP"] = string(cfg["samples_per_period"])
    env["EDM_NSUBSTEPS"] = string(cfg["n_substeps"])
    env["EDM_GPU_BACKEND"] = string(prov["gpu_backend"])
    env["EDM_SYNC_PER_ELECTRON"] = string(cfg["sync_per_electron"])
    return (; commit, env)
end

"""
    expand_sweep(base, vary) -> Vector{Dict{String,String}}

Expand a sweep spec into the **run matrix** — the list of `EDM_*` environments to run.
`base` maps fixed knob names (no prefix, e.g. `"NX" => 64`) to values applied to every run;
`vary` maps swept knob names to a list of values (e.g. `"A0" => [1e-3, 1e-2, 0.1]`). Returns
one `Dict{String,String}` per run — keys `"EDM_<KNOB>"`, stringified values — with the `vary`
axes combined by cartesian product. The many-runs counterpart to [`run_spec_from_manifest`]
(which yields a single env from one stored run); this yields many from a compact grid spec.
"""
function expand_sweep(base::AbstractDict, vary::AbstractDict)
    # cartesian product over the vary value-lists — dimension-agnostic (1/2/3+ axes, or none)
    ks = sort(collect(keys(vary)))
    vals = [vary[k] for k in ks]
    runs = Dict{String, String}[]
    for variant in Iterators.product(vals...)
        run = Dict{String, String}("EDM_$bk" => string(bv) for (bk, bv) in base)
        for (k, v) in zip(ks, variant)
            run["EDM_$k"] = string(v)
        end
        push!(runs, run)
    end
    return runs
end

# ─────────────────────────────────────────────────────────────────────────────
# Manifest readers + writers (run_*.toml / derived_*.toml the dashboard consumes).
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_parent_manifest(dir, datafile) -> (run_id, manifest::Dict) | nothing

Find the run a derived plot was computed from: scan `dir` for a `run_*.toml` whose
`[outputs].datafile` equals `datafile` (a basename) and return its `provenance.run_id`
together with the parsed manifest. Binds derived plots to their run — and lets
post-processing read run parameters from the manifest rather than from filenames.
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

The `provenance.run_id` of the run that produced `datafile` (see [`find_parent_manifest`](@ref)),
or `nothing` if no run manifest binds it.
"""
function find_parent_run(dir::AbstractString, datafile::AbstractString)
    r = find_parent_manifest(dir, datafile)
    return r === nothing ? nothing : r[1]
end

"""
    spp_from_manifest(manifest; default = nothing) -> Int

Read `samples_per_period` from a parsed run `manifest` (`[config]`, falling back to
`[setup]`). Errors if absent and no `default` is given — post-processing should get this
from the run TOML, not by parsing the data filename.
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
function write_derived(
        dir::AbstractString; kind, label, run_id, plot,
        source = nothing, datafile = nothing, setup = Dict(), description = nothing
    )
    d = Dict{String, Any}("kind" => kind, "label" => label, "depends_on" => [run_id], "plot" => plot)
    source === nothing || (d["source"] = source)
    datafile === nothing || (d["datafile"] = datafile)
    # `description`: markdown + $…$ LaTeX, rendered (KaTeX) in the dashboard plot modal.
    description === nothing || (d["description"] = description)
    m = Dict{String, Any}(
        "provenance" => Dict(
            "script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now())
        ),
        "derived" => d,
    )
    isempty(setup) || (m["setup"] = Dict{String, Any}(string(k) => v for (k, v) in setup))
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
function write_run_manifest(
        dir::AbstractString; run_id, script, config = Dict(),
        laser = Dict(), setup = Dict(), derived_from = nothing, datafile = nothing, plots = String[]
    )
    prov = Dict{String, Any}(
        "run_id" => run_id, "script" => script, "repo_commit" => _script_repo_commit(),
        "host" => gethostname(), "timestamp" => string(now())
    )
    derived_from === nothing || (prov["derived_from"] = derived_from)
    outs = Dict{String, Any}("plots" => collect(plots))
    datafile === nothing || (outs["datafile"] = datafile)
    m = Dict{String, Any}("provenance" => prov, "outputs" => outs)
    isempty(config) || (m["config"] = Dict{String, Any}(string(k) => v for (k, v) in config))
    isempty(laser)  || (m["laser"] = Dict{String, Any}(string(k) => v for (k, v) in laser))
    isempty(setup)  || (m["setup"] = Dict{String, Any}(string(k) => v for (k, v) in setup))
    open(io -> TOML.print(io, m; sorted = true), joinpath(dir, "run_$(run_id).toml"), "w")
    return joinpath(dir, "run_$(run_id).toml")
end

# Required [config] keys — the write side of run_spec_from_manifest's replay contract.
const REQUIRED_CONFIG_KEYS = (
    "initial_phase",
    "a0",
    "Nx",
    "N",
    "N_samples",
    "samples_per_period",
    "n_substeps",
    "sync_per_electron",
)

"""
    write_solver_manifest(dir; run_id, provenance, config, laser, setup, outputs, extra = Dict())

Canonical `run_<run_id>.toml` writer for a PRIMARY solver run (one that produces a `.jls`
+ plots). The single owner of the section layout, so the producer scripts
(`thomson_scattering.jl`, `_A.jl`, `lpwa.jl`) can no longer drift apart:

  [provenance] — pass `run_provenance(...)`         [config]  — replay-input knobs
  [laser]      — beam params (dashboard PARAM_SPEC)  [setup]   — Z/Rmax + integration window
  [outputs]    — `datafile` + `plots` (+ any extras the script records)

`extra` maps any further top-level section name to its dict, written verbatim (e.g. lpwa's
lpwa-only `"model"` bookkeeping). Errors if `config` lacks a `REQUIRED_CONFIG_KEYS` entry,
so the replay contract is enforced at write time, not discovered at replay time.
"""
function write_solver_manifest(
        dir::AbstractString; run_id, provenance::AbstractDict,
        config::AbstractDict, laser::AbstractDict, setup::AbstractDict,
        outputs::AbstractDict, extra::AbstractDict = Dict()
    )
    miss = [k for k in REQUIRED_CONFIG_KEYS if !haskey(config, k)]
    isempty(miss) || error(
        "write_solver_manifest: [config] is missing replay key(s) $(join(miss, ", ")); " *
            "run_spec_from_manifest needs them to reproduce this run."
    )
    sec(d) = Dict{String, Any}(string(k) => v for (k, v) in d)
    m = Dict{String, Any}(
        "provenance" => Dict{String, Any}(provenance),
        "config" => sec(config), "laser" => sec(laser),
        "setup" => sec(setup), "outputs" => sec(outputs),
    )
    for (name, d) in extra
        m[string(name)] = sec(d)
    end
    path = joinpath(dir, "run_$(run_id).toml")
    open(io -> TOML.print(io, m; sorted = true), path, "w")
    return path
end

end # module RunManifests
