"""
    RunManifests

In-repo support package for ElectronDynamicsModels research runs. It centralises
everything reproducibility-related:

  * **git provenance** — `git_state`, the standard solver `run_provenance` block;
  * **a clean-tree guard** — `assert_committed`, so a run is never produced from
    uncommitted code its `repo_commit` cannot reproduce;
  * **manifest I/O** — `write_run_manifest` / `write_derived` / `write_comparison` (the
    `run_*.toml`, `derived_*.toml`, and `comparison_*.toml` the results dashboard consumes)
    and the `find_parent_*` readers;
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
export write_derived, write_comparison, write_run_manifest, write_solver_manifest, REQUIRED_CONFIG_KEYS
export MANIFEST_SCHEMA_VERSION, manifest_schema_version, check_schema_version

# ─────────────────────────────────────────────────────────────────────────────
# Manifest schema version. `schema_version` is a top-level Int in every run_*.toml /
# derived_*.toml; bump MANIFEST_SCHEMA_VERSION whenever the section layout changes so a
# reader refuses a layout it can't read instead of silently mis-reading a renamed key.
# Policy (check_schema_version): missing ⇒ legacy v0 (warn, proceed); newer than we know
# ⇒ error. Migration of an older-but-known layout goes here once the first such change
# lands (v0 and v1 are structurally identical — v0 just predates the field).
# ─────────────────────────────────────────────────────────────────────────────
const MANIFEST_SCHEMA_VERSION = 1

"""
    manifest_schema_version(manifest) -> Int

The `schema_version` at the top of a parsed manifest, or `0` for a pre-versioning
("legacy") manifest written before the field existed.
"""
manifest_schema_version(m::AbstractDict) = Int(get(m, "schema_version", 0))

"""
    check_schema_version(manifest; source = "manifest") -> Int

Validate a parsed manifest against `MANIFEST_SCHEMA_VERSION` and return its detected
version. Missing ⇒ warn and treat as legacy `v0`; a version newer than this package
knows ⇒ error (it was written by a newer RunManifests — update this one to read it).
"""
function check_schema_version(m::AbstractDict; source::AbstractString = "manifest")
    v = manifest_schema_version(m)
    if v == 0
        @warn "$source has no schema_version — assuming legacy layout (v0)" expected = MANIFEST_SCHEMA_VERSION
    elseif v > MANIFEST_SCHEMA_VERSION
        error(
            "$source has schema_version=$v, newer than this RunManifests understands " *
                "(v$MANIFEST_SCHEMA_VERSION); update RunManifests to read it."
        )
    end
    return v
end

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
    check_schema_version(manifest; source = "run manifest")
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
    env["EDM_FIELD_MODE"] = string(get(cfg, "mode", "split"))   # default keeps pre-mode manifests replaying as before
    # Inverse-Thomson knobs (inverse_thomson_scattering.jl) — guarded: absent in rest-electron
    # manifests. Without these a "reproduce" of a boosted/narrow run silently reran the γ=10
    # :full defaults (and misused the narrow-mode auto-sized N_samples as a :full window length).
    haskey(cfg, "gamma") && (env["EDM_GAMMA"] = string(cfg["gamma"]))
    haskey(cfg, "window") && (env["EDM_WINDOW"] = string(cfg["window"]))
    haskey(cfg, "screen_hw_w0") && (env["EDM_SCREEN_HW"] = string(cfg["screen_hw_w0"]))
    haskey(cfg, "harmonics") && (env["EDM_HARMONICS"] = join(cfg["harmonics"], ","))
    haskey(cfg, "reltol") && (env["EDM_RELTOL"] = string(cfg["reltol"]))
    haskey(cfg, "abstol") && (env["EDM_ABSTOL"] = string(cfg["abstol"]))
    sv = get(cfg, "interp_saveat", "adaptive")
    sv != "adaptive" && (env["EDM_INTERP_SAVEAT"] = string(sv))
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
    write_derived(dir; kind, label, run_id, plot, source=nothing, datafile=nothing,
                  setup=Dict(), plot_params=Dict(), description=nothing)

Write a `[derived]` sidecar TOML into `dir` binding `plot` (a basename in `dir`) to its parent
run(s). `run_id` is a single id OR a vector of ids — the builder attaches the artifact to EVERY
parent in `depends_on`, so a cross-run comparison passes both run ids and shows up (with lineage)
under both. `source` records the input artifact as provenance.

Two distinct parameter channels (don't conflate them):

  * `setup` → the `[setup]` section. Keys that **vary** across same-kind sidecars become a
    secondary *picker axis* in the dashboard (cf. runs → sweeps); non-varying keys are dropped.
  * `plot_params` → the `[plot_params]` section. **Display-only** diagnostic parameters of how
    the plot was made (e.g. ring radii, an annulus tolerance) — surfaced verbatim in the plot
    modal, never a picker. Use this for values that are constant across the sidecar family and
    so would silently vanish from `setup`. Orthogonal to `[config]`/`[laser]`/`[setup]`,
    analogous to the optional `[timing]` block on runs.
"""
function write_derived(
        dir::AbstractString; kind, label, run_id, plot,
        source = nothing, datafile = nothing, setup = Dict(), plot_params = Dict(),
        description = nothing
    )
    deps = run_id isa AbstractString ? [string(run_id)] : [string(x) for x in run_id]
    d = Dict{String, Any}("kind" => kind, "label" => label, "depends_on" => deps, "plot" => plot)
    source === nothing || (d["source"] = source)
    datafile === nothing || (d["datafile"] = datafile)
    # `description`: markdown + $…$ LaTeX, rendered (KaTeX) in the dashboard plot modal.
    description === nothing || (d["description"] = description)
    m = Dict{String, Any}(
        "schema_version" => MANIFEST_SCHEMA_VERSION,
        "provenance" => Dict(
            "script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now())
        ),
        "derived" => d,
    )
    isempty(setup) || (m["setup"] = Dict{String, Any}(string(k) => v for (k, v) in setup))
    isempty(plot_params) || (m["plot_params"] = Dict{String, Any}(string(k) => v for (k, v) in plot_params))
    suffix = isempty(setup) ? "" : "_" * join(string.(values(setup)), "-")   # filename: setup keys only
    idtag = join((first(x, 8) for x in deps), "-")   # <id8> for one parent, <id8a>-<id8b> for a comparison
    name = "derived_$(kind)$(suffix)_$(idtag).toml"
    open(io -> TOML.print(io, m; sorted = true), joinpath(dir, name), "w")
    return joinpath(dir, name)
end

# Normalise one comparison side spec to (label, dir, script). Accepts a NamedTuple
# `(; label, dir[, script])`, a `Dict`, or a `(label, dir[, script])` tuple.
function _side_fields(s)
    s isa AbstractDict && return (s["label"], s["dir"], get(s, "script", nothing))
    s isa NamedTuple && return (s.label, s.dir, hasproperty(s, :script) ? s.script : nothing)
    s isa Tuple && return (s[1], s[2], length(s) >= 3 ? s[3] : nothing)
    return error("write_comparison: unrecognised side spec of type $(typeof(s))")
end

"""
    write_comparison(dir; label, sides, differs=nothing, along=nothing, filename=nothing)

Write a `[comparison]` declaration sidecar TOML into `dir` — the first-class comparison the
results dashboard surfaces (top-level `comparisons` in `index.json`). Where [`write_derived`](@ref)
records ONE diff plot bound to its parent runs, this declares the RELATIONSHIP: which sweeps (or
runs) are compared, matched cell-by-cell along a shared swept axis.

`sides` is a vector of at least two side specs — each a NamedTuple `(; label, dir[, script])`,
a `Dict`, or a `(label, dir[, script])` tuple. `dir` is a results-dir **basename** the dashboard
resolves to the sweep auto-detected there; the optional `script` disambiguates a dir holding more
than one sweep (e.g. an LPWA and a Thomson run in the same folder). `differs` is a free-form label
for what distinguishes the sides (e.g. `"method"`); `along` names the shared swept axis to match on
— **omit it** to let the dashboard infer the sides' common axis (the usual case, since a per-pair
caller can't see what the whole campaign sweeps).

Idempotent: `filename` defaults to a deterministic slug of the side dirs, so a per-pair comparison
re-run across a sweep rewrites ONE declaration instead of accumulating copies. Stamps the current
`schema_version` like the other writers.
"""
function write_comparison(
        dir::AbstractString; label, sides,
        differs = nothing, along = nothing, filename = nothing
    )
    length(sides) >= 2 || error("write_comparison: need ≥2 sides, got $(length(sides))")
    sidedicts = Dict{String, Any}[]
    dirtags = String[]
    for s in sides
        sl, sd, sc = _side_fields(s)
        d = Dict{String, Any}("label" => string(sl), "dir" => string(sd))
        sc === nothing || (d["script"] = string(sc))
        push!(sidedicts, d)
        push!(dirtags, string(sd))
    end
    comp = Dict{String, Any}("label" => label, "side" => sidedicts)
    differs === nothing || (comp["differs"] = differs)
    along === nothing || (comp["along"] = along)
    m = Dict{String, Any}(
        "schema_version" => MANIFEST_SCHEMA_VERSION,
        "provenance" => Dict(
            "script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now())
        ),
        "comparison" => comp,
    )
    name = filename === nothing ? "comparison_" * join(dirtags, "__") * ".toml" : filename
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
    m = Dict{String, Any}("schema_version" => MANIFEST_SCHEMA_VERSION, "provenance" => prov, "outputs" => outs)
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
so the replay contract is enforced at write time, not discovered at replay time. Stamps a
top-level `schema_version = MANIFEST_SCHEMA_VERSION` that readers validate via
[`check_schema_version`](@ref).
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
        "schema_version" => MANIFEST_SCHEMA_VERSION,
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
