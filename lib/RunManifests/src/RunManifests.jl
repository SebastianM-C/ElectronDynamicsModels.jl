"""
    RunManifests

In-repo support package for ElectronDynamicsModels research runs. It centralises
everything reproducibility-related:

  * **git provenance** — `git_state`, the standard solver `run_provenance` block;
  * **a clean-tree guard** — `assert_committed`, so a run is never produced from
    uncommitted code its `repo_commit` cannot reproduce;
  * **manifest I/O** — `write_run_manifest` / `write_derived` / `write_comparison` /
    `write_summary` (the `run_*.toml`, `derived_*.toml`, `comparison_*.toml`, and
    `summary_*.toml` the results dashboard consumes) and the `find_parent_*` readers;
  * **the replay seed** — `run_spec_from_manifest`, the inverse of how solver scripts
    read `ENV`, used by the reproduce/sweep launcher;
  * **the dashboard client** — `dashboard` / `sweeps` / `sweep` / `runs` / `caches` /
    `loadcache` / `summaries` / `comparisons` (src/remote.jl): read-only browse of a
    published results index and lazy, integrity-checked download of the data files into
    a local Scratch store — post-hoc analysis without the results dir. ("Sweep" is the
    catalogue noun by design — see the dashboard repo's sweep_declarations_design.md.)

The solver/plot scripts get these via `using RunManifests`; `scripts/manifest.jl` is a
thin back-compat shim that re-exports them for scripts that still `include` it.
"""
module RunManifests

using TOML
using Dates
import Downloads
import JSON
import Scratch
import Serialization

export git_state, assert_committed, run_provenance, run_spec_from_manifest, expand_sweep
export ThomsonScatteringSpec, load_spec, write_spec, spec_env, spec_from_manifest, config_dict
export find_parent_manifest, find_parent_run, spp_from_manifest
export write_derived, write_comparison, write_summary, write_run_manifest, write_solver_manifest, REQUIRED_CONFIG_KEYS
export write_sweep_declaration, read_sweep_declarations
export record_reduction!
export units_section, units_from_manifest
export MANIFEST_SCHEMA_VERSION, manifest_schema_version, check_schema_version
export Dashboard, Sweep, RemoteRun
export dashboard, sweep, sweeps, runs, manifest, summaries, comparisons
export caches, cachepath, loadcache, datapath, loaddata, data_store_dir, clear_data_store!

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

# Dirty-tree marker for the same repo — write_run_manifest/write_derived record it so
# standalone analysis nodes carry the SAME honesty flag as solver runs (a verifier node
# produced from an uncommitted tree used to look clean while the solver runs beside it
# were repo_dirty = true; spotted 2026-07-30 during the γ=1 crosscheck).
function _script_repo_dirty()
    return try
        !isempty(readchomp(Cmd(["git", "-C", @__DIR__, "status", "--porcelain"])))
    catch
        false
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

Timestamps: `timestamp` is machine-LOCAL wall clock (legacy, timezone-less — cloud VMs
stamp UTC, the workstation its own zone) and stays for back-compat; `timestamp_utc`
(with an explicit `Z`) is the unambiguous instant — consumers that compare times across
machines (e.g. the dashboard's known-issue cutoffs) should prefer it.

The standard solver-run `[provenance]` block. `gpu_device` (e.g. `CUDA.name(CUDA.device())`)
is recorded only when supplied — kept as an argument so this package stays free of any GPU
backend dependency. `sweep_id` (default: `ENV["EDM_SWEEP"]`, as exported by
`orchestration/run_cell.sh`) records the run's sweep-declaration membership — see
[`write_sweep_declaration`](@ref); omitted entirely when unset, so runs outside a declared
sweep look exactly as before.
"""
function run_provenance(;
        run_id, gpu_backend, repo_dir,
        script::AbstractString = abspath(PROGRAM_FILE), gpu_device = nothing,
        sweep_id = nothing
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
        "cloud_provider" => get(ENV, "EDM_CLOUD_PROVIDER", "local"),
        "gpu_backend" => gpu_backend,
        "julia_version" => string(VERSION),
        "timestamp" => string(now()), "timestamp_utc" => string(now(UTC)) * "Z",
    )
    gpu_device === nothing || (prov["gpu_device"] = string(gpu_device))
    sweep = sweep_id === nothing ? get(ENV, "EDM_SWEEP", "") : string(sweep_id)
    isempty(sweep) || (prov["sweep_id"] = sweep)
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
    miss = [k for k in REQUIRED_CONFIG_KEYS if !haskey(cfg, k)]
    isempty(miss) || error(
        "run manifest [config] is missing replay key(s) $(join(miss, ", ")) — " *
            "cannot reproduce this run."
    )
    # One knob ⇄ env mapping for the whole package: the manifest becomes a
    # ThomsonScatteringSpec and the spec emits its transport (spec.jl). Guarded-knob
    # semantics are the spec's nothing-means-script-default contract, so a legacy run
    # still never grows env out of thin air — while knobs the old hand-rolled emission
    # dropped (EDM_SYSTEM, EDM_OMEGA_SCALE, EDM_SCREEN_HALFW, EDM_POL, EDM_GAMMA_EPS)
    # now replay faithfully.
    env = spec_env(spec_from_manifest(manifest))
    env["EDM_GPU_BACKEND"] = string(prov["gpu_backend"])
    # Pre-mode manifests replay as :split — the scripts' default then and now.
    get!(env, "EDM_FIELD_MODE", "split")
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
        "provenance" => Dict{String, Any}(
            "script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now()), "timestamp_utc" => string(now(UTC)) * "Z",
            (_script_repo_dirty() ? ("repo_dirty" => true,) : ())...,
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

"""
    record_reduction!(dir, run_id, file) -> String

Append `file` (a reduction product — a full path, or a basename resolved in `dir`) to the run's
reduction staging marker `<run_id>.reduced.partial` in `dir`, recording its current byte size.
The first call writes the header (`run_id`/`reduced_at`/`host`/`reduce_commit` + an empty
`reduction` array); later calls append. The orchestration renames `.partial` →
`<run_id>.reduced` atomically once every reducer for the cell succeeds, so the marker's PRESENCE
is the drainer's "reduce complete" handshake and its CONTENTS enumerate the caches that must land
durably on the archive store for the run to be re-renderable WITHOUT the cube (publish-autonomy).

Invariant: every product a drain-path reducer emits must be re-renderable from a cache recorded
here. A plot rendered directly from the cube with NO cache is invisible to this marker and breaks
publish-autonomy — add a cache instead of relying on the cube (which only exists where it was produced/archived).
Today's drain-path reducers (harmonic_products, plot_screen_observables) cache every product.

Re-reduce semantics: when no `.partial` exists but a finalized `<run_id>.reduced` does, the
staging marker is SEEDED from the final marker (header restamped to the current pass), and an
entry with the same `file` basename REPLACES the old one instead of duplicating it. So a
re-reduce refreshes byte sizes for everything it touches while entries it does not touch
survive — the VPS status collector byte-compares marker entries against the archive store, and
a stale size would flip an actually-complete run back to "reduce-pending" (bit the
rest_departure_bridge_refix re-reduce, 2026-08-30).
"""
function record_reduction!(dir::AbstractString, run_id, file::AbstractString)
    path = joinpath(dir, "$(run_id).reduced.partial")
    final = joinpath(dir, "$(run_id).reduced")
    m = if isfile(path)
        TOML.parsefile(path)
    elseif isfile(final)
        # Re-reduce: seed from the finalized marker so untouched entries survive, but the
        # header reflects THIS pass (a re-reduce is a new reduction event, new commit and all).
        prev = TOML.parsefile(final)
        prev["reduced_at"] = string(now())
        prev["host"] = gethostname()
        prev["reduce_commit"] = _script_repo_commit()
        prev
    else
        Dict{String, Any}(
            "run_id" => string(run_id),
            "reduced_at" => string(now()),
            "host" => gethostname(),
            "reduce_commit" => _script_repo_commit(),
            "reduction" => Dict{String, Any}[],
        )
    end
    full = isfile(file) ? file : joinpath(dir, file)
    base = basename(file)
    filter!(e -> e["file"] != base, m["reduction"])   # replace a re-reduced file, don't duplicate
    push!(m["reduction"], Dict{String, Any}("file" => base, "bytes" => filesize(full)))
    open(io -> TOML.print(io, m; sorted = true), path, "w")
    return path
end

# Normalise one comparison side spec to (label, dir, script, where). Accepts a NamedTuple
# `(; label, dir[, script][, var"where"])`, a `Dict` (keys "label"/"dir"/"script"/"where"),
# or a `(label, dir[, script])` tuple. `where` — a Dict of canonical-param => value
# constraints — selects WHICH runs of `dir` form this side (for same-dir sides).
function _side_fields(s)
    s isa AbstractDict &&
        return (s["label"], s["dir"], get(s, "script", nothing), get(s, "where", nothing))
    s isa NamedTuple && return (s.label, s.dir, hasproperty(s, :script) ? s.script : nothing,
        hasproperty(s, :where) ? getproperty(s, :where) : nothing)
    s isa Tuple && return (s[1], s[2], length(s) >= 3 ? s[3] : nothing, nothing)
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
than one sweep (e.g. an LPWA and a Thomson run in the same folder). A side may also carry a
`where` Dict of canonical-param => value constraints (Dict key `"where"`, or NT field
`var"where"`) restricting the side to the matching runs of its dir — required when BOTH sides
live in one campaign dir (e.g. two retarded-time kernels swept in one campaign). `differs` is a
free-form label for what distinguishes the sides (e.g. `"method"`); `along` names the shared
swept axis to match on — **omit it** to let the dashboard infer the sides' common axis (the
usual case, since a per-pair caller can't see what the whole campaign sweeps).

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
        sl, sd, sc, sw = _side_fields(s)
        d = Dict{String, Any}("label" => string(sl), "dir" => string(sd))
        sc === nothing || (d["script"] = string(sc))
        sw === nothing || isempty(sw) ||
            (d["where"] = Dict{String, Any}(string(k) => v for (k, v) in pairs(sw)))
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
            "host" => gethostname(), "timestamp" => string(now()), "timestamp_utc" => string(now(UTC)) * "Z"
        ),
        "comparison" => comp,
    )
    name = filename === nothing ? "comparison_" * join(dirtags, "__") * ".toml" : filename
    open(io -> TOML.print(io, m; sorted = true), joinpath(dir, name), "w")
    return joinpath(dir, name)
end

"""
    write_summary(dir; kind, label, plot, run_ids, axis=nothing, datafile=nothing,
                  setup=Dict(), plot_params=Dict(), description=nothing)

Write a `[summary]` sidecar TOML into `dir` — a CAMPAIGN-level summary artifact, the third
artifact class next to [`write_derived`](@ref) (one run's own post-processing) and
[`write_comparison`](@ref) (a declared relationship between sweeps). A summary is one plot
whose x-axis is the campaign's sweep axis (a convergence curve along `EDM_INTERP_SAVEAT`,
a rel-L2 ∝ a₀²γ scaling law, per-cell profile overlays), so the dashboard attaches it to
the SWEEP card detected in `dir`, not to a single run.

`run_ids` is the member run id(s) the summary was computed from (written as `depends_on`);
the dashboard binds the sidecar to the sweep in the same dir whose members intersect it —
matched against the raw pre-view-collapse run list, so a collapsed view-variant member
(e.g. the LL half of a classical|LL pair) still counts. `plot` is a PNG basename in `dir`;
`axis` optionally names the swept canonical param the plot is along (e.g. `"a0"`,
`"interp_saveat"`). `setup` / `plot_params` / `description` behave exactly as in
[`write_derived`](@ref): `setup` keys that vary across same-kind summary sidecars become a
value picker, `plot_params` is display-only, `description` is markdown + `\$…\$` KaTeX.

Filename: `summary_<kind>[_<setup-vals>]_<id8>-<n>.toml` — `<id8>` is the first 8 chars of
the lexicographically first run id and `<n>` the member count, deterministic for a fixed
member set so a re-run overwrites its own sidecar instead of accumulating copies.
"""
function write_summary(
        dir::AbstractString; kind, label, plot, run_ids,
        axis = nothing, datafile = nothing, setup = Dict(), plot_params = Dict(),
        description = nothing
    )
    deps = run_ids isa AbstractString ? [string(run_ids)] : [string(x) for x in run_ids]
    isempty(deps) && error("write_summary: `run_ids` must name at least one member run")
    d = Dict{String, Any}("kind" => kind, "label" => label, "depends_on" => deps, "plot" => plot)
    axis === nothing || (d["axis"] = string(axis))
    datafile === nothing || (d["datafile"] = datafile)
    # `description`: markdown + \$…\$ LaTeX, rendered (KaTeX) in the dashboard plot modal.
    description === nothing || (d["description"] = description)
    m = Dict{String, Any}(
        "schema_version" => MANIFEST_SCHEMA_VERSION,
        "provenance" => Dict(
            "script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now()), "timestamp_utc" => string(now(UTC)) * "Z"
        ),
        "summary" => d,
    )
    isempty(setup) || (m["setup"] = Dict{String, Any}(string(k) => v for (k, v) in setup))
    isempty(plot_params) || (m["plot_params"] = Dict{String, Any}(string(k) => v for (k, v) in plot_params))
    suffix = isempty(setup) ? "" : "_" * join(string.(values(setup)), "-")   # filename: setup keys only
    idtag = string(first(minimum(deps), 8), "-", length(deps))
    name = "summary_$(kind)$(suffix)_$(idtag).toml"
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
        "host" => gethostname(), "timestamp" => string(now()), "timestamp_utc" => string(now(UTC)) * "Z"
    )
    _script_repo_dirty() && (prov["repo_dirty"] = true)
    derived_from === nothing || (prov["derived_from"] = derived_from)
    # Analysis nodes launched under a declared sweep (EDM_SWEEP in scope) join it like
    # solver runs do — same membership channel, same builder hook.
    sweep = get(ENV, "EDM_SWEEP", "")
    isempty(sweep) || (prov["sweep_id"] = sweep)
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

# ───────────────────────── [sweep] — launch-time sweep declarations ─────────────────────
# A sweep declaration records LAUNCH intent — which [config] knobs a campaign dir sweeps —
# so the dashboard reads structure instead of re-detecting it from outcomes (the full
# design lives in the dashboard repo: docs/sweep_declarations_design.md). Membership rides
# provenance.sweep_id (stamped from EDM_SWEEP by run_cell.sh); runs without it fall to the
# dir's default declaration for their script. The declared name is an IDENTIFIER: the
# dashboard's stable sweep id is "<dir>:<name>", so renaming is a breaking act — `label`
# exists for display.

const SWEEP_NAME_RE = r"^[a-z0-9][a-z0-9_-]*$"
const SWEEP_DESIGNS = ("grid", "oat")

"""
    write_sweep_declaration(dir; name, script, axes, design = "grid", hub = nothing,
                            label = nothing) -> path

Write the `sweep_<name>.toml` declaration into `dir` — the launch-time statement of a
campaign's structure that the dashboard prefers over sweep auto-detection.

  * `name` — a slug (`[a-z0-9_-]`, leading alphanumeric); becomes the stable dashboard id
    `"<dir>:<name>"`.
  * `script` — solver script basename; binds the default-membership rule for runs without
    a `provenance.sweep_id`.
  * `axes` — the swept **manifest `[config]` keys** (`"gamma"`, `"a0"`, …), NOT `EDM_*`
    env names and NOT dashboard display names (the builder translates via its PARAM_SPEC).
    `[]` declares an unstructured group (a card with only an extras line). Axis *values*
    are never declared — they are recovered from the member runs, so intent cannot drift
    from reality.
  * `design` — `"grid"` (default; axes span a product) or `"oat"` (axes are
    one-at-a-time arms sharing a hub cell).
  * `hub` — optional run uuid overriding the inferred OAT hub.
  * `label` — optional display label (defaults to `name` dashboard-side).

Idempotent: rewrites `sweep_<name>.toml` in place (`run_cell.sh` calls this once per
campaign, gated on the file's absence). Stamps `schema_version` like the other writers.
"""
function write_sweep_declaration(
        dir::AbstractString; name, script, axes,
        design = "grid", hub = nothing, label = nothing
    )
    n = string(name)
    occursin(SWEEP_NAME_RE, n) || error(
        "write_sweep_declaration: name $(repr(n)) is not a slug ([a-z0-9_-], leading " *
            "alphanumeric) — it becomes the stable dashboard id \"<dir>:$n\"."
    )
    string(design) in SWEEP_DESIGNS || error(
        "write_sweep_declaration: unknown design $(repr(string(design))) — use one of " *
            join(SWEEP_DESIGNS, ", ") * "."
    )
    ax = String[string(a) for a in axes]
    for a in ax
        startswith(a, "EDM_") && error(
            "write_sweep_declaration: axis $(repr(a)) uses the env spelling — axes name " *
                "manifest [config] keys (e.g. \"gamma\", not \"EDM_GAMMA\")."
        )
    end
    s = Dict{String, Any}(
        "name" => n, "script" => string(script), "axes" => ax, "design" => string(design)
    )
    hub === nothing || (s["hub"] = string(hub))
    label === nothing || (s["label"] = string(label))
    m = Dict{String, Any}(
        "schema_version" => MANIFEST_SCHEMA_VERSION,
        "provenance" => Dict{String, Any}(
            "script" => basename(PROGRAM_FILE), "repo_commit" => _script_repo_commit(),
            "host" => gethostname(), "timestamp" => string(now()), "timestamp_utc" => string(now(UTC)) * "Z"
        ),
        "sweep" => s,
    )
    path = joinpath(dir, "sweep_$(n).toml")
    open(io -> TOML.print(io, m; sorted = true), path, "w")
    return path
end

"""
    read_sweep_declarations(dir) -> Vector{NamedTuple}

Parse every `sweep_*.toml` declaration in `dir` into
`(; name, script, axes, design, hub, label, path)` rows, sorted by filename. TOMLs
without a `[sweep]` table are skipped; a missing required key or a **duplicate name**
errors loudly — declarations decide card structure, so ambiguity here must not pass.
Returns an empty vector for a nonexistent `dir`.
"""
function read_sweep_declarations(dir::AbstractString)
    out = NamedTuple[]
    isdir(dir) || return out
    seen = Set{String}()
    for f in sort(readdir(dir))
        (startswith(f, "sweep_") && endswith(f, ".toml")) || continue
        m = TOML.parsefile(joinpath(dir, f))
        haskey(m, "sweep") || continue
        check_schema_version(m; source = f)
        s = m["sweep"]
        for k in ("name", "script", "axes")
            haskey(s, k) || error("$f: [sweep] is missing $(repr(k))")
        end
        n = string(s["name"])
        n in seen && error(
            "$dir: duplicate sweep declaration name $(repr(n)) — names decide card " *
                "structure; merge the declarations or rename one."
        )
        push!(seen, n)
        push!(out, (;
            name = n, script = string(s["script"]), axes = String.(s["axes"]),
            design = string(get(s, "design", "grid")), hub = get(s, "hub", nothing),
            label = get(s, "label", nothing), path = joinpath(dir, f),
        ))
    end
    return out
end

# ───────────────────────── [units] — display-unit declarations ─────────────────────────
# The manifest's raw numbers stay in the solver's NATIVE units; [units] is a declaration
# layer beside them so consumers (chip renderers, the dashboard) can label axes without
# run-type-specific knowledge (the `backscatter_n0` special case, "is ω₁ scaled?", …).
# Additive optional section — no MANIFEST_SCHEMA_VERSION bump (stale readers ignore it).

# Speed of light in Hartree atomic units — used only to SYNTHESIZE [units] for legacy
# manifests (all of which are :atomic runs); new manifests carry their scales explicitly.
const C_HARTREE = 137.03599908330932

"""
    units_section(ω, λ, w₀; system = "hartree_atomic", n0 = 1, omega_scale = 1.0,
        transverse_length = "w0", extra_defs...) -> Dict

Build the `[units]` manifest section: `defs` names scales (each a `value` in raw solver
units + a `tex` display label); `preferred` picks which named scale each quantity kind is
displayed in (`frequency`, `transverse_length`, and `wavelength_display = "si"` — the
raw→SI anchor is fixed by `system`, e.g. hartree_atomic lengths are Bohr radii
a₀ = 0.052917721 nm, so no numeric anchor is stored).

Always defines `omega_laser`/`lambda_laser`/`w0` from this run's carrier. `n0 > 1`
(backscatter runs) adds `omega_bs = n0·ω` and prefers it for frequencies — the [units]
generalization of the legacy `backscatter_n0` key. `omega_scale ≠ 1` (Doppler-equivalent
runs) adds the unscaled `omega_lab`/`lambda_lab` reference scales. Additional named scales
append via keyword pairs: `moat = (value = …, tex = "r_F")`.

Raw values in [config]/[laser]/[setup] are NEVER converted. Read back (with the legacy
fallback) via [`units_from_manifest`](@ref).
"""
function units_section(ω::Real, λ::Real, w₀::Real; system = "hartree_atomic",
        n0::Integer = 1, omega_scale::Real = 1.0, transverse_length = "w0", extra_defs...)
    def(value, tex) = Dict{String, Any}("value" => Float64(value), "tex" => String(tex))
    defs = Dict{String, Any}(
        "omega_laser" => def(ω, "ω₁"),
        "lambda_laser" => def(λ, "λ₁"),
        "w0" => def(w₀, "w₀"),
    )
    frequency = "omega_laser"
    if n0 > 1
        defs["omega_bs"] = def(n0 * ω, "ω_bs")
        frequency = "omega_bs"
    end
    if !isapprox(omega_scale, 1.0)
        defs["omega_lab"] = def(ω / omega_scale, "ω₀")
        defs["lambda_lab"] = def(λ * omega_scale, "λ₀")
    end
    for (k, v) in pairs(extra_defs)
        defs[string(k)] = def(v.value, v.tex)
    end
    return Dict{String, Any}(
        "system" => String(system),
        "defs" => defs,
        "preferred" => Dict{String, Any}(
            "frequency" => frequency,
            "transverse_length" => String(transverse_length),
            "wavelength_display" => "si",
        ),
    )
end

"""
    units_from_manifest(m::AbstractDict) -> (; system, defs, preferred, n0)

Read a manifest's `[units]` section; for manifests that predate it, synthesize the same
structure from what they do record (`[laser] wavelength/w0`, `[config] backscatter_n0/
omega_scale`) so consumers have ONE code path. `n0` is the derived integer ratio
preferred-frequency-scale / ω₁ (1 when frequencies are ω₁-preferred) — exactly the legacy
`backscatter_n0` contract the powspec axis and harmonic-map labels consume.
"""
function units_from_manifest(m::AbstractDict)
    u = get(m, "units", nothing)
    if u === nothing
        las = get(m, "laser", Dict{String, Any}())
        cfg = get(m, "config", Dict{String, Any}())
        λ = Float64(get(las, "wavelength", NaN))
        u = units_section(
            2π * C_HARTREE / λ, λ, Float64(get(las, "w0", NaN));
            n0 = round(Int, get(cfg, "backscatter_n0", 1)),
            omega_scale = Float64(get(cfg, "omega_scale", 1.0)),
        )
    end
    defs, pref = u["defs"], u["preferred"]
    fscale = get(defs, get(pref, "frequency", "omega_laser"), defs["omega_laser"])
    n0 = round(Int, Float64(fscale["value"]) / Float64(defs["omega_laser"]["value"]))
    return (; system = u["system"], defs, preferred = pref, n0)
end

include("spec.jl")
include("remote.jl")

end # module RunManifests
