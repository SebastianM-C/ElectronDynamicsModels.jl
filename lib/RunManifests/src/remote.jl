# ───────────────────────── dashboard client (read-only browse + lazy data) ──────────────
# Read-only client for the results dashboard's static API: `index.json` (the catalogue the
# dashboard builder emits) and `/data/<uuid>/<basename>` (a plain file server over the
# archive store). Browse campaigns → runs → caches, and lazily fetch cache files into a
# local Scratch.jl store keyed by run uuid. Run products are IMMUTABLE — a uuid's files
# never change — so a downloaded file needs no invalidation and is shared across remotes;
# integrity is checked against the `bytes` the run's `.reduced` marker recorded.
#
# The client prefers the fat-index fields (`dir`, `config`/`laser`/`setup`, `caches`) and
# degrades on older indexes: campaign attribution falls back to sweep cells, and the cache
# list is reconstructed from the run's `data` + derived-plot `data` download links.

const PUBLIC_DASHBOARD_URL = "https://phd.314159265.dev"

"""
    dashboard(host = :public; url = nothing, token = nothing) -> Dashboard

Connect to a results dashboard and fetch its `index.json` catalogue.

  * `dashboard()` — the public dashboard ($(PUBLIC_DASHBOARD_URL)), no auth.
  * `dashboard(:private)` — the private dashboard; the origin comes from
    `ENV["EDM_DASHBOARD_URL"]` (or `url = …`) and the access token from
    `ENV["EDM_DASHBOARD_TOKEN"]` (or `token = …`), sent as the dashboard's
    `research=<token>` cookie.
  * `dashboard(url = "file:///…")` — any origin serving the same layout (used by tests).

The returned [`Dashboard`](@ref) holds the parsed index; everything else
([`campaigns`](@ref), [`campaign`](@ref), [`runs`](@ref), [`caches`](@ref),
[`loadcache`](@ref)) works off it, downloading data files only on demand.
"""
function dashboard(host::Symbol = :public; url = nothing, token = nothing)
    host in (:public, :private) ||
        error("dashboard: unknown host $(repr(host)) — use :public or :private")
    base = url !== nothing ? rstrip(String(url), '/') :
           host === :public ? PUBLIC_DASHBOARD_URL :
           rstrip(get(ENV, "EDM_DASHBOARD_URL") do
               error("dashboard(:private): set EDM_DASHBOARD_URL to the private dashboard " *
                     "origin (or pass url = …)")
           end, '/')
    tok = token !== nothing ? String(token) :
          host === :private ? get(ENV, "EDM_DASHBOARD_TOKEN") do
              error("dashboard(:private): set EDM_DASHBOARD_TOKEN to the dashboard access " *
                    "token (or pass token = …)")
          end : nothing
    headers = tok === nothing ? Pair{String, String}[] : ["Cookie" => "research=$(tok)"]
    buf = IOBuffer()
    _download(base * "/index.json", buf; headers, what = "index.json")
    return Dashboard(base, headers, JSON.parse(String(take!(buf))))
end

"""
    Dashboard

A connected results dashboard: origin, auth headers, and the parsed `index.json`.
Index by campaign dir (`dash["lpwa_boundary"]`), or start from [`campaigns`](@ref).
"""
struct Dashboard
    base::String
    headers::Vector{Pair{String, String}}
    index::Any
end

"""
    RemoteRun

One run entry of a [`Dashboard`](@ref) index, bound to its campaign dir. Index-entry
fields are exposed as properties (`r.id`, `r.label`, `r.params`, `r.commit`, …);
`r.entry` is the raw entry. Data access goes through [`caches`](@ref),
[`cachepath`](@ref), [`loadcache`](@ref), and [`manifest`](@ref).
"""
struct RemoteRun
    dash::Dashboard
    dir::Union{String, Nothing}
    entry::Any
end

function Base.getproperty(r::RemoteRun, s::Symbol)
    s in (:dash, :dir, :entry) && return getfield(r, s)
    e = getfield(r, :entry)
    haskey(e, String(s)) && return e[String(s)]
    error("RemoteRun has no property $s (index entry keys: $(join(keys(e), ", ")))")
end
Base.propertynames(r::RemoteRun) =
    (fieldnames(RemoteRun)..., Symbol.(collect(keys(getfield(r, :entry))))...)

"""
    Campaign

All runs of one campaign dir (plus the sweeps detected there). Select runs with
[`runs`](@ref)`(c; param = value, …)`, by uuid prefix / gallery label via `c["6d8652a2"]`
/ `c["#003"]`, or list them all with `runs(c)`.
"""
struct Campaign
    dash::Dashboard
    dir::String
    runs::Vector{RemoteRun}
    sweeps::Vector{Any}
end

# ── index navigation ──────────────────────────────────────────────────────────

# run id => campaign dir. Fat indexes carry `dir` on every run; legacy indexes only
# attribute runs that appear in a sweep's cells.
function _dir_by_run(index)
    byid = Dict{String, String}()
    for s in get(index, "sweeps", Any[]), c in get(s, "cells", Any[])
        byid[c["run"]] = s["dir"]
    end
    for r in get(index, "runs", Any[])
        d = get(r, "dir", nothing)
        d === nothing || (byid[r["id"]] = d)
    end
    return byid
end

"""
    campaigns(dash::Dashboard) -> Vector{String}

The campaign dir names the index knows about (sorted). On a legacy index (no per-run
`dir` field) standalone runs are unattributable — only campaigns visible through sweeps
are listed.
"""
campaigns(dash::Dashboard) = sort(unique(values(_dir_by_run(dash.index))))

"""
    campaign(dash::Dashboard, dir) -> Campaign

The [`Campaign`](@ref) view of one campaign dir (see [`campaigns`](@ref) for the names).
`dash[dir]` is equivalent.
"""
function campaign(dash::Dashboard, dir::AbstractString)
    byid = _dir_by_run(dash.index)
    rs = [RemoteRun(dash, String(dir), r) for r in get(dash.index, "runs", Any[])
          if get(byid, r["id"], nothing) == dir]
    isempty(rs) && error("campaign: no runs under $(repr(String(dir))) — " *
                         "campaigns(dash) lists what this index has")
    sw = Any[s for s in get(dash.index, "sweeps", Any[]) if s["dir"] == dir]
    return Campaign(dash, String(dir), rs, sw)
end
Base.getindex(dash::Dashboard, dir::AbstractString) = campaign(dash, dir)

"""
    runs(c::Campaign; params...) -> Vector{RemoteRun}

The campaign's runs, optionally filtered on canonical index params:
`runs(c; a0 = 0.1, N_electrons = 2000)`. Numeric values match with a small relative
tolerance (TOML/JSON float round-trip safe); everything else compares as strings.
"""
function runs(c::Campaign; params...)
    out = c.runs
    for (k, v) in pairs(params)
        out = filter(r -> _param_eq(get(r.params, String(k), nothing), v), out)
    end
    return out
end

_param_eq(::Nothing, v) = false
_param_eq(x::Real, v::Real) = x == v || isapprox(Float64(x), Float64(v); rtol = 1e-8)
_param_eq(x, v) = string(x) == string(v)

"""
    sweeps(c::Campaign) -> Vector

The index's sweep entries detected in this campaign dir (axes, cells, coverage) — the
coordinate system for picking runs by physics values via [`runs`](@ref) filters.
"""
sweeps(c::Campaign) = c.sweeps

function Base.getindex(c::Campaign, key::AbstractString)
    hits = [r for r in c.runs
            if startswith(r.id, key) || r.label == key || r.label == "#" * key]
    isempty(hits) && error("campaign $(c.dir): no run matches $(repr(key)) " *
                           "(uuid prefix or gallery label)")
    length(hits) > 1 && error("campaign $(c.dir): $(length(hits)) runs match $(repr(key)): " *
                              join((r.id for r in hits), ", "))
    return hits[1]
end

"""
    manifest(r::RemoteRun) -> NamedTuple

The run's provenance + raw manifest sections as recorded in the index:
`(; config, laser, setup, provenance)`. On a legacy index the sections are `nothing`
(only the canonical `params` existed); `provenance` is always assembled from the entry's
flat fields. The typed spec API will grow on top of this.
"""
function manifest(r::RemoteRun)
    e = getfield(r, :entry)
    g(k) = get(e, k, nothing)
    return (;
        config = g("config"), laser = g("laser"), setup = g("setup"),
        provenance = (;
            run_id = e["id"], script = g("script"), repo_commit = g("commit"),
            repo_dirty = g("repo_dirty"), host = g("host"), backend = g("backend"),
            provider = g("provider"), julia_version = g("julia_version"),
            timestamp = g("timestamp"),
        ),
    )
end

# ── caches: browse + lazy fetch ───────────────────────────────────────────────

# Cache key from a product basename: `hmaps_<uuid>.jls` ⇒ :hmaps, `powspec_…` ⇒ :powspec,
# `gammatau_…`/`ic_…` likewise; the cube (`field_…`) ⇒ :field; `*_obscache.jls` ⇒ :obscache.
function _cache_key(file::AbstractString, run_id::AbstractString)
    stem = first(splitext(basename(file)))
    endswith(stem, "_obscache") && return :obscache
    startswith(stem, "field_") && return :field
    i = findfirst("_" * run_id, stem)
    i === nothing || return Symbol(stem[1:first(i)-1])
    return Symbol(stem)
end

"""
    caches(r::RemoteRun) -> Vector{NamedTuple}

The run's downloadable data files as `(; key, file, bytes, url)` rows. On a fat index
this is the run's `.reduced` reduction enumeration (`bytes` from the durability marker;
`url === nothing` means recorded but not yet published) plus the raw datafile; on a
legacy index it is reconstructed from the run's `data` and derived-plot `data` links
(`bytes` unknown). `key` is a short symbol (`:hmaps`, `:powspec`, `:gammatau`, `:ic`,
`:field`, …) for [`cachepath`](@ref)/[`loadcache`](@ref).
"""
function caches(r::RemoteRun)
    e = getfield(r, :entry)
    out = NamedTuple[]
    seen = Set{String}()
    function add!(file, bytes, url)
        bn = String(file)
        bn in seen && return
        push!(seen, bn)
        push!(out, (key = _cache_key(bn, r.id), file = bn, bytes = bytes,
            url = url === nothing ? nothing : String(url)))
    end
    for ce in get(e, "caches", Any[])
        add!(ce["file"], get(ce, "bytes", nothing), get(ce, "url", nothing))
    end
    # The raw datafile (`data`: the cube, or its hmaps fallback) rides along; on a legacy
    # index (no `caches`) the derived-plot entries' `data` links rebuild the rest.
    d = get(e, "data", nothing)
    d === nothing || add!(split(d, '/')[end], nothing, d)
    for (_, pe) in pairs(get(e, "plots", Dict{String, Any}()))
        pe isa AbstractDict || continue
        for v in (haskey(pe, "values") ? pe["values"] : Any[pe])
            v isa AbstractDict || continue
            u = get(v, "data", nothing)
            u === nothing || add!(split(u, '/')[end], nothing, u)
        end
    end
    return out
end

"""
    cachepath(r::RemoteRun, key) -> String

Local path of one cache of `r`, downloading it into the scratch data store on first use
(and re-downloading if the stored copy's size doesn't match the recorded `bytes`).
`key` is a [`caches`](@ref) key symbol (`:hmaps`, `:powspec`, …) or an exact file
basename. Files land under `data_store_dir()/<run uuid>/` and stay there — run products
are immutable, so there is nothing to invalidate; reclaim space with
[`clear_data_store!`](@ref).
"""
function cachepath(r::RemoteRun, key)
    cs = caches(r)
    hits = key isa AbstractString ? filter(c -> c.file == key, cs) :
           filter(c -> c.key === Symbol(key), cs)
    isempty(hits) &&
        error("run $(first(r.id, 8)): no cache $(repr(key)); available: " *
              (isempty(cs) ? "(none)" : join(unique(c.key for c in cs), ", ")))
    length(hits) > 1 &&
        error("run $(first(r.id, 8)): $(repr(key)) is ambiguous — pass one of: " *
              join((c.file for c in hits), ", "))
    c = only(hits)
    c.url === nothing &&
        error("run $(first(r.id, 8)): cache $(c.file) is recorded but not published yet " *
              "(no download URL on this index)")
    dest = joinpath(data_store_dir(), r.id, c.file)
    isfile(dest) && (c.bytes === nothing || filesize(dest) == c.bytes) && return dest
    mkpath(dirname(dest))
    url = occursin("://", c.url) ? c.url : getfield(r, :dash).base * c.url
    sz = c.bytes === nothing ? "" : " ($(round(c.bytes / 2^20; digits = 1)) MiB)"
    @info "fetching $(c.file)$sz → $(dirname(dest))"
    tmp = dest * ".part"
    _download(url, tmp; headers = getfield(r, :dash).headers, what = c.file)
    if c.bytes !== nothing && filesize(tmp) != c.bytes
        n = filesize(tmp)
        rm(tmp; force = true)
        error("cache $(c.file): downloaded $n bytes, expected $(c.bytes) — " *
              "partial/corrupt transfer, not kept")
    end
    mv(tmp, dest; force = true)
    return dest
end

"""
    loadcache(r::RemoteRun, key)

Deserialize one cache of `r` (see [`cachepath`](@ref) for `key` and the lazy download).
Caches are `Serialization.serialize`d NamedTuples of plain arrays, written by the
reduction scripts — see each producer for the payload layout.
"""
loadcache(r::RemoteRun, key) = Serialization.deserialize(cachepath(r, key))

"""
    data_store_dir() -> String

The local Scratch.jl directory holding downloaded cache files (`<store>/<uuid>/<file>`).
Managed by Julia's scratch-space GC; safe to delete at any time.
"""
data_store_dir() = Scratch.@get_scratch!("dashboard-data")

"""
    clear_data_store!() -> String

Delete every downloaded cache file (the whole [`data_store_dir`](@ref) contents).
"""
function clear_data_store!()
    dir = data_store_dir()
    for entry in readdir(dir; join = true)
        rm(entry; recursive = true, force = true)
    end
    return dir
end

# ── plumbing ──────────────────────────────────────────────────────────────────

function _download(url, output; headers, what)
    try
        Downloads.download(url, output; headers)
    catch err
        if err isa Downloads.RequestError
            resp = err.response
            status = resp === nothing ? nothing : resp.status
            status == 403 && error("$(what): HTTP 403 from $url — wrong or missing access " *
                                   "token? (EDM_DASHBOARD_TOKEN / dashboard(:private))")
            status === nothing ||
                error("$(what): HTTP $status from $url")
        end
        rethrow()
    end
end

function Base.show(io::IO, dash::Dashboard)
    print(io, "Dashboard(", repr(dash.base), ", ",
        length(get(dash.index, "runs", Any[])), " runs, ",
        length(campaigns(dash)), " campaigns)")
end

function Base.show(io::IO, c::Campaign)
    print(io, "Campaign(", repr(c.dir), ", ", length(c.runs), " runs, ",
        length(c.sweeps), " sweeps)")
end

function Base.show(io::IO, r::RemoteRun)
    e = getfield(r, :entry)
    ps = get(e, "params", Dict{String, Any}())
    ks = sort(collect(keys(ps)))
    sel = join(("$k=$(ps[k])" for k in ks[1:min(end, 4)]), ", ")
    length(ks) > 4 && (sel *= ", …")
    print(io, "RemoteRun(", get(e, "label", "?"), " ", first(e["id"], 8), "…",
        isempty(sel) ? "" : "; " * sel, ")")
end
