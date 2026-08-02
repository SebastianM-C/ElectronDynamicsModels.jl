# ───────────────────────── ThomsonScatteringSpec — the typed run spec ────────────────────
# The shared physical + numerical STARTING POINT of every run: all scripts (thomson_
# scattering / lpwa / inverse_thomson_scattering / analyze_trajectories) are operations
# applied to the same scattering problem, so one spec type serves authoring (campaign
# recipes), provenance (the manifest [config] echo), and analysis (the dashboard client).
# The executor is deliberately NOT part of the spec — the same spec can be run by the
# numeric solver or the LPWA analytic path (that's what the clincher comparisons do);
# `SCRIPT` stays orchestration, and provenance records which one ran.
#
# Field vocabulary = manifest [config] keys (the same vocabulary sweep declarations use),
# NOT `EDM_*` env spellings. `nothing` means "the script decides": defaults — including
# computed ones like the inverse script's γ-derived N_samples — stay script-side, and the
# manifest records the RESOLVED values, exactly as before. Knobs without a field (script-
# owned keys like `scattering`, analysis-only `err_*`, derived `backscatter_n0`) ride the
# `extra` dict, so no context is ever dropped.

Base.@kwdef struct ThomsonScatteringSpec
    initial_phase::Union{Nothing, Float64} = nothing
    a0::Union{Nothing, Float64} = nothing
    Nx::Union{Nothing, Int} = nothing
    N::Union{Nothing, Int} = nothing
    N_samples::Union{Nothing, Int} = nothing
    samples_per_period::Union{Nothing, Int} = nothing
    n_substeps::Union{Nothing, Int} = nothing
    sync_per_electron::Union{Nothing, Bool} = nothing
    mode::Union{Nothing, String} = nothing              # split | total
    accumulation_alg::Union{Nothing, String} = nothing  # canonical: GPUKernelRK4 | GPUKernelNewton
    newton_iters::Union{Nothing, Int} = nothing
    reltol::Union{Nothing, Float64} = nothing
    abstol::Union{Nothing, Float64} = nothing           # nothing ⇒ the script's abserr(a0)
    interp_saveat::Union{Nothing, String} = nothing     # "adaptive" | knots-per-period as string
    pol::Union{Nothing, String} = nothing
    omega_scale::Union{Nothing, Float64} = nothing
    screen_halfw::Union{Nothing, Float64} = nothing
    gamma::Union{Nothing, Float64} = nothing
    gamma_eps::Union{Nothing, Float64} = nothing        # ε = γ−1; wins over gamma at emission
    system::Union{Nothing, String} = nothing            # classical | ll
    window::Union{Nothing, String} = nothing            # full | narrow
    screen_hw_w0::Union{Nothing, Float64} = nothing
    tspan_tau::Union{Nothing, Float64} = nothing
    window_lead::Union{Nothing, Float64} = nothing
    window_tail::Union{Nothing, Float64} = nothing
    harmonics::Union{Nothing, Vector{Float64}} = nothing
    bunch_nb::Union{Nothing, Int} = nothing
    bunch_l::Union{Nothing, Int} = nothing
    bunch_chirp::Union{Nothing, Float64} = nothing
    extra::Dict{String, Any} = Dict{String, Any}()
end

# field ⇄ env transport, for the straightforward knobs. The special cases — abstol's
# ""-means-derived, interp_saveat's adaptive skip, the accumulation_alg value map + legacy
# alias, the γ/ε exclusivity — are handled explicitly in load_spec/spec_env below.
const _SPEC_ENV = (
    (:initial_phase, "EDM_INITIAL_PHASE"), (:a0, "EDM_A0"), (:Nx, "EDM_NX"),
    (:N, "EDM_N"), (:N_samples, "EDM_NSAMPLES"), (:samples_per_period, "EDM_SPP"),
    (:n_substeps, "EDM_NSUBSTEPS"), (:sync_per_electron, "EDM_SYNC_PER_ELECTRON"),
    (:mode, "EDM_FIELD_MODE"), (:newton_iters, "EDM_NEWTON_ITERS"),
    (:reltol, "EDM_RELTOL"), (:abstol, "EDM_ABSTOL"),
    (:interp_saveat, "EDM_INTERP_SAVEAT"),
    (:pol, "EDM_POL"), (:omega_scale, "EDM_OMEGA_SCALE"),
    (:screen_halfw, "EDM_SCREEN_HALFW"), (:gamma, "EDM_GAMMA"),
    (:gamma_eps, "EDM_GAMMA_EPS"), (:system, "EDM_SYSTEM"), (:window, "EDM_WINDOW"),
    (:screen_hw_w0, "EDM_SCREEN_HW"), (:tspan_tau, "EDM_TSPAN_TAU"),
    (:window_lead, "EDM_WINDOW_LEAD"), (:window_tail, "EDM_WINDOW_TAIL"),
    (:harmonics, "EDM_HARMONICS"), (:bunch_nb, "EDM_BUNCH_NB"),
    (:bunch_l, "EDM_BUNCH_L"), (:bunch_chirp, "EDM_BUNCH_CHIRP"),
)

const _ALG_TO_ENV = Dict("GPUKernelRK4" => "rk4", "GPUKernelNewton" => "newton")
const _ENV_TO_ALG = Dict(v => k for (k, v) in _ALG_TO_ENV)

_spec_fieldtype(f::Symbol) = fieldtype(ThomsonScatteringSpec, f)
_base_type(::Type{Union{Nothing, T}}) where {T} = T

# Coerce a TOML/manifest value into a field's base type (Int → Float64, Vector{Int} →
# Vector{Float64}, …). Loud on genuinely wrong shapes — a spec typo must not run.
_coerce(::Type{Float64}, v::Real) = Float64(v)
_coerce(::Type{Int}, v::Integer) = Int(v)
_coerce(::Type{Bool}, v::Bool) = v
_coerce(::Type{Bool}, v::Integer) = v == 1 ? true : v == 0 ? false :
    error("spec: cannot use $v for a Bool field (want true/false or 0/1)")
_coerce(::Type{String}, v) = string(v)
_coerce(::Type{Vector{Float64}}, v::AbstractVector) = Float64[Float64(x) for x in v]
_coerce(::Type{T}, v) where {T} =
    error("spec: cannot use $(repr(v)) ($(typeof(v))) for a $(T) field")

# Parse an env-var string into a field's base type.
_parse_env(::Type{Float64}, s) = parse(Float64, s)
_parse_env(::Type{Int}, s) = parse(Int, s)
_parse_env(::Type{Bool}, s) = parse(Bool, lowercase(s))
_parse_env(::Type{String}, s) = String(s)
_parse_env(::Type{Vector{Float64}}, s) = Float64[parse(Float64, x) for x in split(s, ",")]

# Rebuild a spec with some fields replaced (specs are immutable; `extra` is copied).
function _respec(s::ThomsonScatteringSpec; kw...)
    vals = Dict{Symbol, Any}(f => getfield(s, f) for f in fieldnames(ThomsonScatteringSpec))
    vals[:extra] = copy(s.extra)
    for (k, v) in kw
        haskey(vals, k) || error("spec: unknown field $(repr(k)) — fields use manifest " *
                                 "[config] names (e.g. :gamma, :samples_per_period)")
        vals[k] = v === nothing ? nothing : _coerce(_base_type(_spec_fieldtype(k)), v)
    end
    return ThomsonScatteringSpec(; vals...)
end

"""
    write_spec(path, spec::ThomsonScatteringSpec) -> path

Write a spec TOML: top-level `schema_version` + a `[spec]` table holding every set field
(manifest `[config]` vocabulary) with `extra` entries inlined. The inverse of
[`load_spec`](@ref) without its env layer.
"""
function write_spec(path::AbstractString, spec::ThomsonScatteringSpec)
    s = Dict{String, Any}(string(k) => v for (k, v) in pairs(spec.extra))
    for f in fieldnames(ThomsonScatteringSpec)
        f === :extra && continue
        v = getfield(spec, f)
        v === nothing || (s[string(f)] = v)
    end
    m = Dict{String, Any}("schema_version" => MANIFEST_SCHEMA_VERSION, "spec" => s)
    open(io -> TOML.print(io, m; sorted = true), path, "w")
    return path
end

"""
    load_spec(path = get(ENV, "EDM_SPEC", nothing); env = ENV) -> ThomsonScatteringSpec

The solver-side entry point: read the spec file (skipped when `path === nothing` — the
env-only legacy path) and apply `EDM_*` overrides on top, env winning — the same
last-wins ergonomics `run_cell`'s per-cell overrides rely on. Unknown `[spec]` keys land
in `extra`; unknown `EDM_*` vars are ignored (infra: `EDM_OUTDIR`, `EDM_RUN_TAG`, …).
Env special cases mirror the scripts: `EDM_ABSTOL=""` means "derive from a0" (nothing),
`EDM_INTERP_SAVEAT=""` means adaptive, `EDM_ACCUM_ALG`/legacy `EDM_GPU_SOLVER` carry
`rk4|newton` and store the canonical kernel name.
"""
function load_spec(path = get(ENV, "EDM_SPEC", nothing); env = ENV)
    spec = ThomsonScatteringSpec()
    if path !== nothing
        m = TOML.parsefile(String(path))
        check_schema_version(m; source = basename(String(path)))
        s = get(m, "spec", Dict{String, Any}())
        kw = Dict{Symbol, Any}()
        for (k, v) in s
            f = Symbol(k)
            if f !== :extra && hasfield(ThomsonScatteringSpec, f)
                kw[f] = _coerce(_base_type(_spec_fieldtype(f)), v)
            else
                spec.extra[String(k)] = v
            end
        end
        spec = _respec(spec; kw...)
    end
    kw = Dict{Symbol, Any}()
    for (f, key) in _SPEC_ENV
        haskey(env, key) || continue
        raw = env[key]
        if isempty(raw) && f in (:abstol, :interp_saveat)
            kw[f] = nothing            # the scripts' ""-means-default contract
        else
            kw[f] = _parse_env(_base_type(_spec_fieldtype(f)), raw)
        end
    end
    alg = lowercase(get(env, "EDM_ACCUM_ALG", get(env, "EDM_GPU_SOLVER", "")))
    isempty(alg) || (kw[:accumulation_alg] = get(_ENV_TO_ALG, alg) do
        error("EDM_ACCUM_ALG must be \"rk4\" or \"newton\", got $(repr(alg))")
    end)
    return _respec(spec; kw...)
end

"""
    spec_env(spec::ThomsonScatteringSpec) -> Dict{String,String}

The `EDM_*` environment that transports this spec to the current (env-reading) scripts —
set fields only, so a legacy run never grows env out of thin air. Special cases:
`interp_saveat = "adaptive"` is the scripts' default and is omitted; `accumulation_alg`
emits both the canonical `EDM_ACCUM_ALG` and the legacy `EDM_GPU_SOLVER` alias (pinned
old commits read only the latter); `gamma_eps` wins over `gamma` (the scripts reject
both being set — and ε is the higher-fidelity form near rest). `extra` does not emit:
it has no env transport by construction.
"""
function spec_env(spec::ThomsonScatteringSpec)
    env = Dict{String, String}()
    for (f, key) in _SPEC_ENV
        v = getfield(spec, f)
        v === nothing && continue
        f === :interp_saveat && v == "adaptive" && continue
        f === :gamma && spec.gamma_eps !== nothing && continue   # ε wins; scripts reject both
        # Integral harmonics emit without the ".0" — byte-identical to the old
        # join-the-manifest-Ints emission (display/filenames re-integerize either way).
        env[key] = v isa Vector ?
            join((isinteger(x) ? string(Int(x)) : string(x) for x in v), ",") : string(v)
    end
    if spec.accumulation_alg !== nothing
        alg = get(_ALG_TO_ENV, spec.accumulation_alg) do
            error("spec: unknown accumulation_alg $(repr(spec.accumulation_alg)) — " *
                  "use the canonical kernel name (GPUKernelRK4 | GPUKernelNewton)")
        end
        env["EDM_ACCUM_ALG"] = alg
        env["EDM_GPU_SOLVER"] = alg
    end
    return env
end

"""
    spec_from_manifest(manifest) -> ThomsonScatteringSpec

The spec a stored run manifest attests: typed fields from `[config]` (coerced), every
unmapped `[config]` key preserved in `extra` (script-owned keys like `scattering`,
derived ones like `backscatter_n0`, analysis knobs). Reads `[config]` only — `[laser]`
and `[setup]` are DERIVED quantities the scripts compute from these inputs.
"""
function spec_from_manifest(manifest::AbstractDict)
    cfg = get(manifest, "config", Dict{String, Any}())
    spec = ThomsonScatteringSpec()
    kw = Dict{Symbol, Any}()
    for (k, v) in cfg
        f = Symbol(k)
        if f !== :extra && hasfield(ThomsonScatteringSpec, f)
            kw[f] = _coerce(_base_type(_spec_fieldtype(f)), v)
        else
            spec.extra[String(k)] = v
        end
    end
    return _respec(spec; kw...)
end

"""
    config_dict(spec::ThomsonScatteringSpec) -> Dict{String,Any}

The manifest `[config]` section this spec resolves to: set fields under their config
keys, `extra` merged in. `spec_from_manifest ∘ write_solver_manifest ∘ config_dict`
round-trips — the write side of the spec ⇄ manifest contract the pilot script will use.
"""
function config_dict(spec::ThomsonScatteringSpec)
    d = Dict{String, Any}(k => v for (k, v) in pairs(spec.extra))
    for f in fieldnames(ThomsonScatteringSpec)
        f === :extra && continue
        v = getfield(spec, f)
        v === nothing || (d[string(f)] = v)
    end
    return d
end

"""
    expand_sweep(base::ThomsonScatteringSpec, vary) -> Vector{ThomsonScatteringSpec}

The spec-native sweep expansion: `vary` maps FIELD names (manifest vocabulary — `:gamma`,
`"samples_per_period"`) to value lists; returns one spec per cartesian-product point,
`base` carried through. Pure — materialize cells with [`write_spec`](@ref) and hand them
to `run_cell` via `EDM_SPEC=<path>` in the existing per-cell overrides. The env-dict
method (knob spellings, e.g. `"A0"`) remains for the legacy launcher path.
"""
function expand_sweep(base::ThomsonScatteringSpec, vary::AbstractDict)
    ks = sort(Symbol.(collect(keys(vary))))
    vals = [vary[k] for k in sort(collect(keys(vary)); by = k -> Symbol(k))]
    out = ThomsonScatteringSpec[]
    for point in Iterators.product(vals...)
        push!(out, _respec(base; (k => v for (k, v) in zip(ks, point))...))
    end
    return out
end
