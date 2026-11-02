# Run manifest for precompute_radiation.jl: the same building blocks as the solver scripts
# (RunManifests.run_provenance + units_section, scripts/gpu_telemetry.jl's sampler and
# [gpu]/[host] sections, GPUDiagnostics' LaunchTimer), so a radiation cube or a cube
# benchmark leaves a run_<uuid>.toml the dashboard can index like any other run.
#
# Deliberately NOT write_solver_manifest: its [config] replay keys (N_samples,
# samples_per_period, …) describe a scattering screen run, and faking them would let
# run_spec_from_manifest replay a cube as one. The manifest instead carries
# `[config].kind = "radiation_cube"` and the cube's own knobs, spelled like the solver keys
# where the meaning is the same (see manifest_config in precompute_radiation.jl).
#
# One process per GPU (radiation_multigpu.jl): each process writes a PART
# (radpart_<uuid>_gpu<k>.toml, not run_*.toml, so nothing globs it as a run) and the
# wrapper merges the parts into ONE run_<uuid>.toml with [sharding].slices = #GPUs and the
# per-GPU detail under [parts.gpu<k>].
#
# Everything is best-effort: without RunManifests/GPUDiagnostics in the active env (the
# animation/ env lacks them) or on a telemetry error, the cube is still computed and the
# manifest (or one of its sections) is simply omitted, with a warning.

using TOML

const HAS_MANIFEST = Base.find_package("RunManifests") !== nothing &&
    Base.find_package("GPUDiagnostics") !== nothing
if HAS_MANIFEST
    using RunManifests
    include(joinpath(@__DIR__, "..", "scripts", "gpu_telemetry.jl"))
else
    @warn "RunManifests/GPUDiagnostics not in the active env — no run manifest will be written " *
        "(run with --project=scripts, as run_cell.sh does)"
    const GPU_SAMPLE_DT = 1.0
end

new_launch_timer() = HAS_MANIFEST ? LaunchTimer() : nothing

# Run f() under the out-of-process GPU sampler; returns (result, telemetry or nothing).
function sampled(f, backend, tracefile)
    HAS_MANIFEST || return (f(), nothing)
    return with_gpu_sampler(f, backend, GPU_SAMPLE_DT; devices = 1:1, tracefile)
end

slice_stats(ts) = isempty(ts) ? Dict{String, Any}() : Dict{String, Any}(
    "slices" => length(ts), "slice_mean" => sum(ts) / length(ts),
    "slice_min" => minimum(ts), "slice_max" => maximum(ts), "compute" => sum(ts),
)

"""
    write_radiation_manifest(; outdir, run_tag, part, config, laser, setup, timing, outputs,
                             backend, backend_name, n_threads, telem, timer, units)

Write `run_<run_tag>.toml` (or the part file `radpart_<run_tag>_gpu<part>.toml` when `part`
is an integer). Returns the path, or `nothing` when manifests are unavailable.
"""
function write_radiation_manifest(; outdir, run_tag, part = nothing, config, laser, setup,
        timing, outputs, backend, backend_name, n_threads, telem, timer, units)
    HAS_MANIFEST || return nothing
    tag = isempty(run_tag) ? "local" : run_tag
    prov = run_provenance(; run_id = tag, gpu_backend = backend_name,
        repo_dir = pkgdir(ElectronDynamicsModels),
        gpu_device = try gpu_name(backend) catch; nothing end)
    gpu = gpu_manifest_section(backend, backend_name, n_threads, 1, telem)
    timer === nothing || record_kernel_timing!(timing, gpu, timer)
    record_kernel_resources!(gpu, backend)
    sec(d) = Dict{String, Any}(string(k) => v for (k, v) in d)
    m = Dict{String, Any}(
        "schema_version" => MANIFEST_SCHEMA_VERSION, "provenance" => prov,
        "config" => sec(config), "laser" => sec(laser), "setup" => sec(setup),
        "timing" => sec(timing), "outputs" => sec(outputs), "units" => units,
        "sharding" => Dict{String, Any}("slices" => 1),
    )
    gpu === nothing || (m["gpu"] = gpu)
    h = host_manifest_section(backend)
    h === nothing || (m["host"] = h)
    part === nothing && return _install_manifest(outdir, tag, m)
    path = joinpath(outdir, "radpart_$(tag)_gpu$(part).toml")
    open(io -> TOML.print(io, m; sorted = true), path * ".tmp", "w")
    mv(path * ".tmp", path; force = true)
    return path
end

# Install `m` as run_<tag>.toml. A re-run of the same run (resume after a crash or eviction)
# must not erase the first attempt's measurements: an existing manifest moves to
# radattempt_<tag>_<k>.toml (not run_*, so nothing indexes it as a run of its own), the new
# one lists every earlier attempt in [outputs].previous_attempts and carries whole-run totals
# ([timing].attempt, slices_all_attempts, compute_all_attempts).
# An untagged local run ("local") is not a resume of the previous one: it just replaces it.
function _install_manifest(outdir, tag, m)
    path = joinpath(outdir, "run_$(tag).toml")
    attempt(k) = joinpath(outdir, "radattempt_$(tag)_$(k).toml")
    if isfile(path) && tag != "local"
        k = 1
        while isfile(attempt(k)); k += 1; end
        mv(path, attempt(k))
    end
    prev = String[]
    k = 1
    while isfile(attempt(k)); push!(prev, attempt(k)); k += 1; end
    if !isempty(prev)
        tm = m["timing"]
        old = [get(TOML.parsefile(f), "timing", Dict{String, Any}()) for f in prev]
        tm["attempt"] = length(prev) + 1
        for key in ("slices", "compute")
            tm["$(key)_all_attempts"] = get(tm, key, 0) + sum(o -> get(o, key, 0), old)
        end
        get!(m, "outputs", Dict{String, Any}())["previous_attempts"] = basename.(prev)
    end
    open(io -> TOML.print(io, m; sorted = true), path * ".tmp", "w")
    mv(path * ".tmp", path; force = true)
    return path
end

# Merge rule for several GPUs' [timing]/[gpu] sections: vectors concatenate (per device),
# counts (slices, *samples, batches_resumed) and compute seconds add, *_min → min,
# *_mean → unweighted mean over the parts (slice_mean: weighted by slices), everything else
# (wall times, *_max, *_peak) → max. Non-numeric values and
# device descriptors come from the first part.
# A key only some parts carry (a resumed GPU that computed nothing has no slice stats) is
# aggregated over the parts that do; slice_mean is weighted by each part's slice count.
function _merge_stats(ds::Vector)
    isnum(x) = x isa Real && !(x isa Bool)
    out = Dict{String, Any}()
    for d in ds, (k, v) in d   # non-numeric entries: first part that has the key
        isnum(v) || haskey(out, k) || (out[k] = deepcopy(v))
    end
    # per-device vectors (the LaunchTimer's kernel_s / kernel_median_s / … in [gpu], one
    # entry per device a process drove) concatenate, as in a multi-device solver manifest
    for k in unique(k for d in ds for (k, v) in d if v isa AbstractVector)
        out[k] = reduce(vcat, (d[k] for d in ds if haskey(d, k) && d[k] isa AbstractVector))
    end
    for k in unique(k for d in ds for (k, v) in d if isnum(v))
        has = [d for d in ds if haskey(d, k) && isnum(d[k])]
        vals = [d[k] for d in has]
        out[k] = k in ("slices", "batches_resumed", "compute") || endswith(k, "samples") ? sum(vals) :
            endswith(k, "_min") ? minimum(vals) :
            k == "slice_mean" && all(d -> haskey(d, "slices"), has) ?
                sum(d -> d["slice_mean"] * d["slices"], has) / sum(d -> d["slices"], has) :
            endswith(k, "_mean") ? sum(vals) / length(vals) : maximum(vals)
    end
    return out
end

# Per-GPU sampler traces (header "# epoch_s\tdevice\t…", one row per sample, device = 1 in
# every part since each process saw one GPU) → one trace with device = 1…n, time-sorted.
# Differing headers (counter sets) fall back to the first part's trace alone.
function _merge_gpu_traces(outdir, run_tag, files)
    # device = the part's GPU index (gpu<k> → k + 1), even when another part's trace is missing
    gpuidx(f) = (m = match(r"_gpu(\d+)\.tsv$", f); m === nothing ? nothing : parse(Int, m[1]) + 1)
    present = [(gpuidx(f), joinpath(outdir, f)) for f in files if isfile(joinpath(outdir, f))]
    isempty(present) && return first(files)
    paths = last.(present)
    heads = [readline(p) for p in paths]
    allequal(heads) || (@warn "GPU traces have different columns — keeping the first only" files; return basename(first(paths)))
    cols = split(lstrip(first(heads), ['#', ' ']), '\t')
    di = findfirst(==("device"), cols)
    rows = Tuple{Float64, String}[]
    for (k, p) in enumerate(paths), ln in Iterators.drop(eachline(p), 1)
        isempty(ln) && continue
        f = split(ln, '\t')
        di === nothing || (f[di] = string(something(first(present[k]), k)))
        push!(rows, (something(tryparse(Float64, f[1]), Inf), join(f, '\t')))
    end
    sort!(rows; by = first)
    out = joinpath(outdir, "gputrace_$(run_tag).tsv")
    open(out * ".tmp", "w") do io
        println(io, first(heads))
        foreach(r -> println(io, r[2]), rows)
    end
    mv(out * ".tmp", out; force = true)
    foreach(p -> p == out || rm(p), paths)
    return basename(out)
end

"""
    merge_radiation_parts(outdir, run_tag, n) -> path

Fold the n part files into `run_<run_tag>.toml`: [config] gets the per-GPU batch ranges,
[timing]/[gpu] merge per `_merge_stats`, [sharding].slices = n, [outputs] concatenates the
batch and trace lists, and each part's [timing] + batches stay under [parts.gpu<k>]. Bench
parts additionally yield `estimate_full_cube_wall` = NSLICES · mean slice time / n, the
wall-clock of the whole cube split across the n GPUs.
"""
function merge_radiation_parts(outdir, run_tag, n)
    files = [joinpath(outdir, "radpart_$(run_tag)_gpu$(k).toml") for k in 0:(n - 1)]
    missing_parts = filter(!isfile, files)
    isempty(missing_parts) || error("missing manifest parts: $(join(basename.(missing_parts), ", "))")
    ps = TOML.parsefile.(files)
    m = deepcopy(first(ps))
    m["config"]["batches"] = [get(p["config"], "batches", "") for p in ps]
    m["config"]["processes"] = n
    m["timing"] = _merge_stats([p["timing"] for p in ps])
    if all(p -> haskey(p, "gpu"), ps)
        m["gpu"] = _merge_stats([p["gpu"] for p in ps])
        m["gpu"]["device_count"] = n
    end
    m["sharding"] = Dict{String, Any}("slices" => n)
    if get(m["config"], "bench", false)
        m["timing"]["estimate_full_cube_wall"] =
            m["config"]["n_slices"] * m["timing"]["slice_mean"] / n
    end
    outs = Dict{String, Any}()
    scalar = Set{String}()   # keys every part wrote as one file name (the shared batch manifest)
    for p in ps, (k, v) in get(p, "outputs", Dict())
        v isa AbstractVector || push!(scalar, k)
        outs[k] = unique(vcat(get(outs, k, String[]), v isa AbstractVector ? v : [v]))
    end
    for k in scalar
        length(outs[k]) == 1 && (outs[k] = only(outs[k]))
    end
    # The dashboard reads [outputs].gpu_trace as ONE file: fold the per-GPU traces into
    # gputrace_<tag>.tsv, as a multi-device solver run writes it.
    if get(outs, "gpu_trace", nothing) isa AbstractVector
        outs["gpu_trace"] = _merge_gpu_traces(outdir, run_tag, outs["gpu_trace"])
    end
    m["outputs"] = outs
    m["parts"] = Dict{String, Any}("gpu$(k - 1)" => Dict{String, Any}(
        "timing" => ps[k]["timing"], "batches" => get(ps[k]["config"], "batches", ""),
        "gpu_device" => get(ps[k]["provenance"], "gpu_device", "")) for k in 1:n)
    path = _install_manifest(outdir, run_tag, m)
    foreach(rm, files)
    return path
end
