# GPU telemetry for the solver manifests' [gpu] section — shared by thomson_scattering.jl + lpwa.jl.
# A static device snapshot (name, SMs, capacity, VRAM, thread-fill occupancy) plus a sampler that
# records power / compute-util / mem-util / VRAM over the accumulate_field window on a spawned
# thread. Uses the vendor-agnostic gpu_api (CUDA=NVML, AMD=sysfs), which the scripts already
# `using`. Everything is wrapped so a telemetry hiccup NEVER breaks a run — it just omits the section.

# Run `f()` while a background task samples the CURRENT device every `dt` seconds. Returns
# (result_of_f, samples::Vector{NTuple{4,Float64}} = (power_W, compute_util, mem_util, vram_used_B)).
# `f` is FIRST so the do-block form works: `with_gpu_sampler(backend, dt) do … end` (Julia prepends
# the closure). The sampler is the sole writer and the caller reads `samples` only after wait() ⇒ no
# data race. Needs ≥2 Julia threads or the sampler starves behind the field kernel (caller should warn).
function with_gpu_sampler(f, backend, dt::Real)
    running = Threads.Atomic{Bool}(true)
    samples = NTuple{4, Float64}[]
    sampler = Threads.@spawn begin
        while running[]
            try
                u = gpu_utilization(backend)
                m = gpu_memory_info(backend)
                push!(samples, (gpu_power(backend), Float64(u.compute), Float64(u.memory), Float64(m.used)))
            catch
                # transient telemetry read failure — drop this sample, keep sampling
            end
            sleep(dt)
        end
    end
    result = f()
    running[] = false
    wait(sampler)
    return result, samples
end

# Static device snapshot + reduced sampler stats → the manifest's [gpu] table (a plain Dict that
# RunManifests writes verbatim as a top-level section). `n_threads` = the pixel-parallel launch size
# (Nx·Ny) for thread-fill occupancy. In a sharded (device_count>1) run the sampler sees only its own
# current device; the static props are per-device (identical hardware) and device_count records the fan-out.
# Returns `nothing` if telemetry is unavailable (e.g. no vendor extension) so the caller just omits [gpu].
function gpu_manifest_section(backend, backend_name::AbstractString, n_threads::Integer,
        device_count::Integer, samples::Vector{NTuple{4, Float64}})
    try
        gpu = Dict{String, Any}(
            "backend" => String(backend_name),
            "device" => gpu_name(backend),
            "device_count" => Int(device_count),
            "sm_count" => Int(gpu_sm_count(backend)),
            "max_threads_per_sm" => Int(gpu_max_threads_per_sm(backend)),
            "memory_total" => Int(gpu_memory_info(backend).total),
            "thread_fill_occupancy" => Float64(thread_fill_occupancy(backend, n_threads)),
        )
        if !isempty(samples)
            _mean(v) = sum(v) / length(v)
            pw = Float64[s[1] for s in samples]
            cu = Float64[s[2] for s in samples]
            mu = Float64[s[3] for s in samples]
            vr = Float64[s[4] for s in samples]
            gpu["samples"] = length(samples)
            gpu["power_mean"] = _mean(pw);          gpu["power_peak"] = maximum(pw)
            gpu["compute_util_mean"] = _mean(cu);   gpu["compute_util_peak"] = maximum(cu)
            gpu["memory_util_mean"] = _mean(mu);    gpu["memory_util_peak"] = maximum(mu)
            gpu["vram_used_peak"] = maximum(vr)
        end
        return gpu
    catch err
        @warn "GPU telemetry unavailable — omitting [gpu] from the manifest" exception = err
        return nothing
    end
end

# Sampler cadence (s); coarse is fine — field runs are seconds→hours. Override with EDM_GPU_SAMPLE_DT.
const GPU_SAMPLE_DT = parse(Float64, get(ENV, "EDM_GPU_SAMPLE_DT", "1.0"))
