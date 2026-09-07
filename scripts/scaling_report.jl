# scripts/scaling_report.jl — GPU-count scaling tables + figure from campaign manifests.
#
# Usage: julia --project=scripts scripts/scaling_report.jl <campaign_dir>... [--out <dir>]
#
# Reads every run_<uuid>.toml under the given dirs and groups runs by their sweep tag
# (provenance.sweep_id, e.g. mgpu_weak / mgpu_strong from campaigns/mgpu_bench_*.sh; runs
# without one group by directory). Per run: device count D ([sharding].electrons, or the
# older [timing].n_devices), N, Nx, N_samples, the field-phase and end-to-end wall times, and
# the device name. Emits, into --out (default: the first dir):
#   scaling_report.md   — per-group tables: weak scaling (N ∝ D: efficiency = t₁/t_D at
#                         constant work per device), strong scaling (fixed config: speedup
#                         t₁/t_D vs ideal D), and per-device throughput
#                         N·N_samples·Nx²/(t_field·D) [electron·sample·pixel/s] for every run
#                         — the cross-vendor comparison row (H200 / H100 / MI300X …).
#                         Every table carries BOTH the field-phase wall time and the
#                         KERNEL-ACTIVE time: seconds the busiest device's gputrace reports
#                         ≥ 99 % utilization (torn/implausible rows dropped). The wall time
#                         includes the per-process first-call kernel JIT (~60 s) and, in
#                         sharded runs, the serialized per-device compile (~40 s × D); the
#                         kernel-active time is the clean scaling metric.
#   scaling_report.png  — weak (time vs D) + strong (speedup vs D, ideal line) panels
# Field-phase times are the clean GPU-bound numbers; the end-to-end column includes the
# un-sharded Julia load / serialize / reduce and any host contention between lanes.
#
# FLOP columns: per-device FLOP/s (wall and kernel-active), the fraction of the device's vector
# FP64 peak, and the arithmetic intensity, from the manifest's [flops] section when present
# (`flop_total` = algorithmic FLOPs of the run: CountedFloats profile of the kernel × executed
# slots, see scripts/gpu_telemetry.jl). Older manifests without [flops] are costed with
# `flop_profile` from their [config] over the NOMINAL slot count and a device-name peak table,
# and carry a `*` marker. FLOP/s is algorithmic (as-written) work per second — the usual figure
# against a hardware peak, not a hardware instruction rate.

using TOML
using Printf
using Statistics
using CairoMakie
using ElectronDynamicsModels

args = copy(ARGS)
outdir = nothing
if (i = findfirst(==("--out"), args)) !== nothing
    outdir = args[i + 1]; deleteat!(args, i:(i + 1))
end
isempty(args) && error("usage: scaling_report.jl <campaign_dir>... [--out <dir>]")
outdir = something(outdir, first(args))

struct Row
    id::String; label::String; group::String; dir::String
    D::Int; N::Int; Nx::Int; Ns::Int
    t_field::Float64; t_total::Float64; t_traj::Float64
    device::String
    t_kernel::Float64   # busiest device's seconds at ≥ 99 % utilization (NaN without a trace)
    flop_total::Float64 # algorithmic FLOPs of the whole run (NaN if uncostable)
    ai::Float64         # arithmetic intensity [FLOP/B] of the kernel (device-buffer traffic)
    peak::Float64       # vector FP64 peak of the device [FLOP/s] (NaN if unknown)
    flops_src::String   # "manifest" ([flops] section) | "model" (profiled from [config], nominal slots) | "—"
end

# Vector FP64 peaks [FLOP/s] by device-name fragment, for manifests without [flops].peak_fp64_flops
# (boost-clock × SMs × lanes; the extensions compute the same for live runs).
const PEAK_FP64 = (
    "H200" => 33.5e12, "H100" => 33.5e12, "A100" => 9.7e12, "GB200" => 40.0e12, "B200" => 40.0e12,
    "MI300X" => 81.7e12, "MI250X" => 47.9e12, "MI250" => 45.3e12,
    "RTX 5090" => 1.64e12, "RTX 4090" => 1.29e12, "W7900" => 1.35e12,
)
device_peak(dev) = (i = findfirst(p -> occursin(p[1], dev), PEAK_FP64); i === nothing ? NaN : PEAK_FP64[i][2])

# Fallback costing for manifests written before [flops] existed: profile the kernel named in
# [config] (memoized; milliseconds each) over the nominal slot count.
const PROFILES = Dict{Tuple{String, String, Int}, Any}()
function model_flops(cfg, N, Ns, Nx)
    alg = String(get(cfg, "accumulation_alg", "GPUKernelRK4"))
    mode = String(get(cfg, "mode", "split"))
    n = alg == "GPUKernelNewton" ? Int(get(cfg, "newton_iters", 2)) : Int(get(cfg, "n_substeps", 1))
    p = get!(PROFILES, (alg, mode, n)) do
        alg == "GPUKernelNewton" ? flop_profile(GPUKernelNewton(); mode = Symbol(mode), n_iters = n) :
            flop_profile(GPUKernelRK4(); mode = Symbol(mode), n_substeps = n)
    end
    return p.flop_per_slot * N * Ns * Nx^2 + p.flop_per_pixel_launch * N * Nx^2, p.arithmetic_intensity
end

# Kernel-active seconds from a gputrace TSV: per device, count rows with compute_util ≥ 0.99,
# skipping rows that are torn (≠ 6 fields) or implausible (util ∉ [0,1], VRAM > 1 TB); the
# busiest device sets the cell's kernel time (sharded devices finish within seconds of each
# other, so max ≈ every device's).
function kernel_active(dir, id)
    fs = filter(f -> startswith(f, "gputrace_") && occursin(id, f), readdir(dir))
    isempty(fs) && return NaN
    per = Dict{String, Int}()
    for ln in eachline(joinpath(dir, fs[1]))
        startswith(ln, '#') && continue
        f = split(ln, '\t'); length(f) == 6 || continue
        u = tryparse(Float64, f[4]); v = tryparse(Float64, f[6])
        (u === nothing || v === nothing || !(0 <= u <= 1) || v > 1.0e12) && continue
        u >= 0.99 && (per[f[2]] = get(per, f[2], 0) + 1)
    end
    return isempty(per) ? NaN : Float64(maximum(values(per)))
end

function cell_labels(dir)
    f = joinpath(dir, "cells.tsv"); out = Dict{String, String}()
    isfile(f) || return out
    for (i, ln) in enumerate(eachline(f))
        i == 1 && continue
        p = split(ln, '\t'); length(p) >= 2 && (out[String(p[2])] = String(p[1]))
    end
    return out
end

rows = Row[]
for dir in args
    labels = cell_labels(dir)
    for f in sort(filter(x -> startswith(x, "run_") && endswith(x, ".toml"), readdir(dir)))
        m = TOML.parsefile(joinpath(dir, f))
        cfg = m["config"]; tm = get(m, "timing", Dict()); prov = get(m, "provenance", Dict())
        haskey(tm, "field") || continue
        D = Int(get(get(m, "sharding", Dict()), "electrons", get(tm, "n_devices", 1)))
        id = String(get(prov, "run_id", f[5:(end - 5)]))
        group = String(get(prov, "sweep_id", basename(dir)))
        dev = String(something(get(get(m, "gpu", Dict()), "device", nothing), get(prov, "gpu_device", nothing), "?"))
        N, Nx, Ns = Int(cfg["N"]), Int(cfg["Nx"]), Int(cfg["N_samples"])
        fl = get(m, "flops", nothing)
        flop_total, ai, peak, src = if fl !== nothing && haskey(fl, "flop_total")
            Float64(fl["flop_total"]), Float64(get(fl, "arithmetic_intensity", NaN)),
                Float64(get(fl, "peak_fp64_flops", device_peak(dev))), "manifest"
        else
            try
                ft, a = model_flops(cfg, N, Ns, Nx)
                ft, a, device_peak(dev), "model"
            catch err
                @warn "cannot cost run $id" exception = err
                NaN, NaN, NaN, "—"
            end
        end
        push!(rows, Row(id, get(labels, id, id[1:8]), group, dir,
            D, N, Nx, Ns,
            Float64(tm["field"]), Float64(get(tm, "total", NaN)), Float64(get(tm, "trajectories", NaN)), dev,
            kernel_active(dir, id), flop_total, ai, peak, src))
    end
end
isempty(rows) && error("no manifests with [timing].field under $(join(args, ", "))")

work(r) = r.N * r.Ns * r.Nx^2                       # electron·sample·pixel
thr(r) = work(r) / (r.t_field * r.D)                # per device, wall
thrk(r) = isnan(r.t_kernel) ? NaN : work(r) / (r.t_kernel * r.D)   # per device, kernel-active
fl(r) = r.flop_total / (r.t_field * r.D)            # FLOP/s per device, wall
flk(r) = isnan(r.t_kernel) ? NaN : r.flop_total / (r.t_kernel * r.D)   # FLOP/s per device, kernel-active
pk(x, r) = x / r.peak                               # fraction of the device's FP64 peak
fmt_e(x) = isnan(x) ? "—" : @sprintf("%.3e", x)
fmt_t(s) = isnan(s) ? "—" : s < 3600 ? @sprintf("%.0f s", s) : @sprintf("%.2f h", s / 3600)
fmt_pct(x) = isnan(x) ? "—" : @sprintf("%.2f %%", 100x)
fmt_f(x) = isnan(x) ? "—" : @sprintf("%.1f", x)
mark(r) = r.flops_src == "model" ? "*" : ""

io = IOBuffer()
println(io, "# GPU-count scaling report\n")
println(io, "Sources: ", join(args, ", "), "\n")
groups = unique(r.group for r in rows)
weak_groups = String[]; strong_groups = String[]
for g in groups
    rs = sort(filter(r -> r.group == g, rows); by = r -> (r.D, r.N))
    Ds = unique(r.D for r in rs)
    println(io, "## ", g, "  (", length(rs), " runs)\n")
    println(io, "| cell | device | D | N | Nx | N_samples | field (wall) | kernel-active | end-to-end | traj | per-device rate wall / kernel [e·s·px/s] | FLOP/s wall / kernel [per device] | % FP64 peak wall / kernel | AI [FLOP/B] |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs
        @printf(io, "| %s | %s | %d | %d | %d | %d | %s | %s | %s | %s | %.3e / %s | %s / %s%s | %s / %s | %s |\n",
            r.label, r.device, r.D, r.N, r.Nx, r.Ns, fmt_t(r.t_field), fmt_t(r.t_kernel), fmt_t(r.t_total), fmt_t(r.t_traj), thr(r), fmt_e(thrk(r)),
            fmt_e(fl(r)), fmt_e(flk(r)), mark(r), fmt_pct(pk(fl(r), r)), fmt_pct(pk(flk(r), r)), fmt_f(r.ai))
    end
    println(io)
    any(r -> r.flops_src == "model", rs) &&
        println(io, "`*` FLOPs modelled from [config] over the nominal slot count (manifest predates the [flops] section).\n")
    length(Ds) > 1 || continue
    base = filter(r -> r.D == minimum(Ds), rs)
    length(base) == 1 || continue
    b = base[1]
    if all(r -> r.N * b.D == b.N * r.D && r.Nx == b.Nx && r.Ns == b.Ns, rs)
        push!(weak_groups, g)
        println(io, "**Weak scaling** (work per device constant; efficiency = t(D=$(b.D)) / t(D)):\n")
        println(io, "| D | N | field (wall) | efficiency (wall) | kernel-active | efficiency (kernel) | end-to-end | efficiency (end-to-end) |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for r in rs
            ek = isnan(r.t_kernel) || isnan(b.t_kernel) ? "—" : @sprintf("%.2f", b.t_kernel / r.t_kernel)
            @printf(io, "| %d | %d | %s | %.2f | %s | %s | %s | %.2f |\n", r.D, r.N, fmt_t(r.t_field), b.t_field / r.t_field,
                fmt_t(r.t_kernel), ek, fmt_t(r.t_total), b.t_total / r.t_total)
        end
        println(io)
    elseif all(r -> r.N == b.N && r.Nx == b.Nx && r.Ns == b.Ns, rs)
        push!(strong_groups, g)
        println(io, "**Strong scaling** (fixed problem; speedup = t(D=$(b.D)) / t(D), ideal = D/$(b.D)):\n")
        println(io, "| D | field (wall) | speedup (wall) | efficiency (wall) | kernel-active | speedup (kernel) | efficiency (kernel) | end-to-end | speedup (end-to-end) |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        for r in rs
            sp = b.t_field / r.t_field; ideal = r.D / b.D
            spk = isnan(r.t_kernel) || isnan(b.t_kernel) ? NaN : b.t_kernel / r.t_kernel
            @printf(io, "| %d | %s | %.2f | %.2f | %s | %s | %s | %s | %.2f |\n", r.D, fmt_t(r.t_field), sp, sp / ideal,
                fmt_t(r.t_kernel), isnan(spk) ? "—" : @sprintf("%.2f", spk), isnan(spk) ? "—" : @sprintf("%.2f", spk / ideal),
                fmt_t(r.t_total), b.t_total / r.t_total)
        end
        println(io)
    end
end
# cross-vendor: per-device throughput by device name (median over runs)
println(io, "## Per-device throughput by device (median over all runs)\n")
println(io, "| device | runs | wall [e·s·px/s] | relative | kernel-active [e·s·px/s] | relative | FLOP/s wall / kernel | % FP64 peak wall / kernel |")
println(io, "|---|---|---|---|---|---|---|---|")
devs = unique(r.device for r in rows)
nanmedian(v) = (w = filter(!isnan, collect(v)); isempty(w) ? NaN : median(w))
med = Dict(d => median(thr(r) for r in rows if r.device == d) for d in devs)
medk = Dict(d => nanmedian(thrk(r) for r in rows if r.device == d) for d in devs)
medf = Dict(d => nanmedian(fl(r) for r in rows if r.device == d) for d in devs)
medfk = Dict(d => nanmedian(flk(r) for r in rows if r.device == d) for d in devs)
medp = Dict(d => nanmedian(pk(fl(r), r) for r in rows if r.device == d) for d in devs)
medpk = Dict(d => nanmedian(pk(flk(r), r) for r in rows if r.device == d) for d in devs)
best = maximum(values(med)); kvals = filter(!isnan, collect(values(medk))); bestk = isempty(kvals) ? NaN : maximum(kvals)
for d in sort(devs; by = d -> -med[d])
    @printf(io, "| %s | %d | %.3e | %.2f | %s | %s | %s / %s | %s / %s |\n", d, count(r -> r.device == d, rows), med[d], med[d] / best,
        fmt_e(medk[d]), isnan(medk[d]) ? "—" : @sprintf("%.2f", medk[d] / bestk),
        fmt_e(medf[d]), fmt_e(medfk[d]), fmt_pct(medp[d]), fmt_pct(medpk[d]))
end
report = String(take!(io))
mkpath(outdir)
write(joinpath(outdir, "scaling_report.md"), report)
print(report)

# ── figure ──
fig = Figure(size = (1000, 420))
ax1 = Axis(fig[1, 1]; xlabel = "devices D", ylabel = "field time [s]", title = "weak scaling (N ∝ D)",
    xticks = [1, 2, 4, 8])
ax2 = Axis(fig[1, 2]; xlabel = "devices D", ylabel = "speedup t₁/t_D", title = "strong scaling (fixed N)",
    xticks = [1, 2, 4, 8])
for g in weak_groups
    rs = sort(filter(r -> r.group == g, rows); by = r -> r.D)
    scatterlines!(ax1, [r.D for r in rs], [r.t_field for r in rs]; label = "$g field (wall)")
    any(r -> !isnan(r.t_kernel), rs) &&
        scatterlines!(ax1, [r.D for r in rs], [r.t_kernel for r in rs]; linestyle = :dashdot, label = "$g kernel-active")
    any(r -> !isnan(r.t_total), rs) &&
        scatterlines!(ax1, [r.D for r in rs], [r.t_total for r in rs]; linestyle = :dash, label = "$g end-to-end")
end
for g in strong_groups
    rs = sort(filter(r -> r.group == g, rows); by = r -> r.D)
    b = rs[1]
    scatterlines!(ax2, [r.D for r in rs], [b.t_field / r.t_field for r in rs]; label = "$g field (wall)")
    any(r -> !isnan(r.t_kernel), rs) && !isnan(b.t_kernel) &&
        scatterlines!(ax2, [r.D for r in rs], [b.t_kernel / r.t_kernel for r in rs]; linestyle = :dashdot, label = "$g kernel-active")
    any(r -> !isnan(r.t_total), rs) &&
        scatterlines!(ax2, [r.D for r in rs], [b.t_total / r.t_total for r in rs]; linestyle = :dash, label = "$g end-to-end")
end
Dmax = maximum(r.D for r in rows)
lines!(ax2, [1, Dmax], [1, Dmax]; color = :gray, linestyle = :dot, label = "ideal")
isempty(weak_groups) || axislegend(ax1; position = :lt)
axislegend(ax2; position = :lt)
save(joinpath(outdir, "scaling_report.png"), fig)
println("\n→ ", joinpath(outdir, "scaling_report.md"), "  ", joinpath(outdir, "scaling_report.png"))
