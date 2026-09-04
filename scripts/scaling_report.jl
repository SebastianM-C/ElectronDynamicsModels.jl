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
#                         — the cross-vendor comparison row (H200 / H100 / MI300X …)
#   scaling_report.png  — weak (time vs D) + strong (speedup vs D, ideal line) panels
# Field-phase times are the clean GPU-bound numbers; the end-to-end column includes the
# un-sharded Julia load / serialize / reduce and any host contention between lanes.

using TOML
using Printf
using Statistics
using CairoMakie

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
        push!(rows, Row(id, get(labels, id, id[1:8]), group, dir,
            D, Int(cfg["N"]), Int(cfg["Nx"]), Int(cfg["N_samples"]),
            Float64(tm["field"]), Float64(get(tm, "total", NaN)), Float64(get(tm, "trajectories", NaN)), dev))
    end
end
isempty(rows) && error("no manifests with [timing].field under $(join(args, ", "))")

work(r) = r.N * r.Ns * r.Nx^2                       # electron·sample·pixel
thr(r) = work(r) / (r.t_field * r.D)                # per device
fmt_t(s) = isnan(s) ? "—" : s < 3600 ? @sprintf("%.0f s", s) : @sprintf("%.2f h", s / 3600)

io = IOBuffer()
println(io, "# GPU-count scaling report\n")
println(io, "Sources: ", join(args, ", "), "\n")
groups = unique(r.group for r in rows)
weak_groups = String[]; strong_groups = String[]
for g in groups
    rs = sort(filter(r -> r.group == g, rows); by = r -> (r.D, r.N))
    Ds = unique(r.D for r in rs)
    println(io, "## ", g, "  (", length(rs), " runs)\n")
    println(io, "| cell | device | D | N | Nx | N_samples | field | end-to-end | traj | per-device throughput [e·s·px/s] |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for r in rs
        @printf(io, "| %s | %s | %d | %d | %d | %d | %s | %s | %s | %.3e |\n",
            r.label, r.device, r.D, r.N, r.Nx, r.Ns, fmt_t(r.t_field), fmt_t(r.t_total), fmt_t(r.t_traj), thr(r))
    end
    println(io)
    length(Ds) > 1 || continue
    base = filter(r -> r.D == minimum(Ds), rs)
    length(base) == 1 || continue
    b = base[1]
    if all(r -> r.N * b.D == b.N * r.D && r.Nx == b.Nx && r.Ns == b.Ns, rs)
        push!(weak_groups, g)
        println(io, "**Weak scaling** (work per device constant; efficiency = t_field(D=$(b.D)) / t_field(D)):\n")
        println(io, "| D | N | field | efficiency (field) | end-to-end | efficiency (end-to-end) |")
        println(io, "|---|---|---|---|---|---|")
        for r in rs
            @printf(io, "| %d | %d | %s | %.2f | %s | %.2f |\n", r.D, r.N, fmt_t(r.t_field), b.t_field / r.t_field,
                fmt_t(r.t_total), b.t_total / r.t_total)
        end
        println(io)
    elseif all(r -> r.N == b.N && r.Nx == b.Nx && r.Ns == b.Ns, rs)
        push!(strong_groups, g)
        println(io, "**Strong scaling** (fixed problem; speedup = t_field(D=$(b.D)) / t_field(D), ideal = D/$(b.D)):\n")
        println(io, "| D | field | speedup (field) | efficiency | end-to-end | speedup (end-to-end) |")
        println(io, "|---|---|---|---|---|---|")
        for r in rs
            sp = b.t_field / r.t_field; ideal = r.D / b.D
            @printf(io, "| %d | %s | %.2f | %.2f | %s | %.2f |\n", r.D, fmt_t(r.t_field), sp, sp / ideal,
                fmt_t(r.t_total), b.t_total / r.t_total)
        end
        println(io)
    end
end
# cross-vendor: per-device throughput by device name (median over runs)
println(io, "## Per-device throughput by device (median over all runs)\n")
println(io, "| device | runs | throughput [e·s·px/s] | relative |")
println(io, "|---|---|---|---|")
devs = unique(r.device for r in rows)
med = Dict(d => median(thr(r) for r in rows if r.device == d) for d in devs)
best = maximum(values(med))
for d in sort(devs; by = d -> -med[d])
    @printf(io, "| %s | %d | %.3e | %.2f |\n", d, count(r -> r.device == d, rows), med[d], med[d] / best)
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
    scatterlines!(ax1, [r.D for r in rs], [r.t_field for r in rs]; label = "$g field")
    any(r -> !isnan(r.t_total), rs) &&
        scatterlines!(ax1, [r.D for r in rs], [r.t_total for r in rs]; linestyle = :dash, label = "$g end-to-end")
end
for g in strong_groups
    rs = sort(filter(r -> r.group == g, rows); by = r -> r.D)
    b = rs[1]
    scatterlines!(ax2, [r.D for r in rs], [b.t_field / r.t_field for r in rs]; label = "$g field")
    any(r -> !isnan(r.t_total), rs) &&
        scatterlines!(ax2, [r.D for r in rs], [b.t_total / r.t_total for r in rs]; linestyle = :dash, label = "$g end-to-end")
end
Dmax = maximum(r.D for r in rows)
lines!(ax2, [1, Dmax], [1, Dmax]; color = :gray, linestyle = :dot, label = "ideal")
isempty(weak_groups) || axislegend(ax1; position = :lt)
axislegend(ax2; position = :lt)
save(joinpath(outdir, "scaling_report.png"), fig)
println("\n→ ", joinpath(outdir, "scaling_report.md"), "  ", joinpath(outdir, "scaling_report.png"))
