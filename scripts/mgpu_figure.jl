# scripts/mgpu_figure.jl — data for the multi-device scaling figure and table (thesis
# fig gpu-scaling / tab edm-mgpu), from the manifests of an mgpu_bench_* campaign.
#
# Usage: julia --project=scripts scripts/mgpu_figure.jl <campaign_dir>... [--out <dir>]
#
# Per run (grouped by [provenance].sweep_id: mgpu_strong, mgpu_weak, mgpu_capacity):
#   D            device count ([sharding].electrons or [gpu].device_count)
#   field_s      [timing].field — wall clock of the GPU field stage
#   kernel_s     summed device-event time of the BUSIEST device net of its first-launch
#                compilation: max_d([gpu].kernel_s[d] − [gpu].kernel_first_s[d]) — the clean
#                per-device scaling metric of the figure
#   kernel_raw_s the same with the compilation included; first_s the compilation itself
#   launch_ms    median device time per launch on that device ([gpu].kernel_median_s)
#   reduce_s     [timing].reduce_fold — on-device summation of the partial fields (D > 1)
#   download_s   [timing].reduce_download — the single cube download (D > 1)
# and, per sweep, the derived numbers the text quotes: strong speedup t1/tD for kernel and
# field, parallel efficiency, weak efficiency, per-launch drift. Emits mgpu.json into --out.

using TOML
using JSON
using Printf

args = copy(ARGS)
outdir = nothing
let i = findfirst(==("--out"), args)
    i === nothing || (global outdir = args[i + 1]; deleteat!(args, i:i + 1))
end
isempty(args) && (println(stderr, "usage: mgpu_figure.jl <campaign_dir>... [--out <dir>]"); exit(64))

getpath(d, path...) = (x = d; for k in path; (x isa AbstractDict && haskey(x, k)) || return nothing; x = x[k]; end; x)
num(d, path...) = (v = getpath(d, path...); v isa Number ? Float64(v) : NaN)
vec(d, path...) = (v = getpath(d, path...); v isa AbstractVector ? Float64.(v) : v isa Number ? [Float64(v)] : Float64[])
nanfree(x) = x isa Number && isnan(x) ? nothing : x

rows = Any[]
for dir in args, f in sort(readdir(dir))
    startswith(f, "run_") && endswith(f, ".toml") || continue
    m = TOML.parsefile(joinpath(dir, f))
    sw = getpath(m, "provenance", "sweep_id"); sw === nothing && continue
    ks = vec(m, "gpu", "kernel_s"); kf = vec(m, "gpu", "kernel_first_s"); km = vec(m, "gpu", "kernel_median_s")
    isempty(ks) && continue
    net = isempty(kf) ? ks : ks .- kf
    b = argmax(net)   # the busiest device
    D = Int(something(getpath(m, "sharding", "electrons"), getpath(m, "gpu", "device_count"), length(ks)))
    push!(rows, Dict("uuid" => f[5:end-5], "sweep" => String(sw), "D" => D,
        "N" => Int(num(m, "config", "N")), "Nx" => Int(num(m, "config", "Nx")),
        "field_s" => num(m, "timing", "field"), "kernel_s" => net[b], "kernel_raw_s" => ks[b],
        "first_s" => isempty(kf) ? NaN : kf[b], "launch_ms" => isempty(km) ? NaN : 1e3 * km[b],
        "reduce_s" => num(m, "timing", "reduce_fold"), "download_s" => num(m, "timing", "reduce_download"),
        "device" => String(something(getpath(m, "gpu", "device"), "")),
        "campaign" => basename(abspath(dir)), "repo_commit" => String(something(getpath(m, "provenance", "repo_commit"), ""))))
end
sort!(rows; by = r -> (r["sweep"], r["D"]))
bysweep(s) = [r for r in rows if r["sweep"] == s]
derived = Dict{String, Any}()
let S = bysweep("mgpu_strong")
    if !isempty(S)
        r1 = S[1]; rD = S[end]
        derived["strong"] = Dict("D_max" => rD["D"],
            "speedup_kernel" => r1["kernel_s"] / rD["kernel_s"], "speedup_field" => r1["field_s"] / rD["field_s"],
            "speedup_kernel_raw" => r1["kernel_raw_s"] / rD["kernel_raw_s"],
            "speedup_launch" => r1["launch_ms"] / rD["launch_ms"],
            "efficiency_kernel" => r1["kernel_s"] / rD["kernel_s"] / rD["D"],
            "launch_drift" => rD["launch_ms"] / r1["launch_ms"] - 1)
    end
end
let W = bysweep("mgpu_weak")
    if !isempty(W)
        r1 = W[1]; rD = W[end]
        derived["weak"] = Dict("D_max" => rD["D"], "efficiency_kernel" => r1["kernel_s"] / rD["kernel_s"],
            "efficiency_field" => r1["field_s"] / rD["field_s"], "launch_drift" => rD["launch_ms"] / r1["launch_ms"] - 1)
    end
end
od = something(outdir, first(args)); mkpath(od)
open(joinpath(od, "mgpu.json"), "w") do io
    JSON.print(io, Dict("runs" => [Dict(k => nanfree(v) for (k, v) in r) for r in rows], "derived" => derived,
        "sources" => basename.(abspath.(args))), 2)
end
for r in rows
    @printf("%-14s D=%d N=%6d Nx=%4d field %7.1f s  kernel %7.1f s (raw %7.1f, first %5.1f)  %6.2f ms/launch  reduce %5s  download %5s\n",
        r["sweep"], r["D"], r["N"], r["Nx"], r["field_s"], r["kernel_s"], r["kernel_raw_s"], r["first_s"], r["launch_ms"],
        isnan(r["reduce_s"]) ? "—" : @sprintf("%.2f", r["reduce_s"]), isnan(r["download_s"]) ? "—" : @sprintf("%.2f", r["download_s"]))
end
println(JSON.json(derived))
