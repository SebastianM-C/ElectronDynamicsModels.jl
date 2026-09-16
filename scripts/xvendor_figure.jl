# scripts/xvendor_figure.jl — data for the cross-vendor figure (thesis fig gpu-xvendor, article
# table): per card, the strong and weak cell of an xvendor_bench_* campaign, from manifests only.
#
# Usage: julia --project=scripts scripts/xvendor_figure.jl <xvendor_campaign_dir>... [--peaks <diag_dir>...] [--out <dir>]
#
# Per card (one xvendor_strong and one xvendor_weak run; the newest of each if repeated):
#   field_s          [timing].field — wall clock of the GPU field stage
#   flop_rate        [flops].flop_rate_field — algorithmic FP64 FLOP/s over the field stage
#   peak_run         [flops].peak_fp64_flops — the FP64 peak the card measured on itself in that run
#   fraction_run     flop_rate / peak_run  (what the figure draws: self-consistent per run)
#   peak_swept       median [flops].peak_probe_flops of the card's --peaks campaign (the swept
#                    probe of GPUDiagnostics ≥ 0.4), when given; fraction_swept = flop_rate / peak_swept
#   kernel           [config].accumulation_alg + newton_iters + sample_chunks — so a run with the
#                    wrong kernel (RK4 defaults, see EDM #134) is visible and can be excluded
# Emits xvendor.json into --out (default: the first campaign dir): cards in the order given, each
# value with the uuid it came from. Runs whose kernel is not GPUKernelNewton are listed under
# "excluded" with the reason, never silently dropped or mixed into the table.

using TOML
using JSON
using Statistics
using Printf

args = copy(ARGS)
outdir = nothing; peakdirs = String[]
let i = findfirst(==("--out"), args)
    i === nothing || (global outdir = args[i + 1]; deleteat!(args, i:i + 1))
end
let i = findfirst(==("--peaks"), args)
    if i !== nothing
        j = i + 1
        while j <= length(args) && !startswith(args[j], "--")
            push!(peakdirs, args[j]); j += 1
        end
        deleteat!(args, i:j - 1)
    end
end
isempty(args) && (println(stderr, "usage: xvendor_figure.jl <xvendor_dir>... [--peaks <diag_dir>...] [--out <dir>]"); exit(64))

getpath(d, path...) = (x = d; for k in path; (x isa AbstractDict && haskey(x, k)) || return nothing; x = x[k]; end; x)
num(d, path...) = (v = getpath(d, path...); v isa Number ? Float64(v) : NaN)
vecfirst(d, path...) = (v = getpath(d, path...); v isa AbstractVector && !isempty(v) ? Float64(first(v)) : v isa Number ? Float64(v) : NaN)

manifests(dir) = [(uuid = f[5:end-5], m = TOML.parsefile(joinpath(dir, f))) for f in sort(readdir(dir)) if startswith(f, "run_") && endswith(f, ".toml")]

# swept probe peaks per device name
swept = Dict{String, Vector{Float64}}()
for d in peakdirs, r in manifests(d)
    p = num(r.m, "flops", "peak_probe_flops"); isnan(p) && continue
    push!(get!(swept, String(getpath(r.m, "gpu", "device")), Float64[]), p)
end

cards = Any[]; excluded = Any[]
seen = Dict{String, Dict{String, Any}}()   # device → cell → row
order = String[]
for d in args, r in manifests(d)
    sw = getpath(r.m, "provenance", "sweep_id"); sw in ("xvendor_strong", "xvendor_weak") || continue
    dev = String(getpath(r.m, "gpu", "device"))
    # older manifests lack config.accumulation_alg: the kernel is then identified by its FLOP count
    # per slot (737 = Newton n = 2, 973 = RK4 retarded-time kernel)
    alg0 = getpath(r.m, "config", "accumulation_alg")
    fps = num(r.m, "flops", "flop_per_slot")
    alg = alg0 !== nothing ? String(alg0) : fps == 737 ? "GPUKernelNewton (from flop_per_slot)" : fps == 973 ? "GPUKernelRK4 (from flop_per_slot)" : "?"
    row = Dict("device" => dev, "uuid" => r.uuid, "campaign" => basename(abspath(d)), "cell" => sw,
        "timestamp" => String(something(getpath(r.m, "provenance", "timestamp_utc"), getpath(r.m, "provenance", "timestamp"), "")),
        "N" => Int(num(r.m, "config", "N")), "Nx" => Int(num(r.m, "config", "Nx")),
        "field_s" => num(r.m, "timing", "field"), "kernel_s" => vecfirst(r.m, "gpu", "kernel_s"),
        "flop_rate" => num(r.m, "flops", "flop_rate_field"), "flop_per_slot" => num(r.m, "flops", "flop_per_slot"),
        "peak_run" => num(r.m, "flops", "peak_fp64_flops"), "peak_run_method" => String(something(getpath(r.m, "flops", "peak_fp64_method"), "")),
        "kernel" => alg, "newton_iters" => getpath(r.m, "config", "newton_iters"), "sample_chunks" => getpath(r.m, "config", "sample_chunks"),
        "coef_reuse" => getpath(r.m, "config", "coef_reuse"), "provider" => String(something(getpath(r.m, "provenance", "cloud_provider"), "local")),
        "repo_commit" => String(something(getpath(r.m, "provenance", "repo_commit"), "")))
    row["fraction_run"] = row["flop_rate"] / row["peak_run"]
    if !startswith(alg, "GPUKernelNewton")
        row["reason"] = "kernel $alg, not the Newton kernel of the table (EDM #134)"
        push!(excluded, row); continue
    end
    dev in order || push!(order, dev)
    cells = get!(seen, dev, Dict{String, Any}())
    if !haskey(cells, sw) || row["timestamp"] > cells[sw]["timestamp"]
        cells[sw] = row
    end
end
nanfree(x) = x isa Number && isnan(x) ? nothing : x
for dev in order
    cells = seen[dev]
    ps = get(swept, dev, Float64[])
    peak_swept = isempty(ps) ? NaN : median(ps)
    entry = Dict("device" => dev, "peak_swept" => nanfree(peak_swept), "peak_swept_n" => length(ps))
    for (cell, row) in cells
        row["fraction_swept"] = isnan(peak_swept) ? NaN : row["flop_rate"] / peak_swept
        entry[replace(cell, "xvendor_" => "")] = Dict(k => nanfree(v) for (k, v) in row)
    end
    push!(cards, entry)
end
od = something(outdir, first(args)); mkpath(od)
open(joinpath(od, "xvendor.json"), "w") do io
    JSON.print(io, Dict("cards" => cards, "excluded" => [Dict(k => nanfree(v) for (k, v) in r) for r in excluded],
        "sources" => Dict("xvendor" => basename.(abspath.(args)), "peaks" => basename.(abspath.(peakdirs)))), 2)
end
@printf("%-26s %9s %9s %8s %8s %8s %8s %8s\n", "card", "strong s", "weak s", "peak run", "swept", "str %run", "str %swp", "weak %run")
for c in cards
    s = get(c, "strong", nothing); w = get(c, "weak", nothing)
    g(r, k) = r === nothing || r[k] === nothing ? NaN : r[k]
    @printf("%-26s %9.1f %9.1f %8.2f %8s %8.1f %8s %8.1f\n", c["device"], g(s, "field_s"), g(w, "field_s"), g(s, "peak_run") / 1e12,
        c["peak_swept"] === nothing ? "—" : @sprintf("%.2f", c["peak_swept"] / 1e12), 100 * g(s, "fraction_run"),
        c["peak_swept"] === nothing ? "—" : @sprintf("%.1f", 100 * g(s, "fraction_swept")), 100 * g(w, "fraction_run"))
end
for r in excluded
    @printf("excluded: %-22s %-14s %s (%s)\n", r["device"], r["cell"], r["reason"], r["uuid"][1:8])
end
println("→ ", joinpath(od, "xvendor.json"))
