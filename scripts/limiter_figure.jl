# scripts/limiter_figure.jl — "what limits the kernel on this card": the per-card figure for
# the thesis (and its short form in the article), drawn from run manifests only.
#
# Usage: julia --project=scripts scripts/limiter_figure.jl <campaign_dir>... [--out <dir>]
#
# A campaign dir is one card's diag_validate_* run (orchestration/campaigns/diag_validate_*.sh:
# a sample-chunk sweep C ∈ {1, 2, 4, 8, 16} at N = 2000 under the sampler — `diag_chunks_power` —
# and at N = 32 under the `occupancy` counter set — `diag_chunks_occupancy` — plus the single-C
# `fp64`, `issue`, `memory`, `l2` counter cells — `diag_counters`; older 3-point campaigns
# (`diag_power`, `diag_counters` with occupancy_c*) are read the same way). Per campaign it emits,
# into --out (default: the campaign dir):
#   limiter_<slug>.csv   — one row per (sweep, C, metric): value + the manifest uuid it came from
#   limiter_<slug>.json  — the same, structured, with the card header (probe peak, GEMM reference,
#                          counted FP64 FLOP per slot, provider, commit) — the thesis figure's data
#   limiter_<slug>.png   — three panels over C: (a) kernel time per slot and achieved FLOP/s,
#                          (b) throttle signals: busy-median clock, power against the cap, capped
#                          fraction, PPT/thermal residency fractions where the sampler has them,
#                          (c) counters: achieved occupancy, FP64-pipe / VALU fraction, wave-wait
#                          fraction (AMD) or GPM FP64 utilisation (NVIDIA), profiler time per slot.
# Every number comes from `[timing]`, `[flops]`, `[gpu]` (sampler_* / gpm_* / hw_*) and `[config]`
# of the manifests; nothing is typed in. Keys a card lacks (no counters on a container that refuses
# them, no residencies before the sampler had them) stay NaN and their panel line is simply absent.

using TOML
using JSON
using Printf
using Statistics
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))

# ── args ────────────────────────────────────────────────────────────────────────────────────
args = copy(ARGS)
outdir = nothing
let i = findfirst(==("--out"), args)
    if i !== nothing
        global outdir = args[i + 1]
        deleteat!(args, i:i + 1)
    end
end
isempty(args) && (println(stderr, "usage: limiter_figure.jl <campaign_dir>... [--out <dir>]"); exit(64))

# ── manifest access ─────────────────────────────────────────────────────────────────────────
getpath(d, path...; default = missing) = begin
    x = d
    for k in path
        x isa AbstractDict && haskey(x, k) || return default
        x = x[k]
    end
    x
end
num(d, path...) = (v = getpath(d, path...); v isa Number ? Float64(v) : NaN)
vecfirst(d, path...) = (v = getpath(d, path...); v isa AbstractVector && !isempty(v) ? Float64(first(v)) :
    v isa Number ? Float64(v) : NaN)

struct Run
    uuid::String
    label::String
    m::Dict{String, Any}
end
sweep(r::Run) = something(getpath(r.m, "provenance", "sweep_id"; default = nothing), "")
chunks(r::Run) = Int(num(r.m, "config", "sample_chunks"))
nelec(r::Run) = Int(num(r.m, "config", "N"))
vendor(r::Run) = getpath(r.m, "gpu", "backend"; default = "") == "rocm" ? :amd : :nvidia
device(r::Run) = String(getpath(r.m, "gpu", "device"; default = "unknown device"))

function read_campaign(dir)
    labels = Dict{String, String}()
    cells = joinpath(dir, "cells.tsv")
    if isfile(cells)
        for l in Iterators.drop(eachline(cells), 1)
            f = split(l, '\t')
            length(f) >= 2 && (labels[f[2]] = f[1])
        end
    end
    runs = Run[]
    for f in sort(readdir(dir))
        startswith(f, "run_") && endswith(f, ".toml") || continue
        uuid = f[5:end-5]
        m = TOML.parsefile(joinpath(dir, f))
        push!(runs, Run(uuid, get(labels, uuid, ""), m))
    end
    runs
end

# ── per-run metrics (NaN where the manifest lacks the key) ──────────────────────────────────
# Wall-clock kernel time of the field stage: [gpu].kernel_s (busiest device) net of the
# first-launch compile; [timing].kernel as the fallback of older manifests.
function kernel_seconds(r::Run)
    ks = vecfirst(r.m, "gpu", "kernel_s"); kf = vecfirst(r.m, "gpu", "kernel_first_s")
    isnan(ks) && return num(r.m, "timing", "kernel")
    isnan(kf) ? ks : ks - kf
end
slots(r::Run) = num(r.m, "flops", "slots_executed")

# Busy-phase power from the sampler trace the manifest names ([outputs].gpu_trace, a TSV with a
# `# `-prefixed header): the manifest's power_mean averages the idle setup phase in, so it sits
# well below the cap even when every busy sample is capped. Busy = compute_util ≥ 0.5.
function busy_power(r::Run, dir)
    f = getpath(r.m, "outputs", "gpu_trace"; default = nothing)
    f isa AbstractString || return (NaN, NaN)
    path = joinpath(dir, f); isfile(path) || return (NaN, NaN)
    hdr = nothing; pw = Float64[]
    for l in eachline(path)
        if startswith(l, "#")
            hdr = split(strip(l, ['#', ' ']), '\t'); continue
        end
        hdr === nothing && continue
        f = split(l, '\t'); length(f) == length(hdr) || continue
        ip = findfirst(==("power_W"), hdr); iu = findfirst(==("compute_util"), hdr)
        (ip === nothing || iu === nothing) && return (NaN, NaN)
        u = tryparse(Float64, f[iu]); p = tryparse(Float64, f[ip])
        (u === nothing || p === nothing || isnan(u) || isnan(p)) && continue
        u >= 0.5 && push!(pw, p)
    end
    isempty(pw) ? (NaN, NaN) : (median(pw), maximum(pw))
end

function power_metrics(r::Run, dir)
    g = r.m["gpu"]; fl = get(r.m, "flops", Dict())
    ks = kernel_seconds(r); sl = slots(r)
    pb, pbmax = busy_power(r, dir)
    (; C = chunks(r), N = nelec(r), uuid = r.uuid,
        ns_per_slot = ks / sl * 1e9,
        power_busy_W = pb, power_busy_max_W = pbmax,
        flop_rate = num(fl, "flop_rate_field"),
        peak_fraction = num(fl, "peak_fraction_field"),
        clock_MHz = num(g, "sampler_sm_clock_MHz_busy_median"),
        clock_peak_MHz = num(g, "sampler_sm_clock_MHz_peak"),
        power_W = num(g, "power_mean"),
        power_peak_W = num(g, "power_peak"),
        power_limit_W = num(g, "sampler_power_limit_W_busy_median"),
        capped_fraction = num(g, "sampler_power_capped_fraction"),
        ppt_fraction = num(g, "sampler_power_violation_fraction"),
        thermal_fraction = num(g, "sampler_thermal_violation_fraction"),
        hotspot_C = num(g, "sampler_hotspot_C_busy_median"),
        temperature_C = num(g, "sampler_temperature_C_busy_median"),
        gpm_fp64 = num(g, "gpm_fp64_util_busy_median"),
        gpm_sm_occupancy = num(g, "gpm_sm_occupancy_busy_median"),
        gpm_dram_bw = num(g, "gpm_dram_bw_util_busy_median"),
        gpm_sm_util = num(g, "gpm_sm_util_busy_median"),
        thread_fill = num(g, "thread_fill_occupancy"))
end

# Counter cells: the derived medians the merge writes (scripts/hw_counter_merge.jl); the
# profiler's own per-dispatch duration gives a second time-per-slot (durations, not timings:
# on ncu each dispatch is replayed once per counter pass).
function counter_metrics(r::Run)
    g = r.m["gpu"]
    st = getpath(g, "hw_counter_status"; default = "absent")
    d(k) = num(g, "hw_derived_" * k * "_median")
    dispatch_slots = num(g, "hw_slots") / num(g, "hw_dispatches")
    (; C = chunks(r), N = nelec(r), uuid = r.uuid, status = String(st),
        prof_ns_per_slot = num(g, "hw_dispatch_s_median") / dispatch_slots * 1e9,
        active_clock_GHz = d("amd_active_clock_GHz"),
        # occupancy set
        occupancy = isnan(d("amd_elapsed_occupancy")) ? d("nvidia_active_occupancy") : d("amd_elapsed_occupancy"),
        waves_per_cu = d("amd_waves_per_cu"),
        sq_busy = d("amd_sq_busy"),
        wave_wait_frac = d("amd_wave_wait_frac"),
        wave_active_valu_frac = d("amd_wave_active_valu_frac"),
        wave_active_inst_frac = d("amd_wave_active_inst_frac"),
        # fp64 set
        fp64_flop_per_slot = d("fp64_flop_per_slot"),
        fp64_fma_per_slot = d("insts_per_slot_fp64_fma"),
        fp64_mul_per_slot = d("insts_per_slot_fp64_mul"),
        fp64_add_per_slot = d("insts_per_slot_fp64_add"),
        # fp64 pipe: ncu's sm__pipe_fp64_cycles_active as a fraction of its sustained peak (NVIDIA fp64 preset)
        fp64_pipe_fraction = d("nvidia_fp64_pipe_peak_fraction"),
        # issue / l2 sets
        insts_per_slot = isnan(d("nvidia_insts_per_slot")) ? d("amd_insts_per_slot") : d("nvidia_insts_per_slot"),
        l2_hit_rate = isnan(d("nvidia_l2_sector_hit_rate")) ? d("amd_l2_hit_rate") : d("nvidia_l2_sector_hit_rate"))
end

# ── classify cells ──────────────────────────────────────────────────────────────────────────
is_power(r) = (sweep(r) in ("diag_chunks_power", "diag_power")) || (sweep(r) == "" && nelec(r) >= 1000)
is_occupancy(r) = sweep(r) == "diag_chunks_occupancy" ||
    (startswith(r.label, "occupancy") && getpath(r.m, "gpu", "hw_counter_status"; default = "") != "")
is_single(r, set) = r.label == set || (r.label == "" && occursin(set, join(something(getpath(r.m, "gpu", "hw_counters"; default = String[]), String[]), " ")))

# A probe (or GEMM reference) measured inside a counter cell ran under the profiler — ncu replays
# every dispatch once per pass, rocprofv3 adds its own overhead — and is not a peak. The card
# header therefore takes them from cells without hardware counters only.
profiled(r) = getpath(r.m, "gpu", "hw_counter_status"; default = nothing) !== nothing ||
    getpath(r.m, "gpu", "hw_counters"; default = nothing) !== nothing
function card_header(runs)
    r0 = first(runs)
    fl(r, k) = num(r.m, "flops", k)
    clean = [r for r in runs if !profiled(r)]
    probe = [fl(r, "peak_probe_flops") for r in clean if !isnan(fl(r, "peak_probe_flops"))]
    probe_clock = [fl(r, "peak_probe_clock_MHz") for r in clean if !isnan(fl(r, "peak_probe_clock_MHz"))]
    probe_capped = [fl(r, "peak_probe_power_capped_fraction") for r in clean if !isnan(fl(r, "peak_probe_power_capped_fraction"))]
    gemm = [fl(r, "peak_gemm_fp64_flops") for r in clean if !isnan(fl(r, "peak_gemm_fp64_flops"))]
    fp64 = [counter_metrics(r).fp64_flop_per_slot for r in runs if is_single(r, "fp64")]
    fp64 = filter(!isnan, fp64)
    pipe = filter(!isnan, [counter_metrics(r).fp64_pipe_fraction for r in runs if is_single(r, "fp64")])
    insts = filter(!isnan, [counter_metrics(r).insts_per_slot for r in runs if is_single(r, "issue")])
    (; device = device(r0), vendor = String(vendor(r0)),
        provider = String(something(getpath(r0.m, "provenance", "cloud_provider"; default = nothing), "local")),
        commit = String(something(getpath(r0.m, "provenance", "repo_commit"; default = nothing), "")),
        peak_probe_flops = isempty(probe) ? NaN : median(probe),
        peak_probe_clock_MHz = isempty(probe_clock) ? NaN : median(probe_clock),
        peak_probe_capped_fraction = isempty(probe_capped) ? NaN : median(probe_capped),
        peak_gemm_flops = isempty(gemm) ? NaN : median(gemm),
        algorithmic_flop_per_slot = fl(r0, "flop_per_slot"),
        counted_fp64_flop_per_slot = isempty(fp64) ? NaN : median(fp64),
        fp64_pipe_fraction = isempty(pipe) ? NaN : median(pipe),
        insts_per_slot = isempty(insts) ? NaN : median(insts),
        n_runs = length(runs), n_probe_runs = length(probe))
end

slugify(s) = lowercase(replace(replace(s, r"NVIDIA |AMD |Instinct |GeForce |Radeon |PRO " => ""), r"[^A-Za-z0-9]+" => "_"))

fmt(x; d = 3) = isnan(x) ? "—" : string(round(x; sigdigits = d))
tflops(x) = isnan(x) ? "—" : @sprintf("%.1f", x / 1e12)

# ── figure ──────────────────────────────────────────────────────────────────────────────────
function draw(hdr, P, O, S, out)
    c_time = RGBf(0.0, 0.51, 0.42); c_rate = RGBf(0.32, 0.27, 0.84); c_clock = RGBf(0.85, 0.37, 0.0)
    c_pow = RGBf(0.6, 0.0, 0.1); c_dim = RGBf(0.45, 0.45, 0.45)
    fig = Figure(size = (1250, 900))
    title = @sprintf("%s (%s) — swept FP64 probe %s TFLOP/s, GEMM %s TFLOP/s, counted %s FP64 FLOP/slot of %s algorithmic",
        hdr.device, hdr.provider, tflops(hdr.peak_probe_flops), tflops(hdr.peak_gemm_flops),
        fmt(hdr.counted_fp64_flop_per_slot; d = 4), fmt(hdr.algorithmic_flop_per_slot; d = 3))
    Label(fig[0, 1:2], title; fontsize = 14, tellwidth = false)
    Cs = sort(unique(vcat([p.C for p in P], [o.C for o in O])))
    xticks = (Cs, string.(Cs))
    xs = [p.C for p in P]
    Np = isempty(P) ? "?" : string(first(P).N)

    # (a) time per slot + achieved fraction of the probe peak
    ax1 = Axis(fig[1, 1]; xscale = log2, xticks, xlabel = "sample chunks C", ylabel = "kernel time per slot [ns]",
        title = "(a) time per slot and achieved rate  (N = $Np, sampler)")
    if !isempty(P)
        lines!(ax1, xs, [p.ns_per_slot for p in P]; color = c_time); scatter!(ax1, xs, [p.ns_per_slot for p in P]; color = c_time)
        ax1r = Axis(fig[1, 1]; xscale = log2, yaxisposition = :right, ylabelcolor = c_rate,
            ylabel = isnan(hdr.peak_probe_flops) ? "achieved FP64 [TFLOP/s]" : "achieved / probe peak")
        hidespines!(ax1r); hidexdecorations!(ax1r)
        ys = isnan(hdr.peak_probe_flops) ? [p.flop_rate / 1e12 for p in P] : [p.flop_rate / hdr.peak_probe_flops for p in P]
        lines!(ax1r, xs, ys; color = c_rate, linestyle = :dash); scatter!(ax1r, xs, ys; color = c_rate, marker = :rect)
        linkxaxes!(ax1, ax1r)
    end

    # (b) clock + power against the cap
    ax2 = Axis(fig[1, 2]; xscale = log2, xticks, xlabel = "sample chunks C", ylabel = "busy-median SM/XCD clock [MHz]",
        title = "(b) clock and power against the cap")
    if !isempty(P)
        lines!(ax2, xs, [p.clock_MHz for p in P]; color = c_clock); scatter!(ax2, xs, [p.clock_MHz for p in P]; color = c_clock)
        pk = filter(!isnan, [p.clock_peak_MHz for p in P])
        isempty(pk) || hlines!(ax2, [maximum(pk)]; color = c_clock, linestyle = :dot)
        ax2r = Axis(fig[1, 2]; xscale = log2, yaxisposition = :right, ylabel = "busy-median power [W]", ylabelcolor = c_pow)
        hidespines!(ax2r); hidexdecorations!(ax2r)
        pw = [isnan(p.power_busy_W) ? p.power_W : p.power_busy_W for p in P]
        lines!(ax2r, xs, pw; color = c_pow); scatter!(ax2r, xs, pw; color = c_pow)
        lim = filter(!isnan, [p.power_limit_W for p in P])
        if !isempty(lim)
            hlines!(ax2r, [median(lim)]; color = c_pow, linestyle = :dot)
            ylims!(ax2r, 0, 1.08 * median(lim))
        end
        linkxaxes!(ax2, ax2r)
    end

    # (c) limiter fractions from the sampler: share of busy samples at the cap, residencies, GPM pipes
    ax3 = Axis(fig[2, 1]; xscale = log2, xticks, xlabel = "sample chunks C", ylabel = "fraction",
        title = "(c) limiter fractions  (sampler / GPM, N = $Np)")
    plotted = false
    for (key, col, ls, lab) in ((:capped_fraction, c_pow, :solid, "busy samples at the power cap"),
            (:ppt_fraction, c_pow, :dash, "PPT residency (gpu_metrics)"), (:thermal_fraction, c_clock, :dash, "thermal residency"),
            (:gpm_fp64, c_rate, :solid, "GPM FP64 pipe"), (:gpm_sm_occupancy, c_time, :solid, "GPM SM occupancy"),
            (:gpm_dram_bw, c_dim, :solid, "GPM DRAM bandwidth"))
        ys = [getfield(p, key) for p in P]; (isempty(P) || all(isnan, ys)) && continue
        lines!(ax3, xs, ys; color = col, linestyle = ls, label = lab); scatter!(ax3, xs, ys; color = col); plotted = true
    end
    plotted && axislegend(ax3; position = :rt, framevisible = false, labelsize = 10)
    ylims!(ax3, 0, 1.05)

    # (d) hardware counters on the small cell
    ax4 = Axis(fig[2, 2]; xscale = log2, xticks, xlabel = "sample chunks C", ylabel = "fraction",
        title = "(d) hardware counters  (" * (hdr.vendor == "amd" ? "rocprofv3" : "ncu") * ", N = $(isempty(O) ? "?" : first(O).N))")
    plotted = false
    for (key, col, lab) in ((:occupancy, c_time, "achieved occupancy"), (:wave_active_valu_frac, c_rate, "VALU-active fraction"),
            (:wave_wait_frac, c_clock, "wave-wait fraction"), (:sq_busy, c_dim, "SQ busy"))
        ys = [getfield(o, key) for o in O]; (isempty(O) || all(isnan, ys)) && continue
        lines!(ax4, [o.C for o in O], ys; color = col, label = lab); scatter!(ax4, [o.C for o in O], ys; color = col); plotted = true
    end
    plotted && axislegend(ax4; position = :rt, framevisible = false, labelsize = 10)
    ylims!(ax4, 0, 1.05)
    if !isempty(O) && !all(isnan, [o.prof_ns_per_slot for o in O])
        ax4r = Axis(fig[2, 2]; xscale = log2, yaxisposition = :right, ylabel = "profiler time per slot [ns]", ylabelcolor = c_dim)
        hidespines!(ax4r); hidexdecorations!(ax4r)
        lines!(ax4r, [o.C for o in O], [o.prof_ns_per_slot for o in O]; color = c_dim, linestyle = :dot)
        scatter!(ax4r, [o.C for o in O], [o.prof_ns_per_slot for o in O]; color = c_dim, marker = :diamond)
        ylims!(ax4r, 0, nothing); linkxaxes!(ax4, ax4r)
    end
    save(out, fig)
end

# ── main ────────────────────────────────────────────────────────────────────────────────────
for dir in args
    runs = read_campaign(dir)
    isempty(runs) && (println(stderr, "no manifests in $dir"); continue)
    hdr = card_header(runs)
    P = sort([power_metrics(r, dir) for r in runs if is_power(r)]; by = p -> p.C)
    O = sort([counter_metrics(r) for r in runs if is_occupancy(r) && counter_metrics(r).status == "ok"]; by = o -> o.C)
    S = Dict(set => [counter_metrics(r) for r in runs if is_single(r, set)] for set in ("fp64", "issue", "memory", "l2"))
    slug = slugify(hdr.device)
    od = something(outdir, dir); mkpath(od)
    # long CSV
    open(joinpath(od, "limiter_$slug.csv"), "w") do io
        println(io, "sweep,C,N,metric,value,uuid")
        for p in P, k in propertynames(p)
            k in (:C, :N, :uuid) && continue
            println(io, "power,$(p.C),$(p.N),$k,$(getfield(p, k)),$(p.uuid)")
        end
        for o in O, k in propertynames(o)
            k in (:C, :N, :uuid, :status) && continue
            println(io, "occupancy,$(o.C),$(o.N),$k,$(getfield(o, k)),$(o.uuid)")
        end
        for (set, rows) in S, o in rows, k in propertynames(o)
            k in (:C, :N, :uuid, :status) && continue
            isnan(getfield(o, k)) && continue
            println(io, "$set,$(o.C),$(o.N),$k,$(getfield(o, k)),$(o.uuid)")
        end
    end
    nanfree(x) = x isa Number && isnan(x) ? nothing : x
    tojson(nt) = Dict(String(k) => nanfree(getfield(nt, k)) for k in propertynames(nt))
    open(joinpath(od, "limiter_$slug.json"), "w") do io
        JSON.print(io, Dict("card" => tojson(hdr), "campaign" => basename(abspath(dir)),
            "power_sweep" => tojson.(P), "occupancy_sweep" => tojson.(O),
            "single" => Dict(set => tojson.(rows) for (set, rows) in S)), 2)
    end
    png = joinpath(od, "limiter_$slug.png")
    draw(hdr, P, O, S, png)
    @printf("%-28s %-9s probe %6s TFLOP/s  GEMM %6s  counted %6s FLOP/slot  power sweep C=%s  counter sweep C=%s  → %s\n",
        hdr.device, hdr.provider, tflops(hdr.peak_probe_flops), tflops(hdr.peak_gemm_flops),
        fmt(hdr.counted_fp64_flop_per_slot; d = 4), join([p.C for p in P], ","), join([o.C for o in O], ","), png)
    for p in P
        @printf("   C=%2d  %7.2f ns/slot  %6.2f TFLOP/s  clock %6.0f MHz  power %5.0f/%4.0f W  capped %4.2f  ppt %4s  gpm_fp64 %4s  occ %4s\n",
            p.C, p.ns_per_slot, p.flop_rate / 1e12, p.clock_MHz, isnan(p.power_busy_W) ? p.power_W : p.power_busy_W, p.power_limit_W, p.capped_fraction,
            fmt(p.ppt_fraction; d = 2), fmt(p.gpm_fp64; d = 2), fmt(p.gpm_sm_occupancy; d = 2))
    end
    for o in O
        @printf("   C=%2d  counters: occupancy %5s  valu %5s  wait %5s  sq_busy %5s  prof %6.2f ns/slot\n",
            o.C, fmt(o.occupancy), fmt(o.wave_active_valu_frac), fmt(o.wave_wait_frac), fmt(o.sq_busy), o.prof_ns_per_slot)
    end
end
