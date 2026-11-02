# Multi-GPU driver for precompute_radiation.jl: one process per GPU (CUDA_VISIBLE_DEVICES=k),
# all running AT THE SAME TIME, then ONE run manifest for the run (radiation_manifest.jl merges
# the per-GPU parts: [sharding].slices = #GPUs, per-GPU detail under [parts.gpu<k>]).
#
#   EDM_BENCH=1        each GPU times the mid slice (bench) — the per-slice times include
#                      whatever the concurrent processes cost each other (host threads, PCIe,
#                      launch queues), i.e. what a real multi-GPU cube run sees.
#   EDM_RAD_BATCH > 0  each GPU computes a contiguous share of the batches into the same
#                      EDM_OUTDIR/EDM_RUN_TAG (sealed batches are skipped, so a re-run resumes).
#
# Every other EDM_* knob passes through unchanged.
# ENV: EDM_RAD_GPUS (or the legacy EDM_BENCH_GPUS)  processes (default: every GPU nvidia-smi lists)
# Logs: $EDM_OUTDIR/radiation_<tag>_gpu<k>.log; the key lines are echoed at the end.
#
# Run: julia --project=scripts animation/radiation_multigpu.jl

include(joinpath(@__DIR__, "radiation_manifest.jl"))   # HAS_MANIFEST + merge_radiation_parts

function gpu_count()
    s = get(ENV, "EDM_RAD_GPUS", get(ENV, "EDM_BENCH_GPUS", ""))
    isempty(s) || return parse(Int, s)
    try
        return count(l -> startswith(l, "GPU "), readlines(`nvidia-smi -L`))
    catch
        error("cannot count GPUs with nvidia-smi (ROCm host?): set EDM_RAD_GPUS")
    end
end
const NGPU = gpu_count()
const BENCH = get(ENV, "EDM_BENCH", "0") == "1"
const BATCH = parse(Int, get(ENV, "EDM_RAD_BATCH", "0"))
const NSLICES = parse(Int, get(ENV, "EDM_RAD_NSLICES", "128"))
BENCH || BATCH > 0 || NGPU == 1 ||
    error("a multi-GPU cube needs EDM_RAD_BATCH > 0 (the GPUs split the batches)")
const OUT = get(ENV, "EDM_OUTDIR", @__DIR__)
const TAG = get(ENV, "EDM_RUN_TAG", "local")
const SCRIPT = joinpath(@__DIR__, "precompute_radiation.jl")
const THREADS = max(1, Sys.CPU_THREADS ÷ NGPU)
mkpath(OUT)

# contiguous batch shares, as even as possible
const NB = BATCH > 0 ? cld(NSLICES, BATCH) : 0
share(k) = let q = NB ÷ NGPU, r = NB % NGPU
    lo = k * q + min(k, r) + 1
    lo:(lo + q - (k < r ? 0 : 1))
end

@info "launching $NGPU concurrent precompute processes ($THREADS threads each)" BENCH BATCH NB
logs = [joinpath(OUT, "radiation_$(TAG)_gpu$(k).log") for k in 0:(NGPU - 1)]
# a part left by an earlier attempt must never stand in for one this run failed to write
foreach(k -> rm(joinpath(OUT, "radpart_$(TAG)_gpu$(k).toml"); force = true), 0:(NGPU - 1))
t = @elapsed begin
    procs = map(0:(NGPU - 1)) do k
        env = Pair{String, String}["CUDA_VISIBLE_DEVICES" => string(k), "EDM_RAD_PART" => string(k)]
        BENCH || NB == 0 || push!(env, "EDM_RAD_BATCHES" => "$(first(share(k))):$(last(share(k)))")
        cmd = addenv(`$(Base.julia_cmd()) --startup=no --project=$(Base.active_project()) -t $THREADS $SCRIPT`, env...)
        # ONE handle for both streams: two opens of the same path truncate and overwrite each other
        io = open(logs[k + 1], "w")
        (run(pipeline(cmd; stdout = io, stderr = io); wait = false), io)
    end
    foreach(((p, io),) -> (wait(p); close(io)), procs)
end
procs = first.(procs)
# success(), not exitcode == 0: a process killed by a signal (OOM, eviction) has exitcode 0
status(p) = p.termsignal != 0 ? "killed by signal $(p.termsignal)" : "exit $(p.exitcode)"
for (k, l) in enumerate(logs)
    lines = filter(s -> occursin(r"bench:|NSUB|radiation precompute|batches .* done|manifest|ERROR|Warning", s), readlines(l))
    println("── GPU $(k - 1) ($(status(procs[k]))):")
    foreach(s -> println("   ", s), lines)
end
@info "all $NGPU processes finished in $(round(t / 60; digits = 1)) min"
if HAS_MANIFEST && all(success, procs)
    try
        @info "manifest → $(merge_radiation_parts(OUT, TAG, NGPU))"
    catch err
        @warn "manifest merge failed — the per-GPU parts are kept" exception = err
    end
end
all(success, procs) || exit(1)
