# Multi-GPU throughput benchmark for precompute_radiation.jl: launches one EDM_BENCH=1 process
# per GPU (CUDA_VISIBLE_DEVICES=k) AT THE SAME TIME, so the per-slice times include whatever the
# processes cost each other (host threads, PCIe, launch queues), i.e. the throughput a real
# multi-GPU cube run (disjoint EDM_RAD_BATCHES per GPU) would see. Every other EDM_* knob in the
# environment passes through unchanged.
#
# ENV: EDM_BENCH_GPUS   processes to launch (default: every GPU nvidia-smi lists)
# Logs: $EDM_OUTDIR/bench_<tag>_gpu<k>.log; the bench lines are echoed at the end.
#
# Run: julia --project=scripts animation/bench_multigpu.jl

const NGPU = let s = get(ENV, "EDM_BENCH_GPUS", "")
    isempty(s) ? count(l -> startswith(l, "GPU "), readlines(`nvidia-smi -L`)) : parse(Int, s)
end
const OUT = get(ENV, "EDM_OUTDIR", @__DIR__)
const TAG = get(ENV, "EDM_RUN_TAG", "local")
const SCRIPT = joinpath(@__DIR__, "precompute_radiation.jl")
const THREADS = max(1, Sys.CPU_THREADS ÷ NGPU)
mkpath(OUT)

@info "launching $NGPU concurrent bench processes ($THREADS threads each)"
logs = [joinpath(OUT, "bench_$(TAG)_gpu$(k).log") for k in 0:(NGPU - 1)]
t = @elapsed begin
    procs = map(0:(NGPU - 1)) do k
        cmd = addenv(`$(Base.julia_cmd()) --startup=no --project=$(Base.active_project()) -t $THREADS $SCRIPT`,
            "CUDA_VISIBLE_DEVICES" => string(k), "EDM_BENCH" => "1")
        run(pipeline(cmd; stdout = logs[k + 1], stderr = logs[k + 1]); wait = false)
    end
    foreach(wait, procs)
end
for (k, l) in enumerate(logs)
    lines = filter(s -> occursin(r"bench:|NSUB|radiation precompute|ERROR", s), readlines(l))
    println("── GPU $(k - 1) (exit $(procs[k].exitcode)):")
    foreach(s -> println("   ", s), lines)
end
@info "all $NGPU bench processes finished in $(round(t / 60; digits = 1)) min"
any(p -> p.exitcode != 0, procs) && exit(1)
