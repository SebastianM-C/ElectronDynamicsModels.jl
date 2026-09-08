# Static instruction mix of the production field kernel — natively, or CROSS-COMPILED for a GPU
# that is not here (MI300X = gfx942, H100/H200 = sm_90) — printed as a table. An offline analysis
# tool: the manifests only carry the native totals (`[gpu].kernel_mix_*`, via gpu_telemetry.jl).
#
#   julia --project=scripts scripts/instruction_mix.jl [--target gfx942|sm_90] [--alg newton|rk4]
#       [--mode total|split] [--dump FILE] [--slots N] [--kernel-s T] [--peak FLOPS] [--n-iters 2]
#
# EDM_GPU_BACKEND=cuda (default) | rocm selects the vendor, as in thomson_scattering.jl. The kernel
# is compiled by one launch on a tiny analytic trajectory (the machine code does not depend on the
# data; the same closure the production drivers launch, with the same options — CUDA
# `always_inline = true`, the static 256-thread workgroup). `--dump` saves the disassembly.
# `--slots` (executed hot-loop iterations per launch, e.g. Nx·Ny·N_samples when every slot lies in
# the window) turns the hot loop's FP64 count into the FP64-issue floor per launch; `--peak` is the
# FP64 rate to use in FLOP/s (default: `measure_peak_fp64_flops` on the current device — for a
# cross-compiled target pass the target's measured peak); `--kernel-s` a measured median launch
# time for the fraction.

using ElectronDynamicsModels
using GPUDiagnostics
using StaticArrays, DataInterpolations

const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "cuda"))
if GPU_BACKEND == "cuda"
    using CUDA
    const backend = CUDA.CUDABackend(; always_inline = true)   # matches the production scripts
elseif GPU_BACKEND == "rocm"
    using AMDGPU
    const backend = AMDGPU.ROCBackend()
else
    error("EDM_GPU_BACKEND must be \"cuda\" or \"rocm\", got $(repr(GPU_BACKEND))")
end

function parse_args(args)
    opts = Dict{String, Any}("target" => nothing, "alg" => "newton", "mode" => "total", "dump" => nothing,
        "slots" => nothing, "kernel-s" => nothing, "peak" => nothing, "n-iters" => 2)
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument $a")
        key = a[3:end]
        haskey(opts, key) || error("unknown option $a")
        i + 1 <= length(args) || error("$a needs a value")
        v = args[i + 1]
        opts[key] = key in ("slots", "kernel-s", "peak") ? parse(Float64, v) : key == "n-iters" ? parse(Int, v) : v
        i += 2
    end
    return opts
end
const opts = parse_args(ARGS)

# One launch of the production field kernel on an analytic worldline (helix-like drift) so it
# lands in the vendor's compiled-kernel cache.
function analytic_traj(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0, τspan = (0.0, 20.0), N = 600, K = 1.0)
    ts = collect(range(τspan[1], τspan[2], length = N))
    us = [SVector{8}(g * τ, A * sin(Ω * τ), 0.0, vz * τ, g, A * Ω * cos(Ω * τ), 0.0, vz) for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    as = [SVector{4}(0.0, -A * Ω^2 * sin(Ω * τ), 0.0, 0.0) for τ in ts]
    a_itp = CubicSpline(as, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4), SVector{4, Int}(5, 6, 7, 8), K)
end
alg = opts["alg"] == "newton" ? GPUKernelNewton() : opts["alg"] == "rk4" ? GPUKernelRK4() : error("--alg newton|rk4")
mode = opts["mode"] == "total" ? Val(:total) : opts["mode"] == "split" ? Val(:split) : error("--mode total|split")
screen = ObserverScreen(LinRange(-1.0, 1.0, 8), LinRange(-1.0, 1.0, 8), 30.0, range(30.0, 50.0; length = 64); c = 1.0)
kw = alg isa GPUKernelNewton ? (; n_iters = opts["n-iters"]) : (;)
accumulate_field([analytic_traj()], screen, alg, backend; mode, kw...)

pattern = alg isa GPUKernelNewton ? r"_gpu_newton_field_one_electron!" : r"_gpu_unified_field_one_electron!"
ck = only(compiled_kernels(backend; pattern))
t = @elapsed mix = kernel_instruction_mix(backend, ck; target = opts["target"], dump = opts["dump"])

println("kernel   ", ck.name, "  (", something(match(pattern, ck.signature), (; match = "?")).match, ", workgroup ", ck.workgroup_size, ")")
println("device   ", gpu_name(backend), "  (", gpu_arch(backend), ")")
println("target   ", mix.target, mix.native ? "  [native: this is the code that runs here]" : "  [CROSS-COMPILED: not this device's code]",
    "   registers ", something(mix.registers, "n/a (see kernel_resources)"), "   ", round(t; digits = 1), " s")
println("blocks   ", mix.blocks, "   loops ", length(mix.loops), "   hot loop ", mix.hot_loop === nothing ? "none" : mix.hot_loop.header,
    " (confidence ", mix.hot_loop_confidence, mix.llvm_loops_agree === nothing ? "" : ", LLVM loop annotations " * (mix.llvm_loops_agree ? "agree" : "DISAGREE"), ")")
println()
cols = [("total (static)", mix.counts)]
mix.hot_loop === nothing || push!(cols, ("hot loop", mix.hot_loop.counts), ("hot loop excl.", mix.hot_loop.exclusive_counts))
for l in mix.loops
    (mix.hot_loop !== nothing && l.header == mix.hot_loop.header) && continue
    push!(cols, ("$(l.header) d$(l.depth)", l.counts))
end
println(rpad("class", 14), join([lpad(first(c), max(16, length(first(c)) + 2)) for c in cols]))
for c in MIX_CLASSES
    println(rpad(String(c), 14), join([lpad(string(cnt[c]), max(16, length(nm) + 2)) for (nm, cnt) in cols]))
end
println(rpad("fp64 (all)", 14), join([lpad(string(sum(cnt[k] for k in GPUDiagnostics.FP64_CLASSES)), max(16, length(nm) + 2)) for (nm, cnt) in cols]))
println(rpad("TOTAL", 14), join([lpad(string(sum(cnt)), max(16, length(nm) + 2)) for (nm, cnt) in cols]))
println()
println("loop nest (largest first; total = one pass incl. nested loops counted once, excl = own blocks):")
for l in mix.loops
    println("  ", rpad(l.header, 12), " depth ", l.depth, "  blocks ", lpad(l.blocks, 4), "  total ", lpad(l.total, 6), "  excl ", lpad(l.exclusive_total, 6),
        "  fp64 ", lpad(sum(l.counts[k] for k in GPUDiagnostics.FP64_CLASSES), 5), "  waits ", lpad(l.counts.wait, 4), "  loads ", lpad(l.counts.mem_load, 4))
end
other = sort!([(k, v) for (k, v) in mix.opcodes if (mix.vendor === :amd ? GPUDiagnostics._classify_amd(k) : GPUDiagnostics._classify_sass(k)) === :other]; by = x -> -x[2])
isempty(other) || println("\n'other' = ", join(["$k×$v" for (k, v) in other[1:min(end, 10)]], ", "))

if opts["slots"] !== nothing && mix.hot_loop !== nothing
    peak = opts["peak"]
    if peak === nothing
        mix.native || error("--peak is required for a cross-compiled target (the current device's FP64 rate is not the target's)")
        peak = measure_peak_fp64_flops(backend)
    end
    fl = fp64_issue_floor(mix; n_slots = opts["slots"], peak_fp64_flops = peak, kernel_time_s = opts["kernel-s"])
    println("\nFP64-issue floor: ", fl.fp64_per_slot, " FP64 instructions per hot-loop pass × ", opts["slots"], " slots ÷ (",
        round(peak / 1e12; digits = 3), " TFLOP/s ÷ 2) = ", round(fl.floor_s * 1e3; digits = 2), " ms per launch",
        fl.kernel_time_s === nothing ? "" : " = $(round(100 * fl.fp64_issue_fraction; digits = 1)) % of the $(fl.kernel_time_s) s launch")
    println("  (", fl.assumptions, ")")
end
