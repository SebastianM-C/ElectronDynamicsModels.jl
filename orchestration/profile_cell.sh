#!/usr/bin/env bash
# profile_cell.sh — run ONE solver cell under per-dispatch hardware counters (rocprofv3 on ROCm,
# Nsight Compute on CUDA; the collector follows EDM_GPU_BACKEND) and merge the counters into the
# cell's run manifest as [gpu].hw_* keys, registering the collector's files in [outputs]. The GPM
# sampler gives pipe utilisations in-process on Hopper; per-dispatch instruction counts need the
# whole solver process wrapped by the profiler (GPUDiagnostics.jl `hw_counter_command`, under
# `timeout -k` because a counter set the hardware refuses aborts rocprofv3 with signal 6 and
# leaves the child hung) and the CSV it writes is parsed afterwards (`hw_counters` →
# scripts/hw_counter_merge.jl). run_cell.sh calls this for a cell carrying EDM_PROFILE=<set>.
#
#   bash orchestration/profile_cell.sh <set|"COUNTER …"> <outdir> <tag> [EDM_VAR=val …]
#
#   <set>     an intent name from GPUDiagnostics.COUNTER_SETS[vendor] (issue, occupancy, memory,
#             fp64, l2; the 0.2 names sq_issue / sq_waves / l1_pipe are accepted as aliases) or a
#             space-separated list of native counter names (one pass: what the hardware can
#             collect at once — `rocprofv3 --list-avail` / `ncu --query-metrics` name them).
#             Device-qualified names (SQ_WAVES:device=0) pass through and are what rocprofv3
#             needs when it enumerates an unsupported GPU beside the target one
#   <outdir>  EDM_OUTDIR of the cell; receives run_<tag>.toml and the collector's files:
#             <tag>_counter_collection.csv, <tag>_kernel_trace.csv, <tag>_agent_info.csv and
#             rocprof_<tag>.log on ROCm; <tag>_ncu.csv, <tag>.ncu-rep and ncu_<tag>.log on CUDA
#   <tag>     EDM_RUN_TAG of the cell (= the rocprofv3 output prefix)
#   EDM_*     the cell's environment, exactly as run_cell.sh would receive it (the backend env
#             such as ROCR_VISIBLE_DEVICES / EDM_GPU_BACKEND=rocm goes here too)
#
# Env: GPUDIAGNOSTICS_COUNTER_DEVICE (rocprofv3: qualify every counter with :device=N — needed
#      when the runtime also enumerates a GPU the tool cannot profile, e.g. an iGPU),
#      SCRIPT (default scripts/inverse_thomson_scattering.jl), JL (julia launcher array, default
#      `julia --startup=no -t auto`), PROFILE_TIMEOUT (s, default 600 — the whole cell, JIT
#      included), PROFILE_KERNEL (regex selecting the profiled kernel in the CSV, default
#      forindices = the KernelAbstractions field kernel). Exit status is the merge's; the
#      profiler's own status is in the log. One counter set per run — profile the same cell once
#      per set (the counters are per dispatch, so runs are directly comparable).
#
# Instinct parts collect in the default performance state, but RDNA (gfx11/gfx12) needs STABLE_STD
# set on the target GPU first or the GRBM/SQ cycle counters read zero while wave counts look fine —
# GPUDiagnostics' hardware-counter guide has the sysfs/amd-smi recipe and why it must be restored.
# Instruction counts (the fp64 set) are clock-independent on every vendor, so the FP64-per-slot
# figure never needs a performance-level change. NVIDIA drivers with RmProfilingAdminOnly=1 refuse
# counter collection to non-admin users (ERR_NVGPUCTRPERM in the log); GPM sampling still works.
set -uo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$ORCH/.." && pwd)"
[ "$#" -ge 3 ] || { sed -n '2,32p' "$0"; exit 64; }
: "${SCRIPT:=scripts/inverse_thomson_scattering.jl}"
: "${PROFILE_TIMEOUT:=600}"
: "${PROFILE_KERNEL:=forindices}"
# The launcher arrives either as the bash array JL (sourced callers) or, from run_cell.sh
# for an EDM_PROFILE cell, as the space-joined string JL_CMD in the environment (an array
# cannot be exported): split the string, or the whole launcher line is looked up as one
# command (exit 127).
[ -n "${JL_CMD:-}" ] && read -r -a JL <<< "$JL_CMD"
[ -n "${JL[*]:-}" ] || JL=(julia --startup=no -t auto)
set_or_counters=$1; out=$2; tag=$3; shift 3
mkdir -p "$out"
# The collector: rocprofv3 (ROCm) or ncu (CUDA), from EDM_GPU_BACKEND in the cell's environment
# (default rocm — the path that existed before CUDA support). GPUDIAGNOSTICS_COUNTER_TOOL points
# at an ncu that is not on PATH (the Nsight Compute installs under /opt/nvidia are not).
backend=rocm
for kv in "$@"; do case "$kv" in EDM_GPU_BACKEND=*) backend="${kv#EDM_GPU_BACKEND=}" ;; esac; done
case "$backend" in
    rocm) collector=RocprofV3; tool=rocprofv3; logname="rocprof_$tag.log" ;;
    cuda) collector=NsightCompute; tool="${GPUDIAGNOSTICS_COUNTER_TOOL:-ncu}"; logname="ncu_$tag.log" ;;
    *) echo "[profile] no hardware-counter collector for backend $backend" >&2; exit 69 ;;
esac
command -v "$tool" >/dev/null || { echo "[profile] $tool not on PATH (rocprofv3: ROCm ≥ 6.2; ncu: set GPUDIAGNOSTICS_COUNTER_TOOL)" >&2; exit 69; }
tool_path=$(command -v "$tool")

echo "[profile] $tag under $tool ($set_or_counters) → $out  (timeout ${PROFILE_TIMEOUT}s)"
( cd "$REPO" && "${JL[@]}" -t 1 --project=scripts -e '
    using GPUDiagnostics
    spec, dir, name, timeout_s = ARGS[1], ARGS[2], ARGS[3], parse(Float64, ARGS[4])
    collector, executable = ARGS[5] == "NsightCompute" ? (NsightCompute(), ARGS[6]) : (RocprofV3(), nothing)
    vendor = collector isa NsightCompute ? :nvidia : :amd
    aliases = Dict(:sq_issue => :issue, :sq_waves => :occupancy, :l1_pipe => :memory)  # 0.2 set names
    sym = Symbol(replace(strip(spec), r"^:" => ""))   # no single quotes: this code sits inside a bash single-quoted string
    sym = get(aliases, sym, sym)
    sel = haskey(COUNTER_SETS[vendor], sym) ? (; set = sym) : (; metrics = String.(split(spec)))
    # rocprofv3 enumerates every GPU the runtime sees; on a host with a second, unsupported
    # part (an integrated GPU beside a workstation card) an unqualified counter set aborts the
    # tool (unordered_map::at). GPUDIAGNOSTICS_COUNTER_DEVICE=N qualifies every name with
    # :device=N, the same knob the GPUDiagnostics hardware suite uses.
    qual = get(ENV, "GPUDIAGNOSTICS_COUNTER_DEVICE", "")
    if collector isa RocprofV3 && !isempty(qual)
        ms = haskey(sel, :set) ? COUNTER_SETS[:amd][sel.set].metrics : sel.metrics
        sel = (; metrics = [occursin(":device=", m) ? m : m * ":device=" * qual for m in ms])
    end
    # The workload arrives as `env K=V … julia …`: fold the assignments into the Cmd
    # environment (addenv inherits the rest) so the profiled process IS the julia process:
    # ncu profiles only the process it launches unless told to follow children, and rocprofv3
    # follows them anyway. :application keeps the sampler child and precompile workers out.
    # (No single quotes anywhere in this snippet: it sits inside a bash single-quoted string.)
    words = String.(ARGS[7:end])
    words[1] == "env" && popfirst!(words)
    nenv = findfirst(w -> !occursin(r"^[A-Za-z_][A-Za-z0-9_]*=", w), words) - 1
    workload = addenv(Cmd(words[(nenv + 1):end]), words[1:nenv])
    extra = collector isa NsightCompute ? (; executable, target_processes = :application) : (;)
    cmd = hw_counter_command(collector, workload; dir, name, timeout_s, sel..., extra...)
    println("[profile] ", cmd); flush(stdout)
    run(cmd)' \
    "$set_or_counters" "$out" "$tag" "$PROFILE_TIMEOUT" "$collector" "$tool_path" \
    env "$@" EDM_OUTDIR="$out" EDM_RUN_TAG="$tag" "${JL[@]}" --project=scripts "$SCRIPT" ) \
    > "$out/$logname" 2>&1 || echo "[profile] profiler exited nonzero — see $out/$logname" >&2
grep -iE 'exceeds the capabilities|caught signal|unordered_map::at|ERR_NVGPUCTRPERM|not permitted' "$out/$logname" | head -3 >&2 || true

( cd "$REPO" && "${JL[@]}" -t 1 --project=scripts scripts/hw_counter_merge.jl "$out" "$tag" --kernel="$PROFILE_KERNEL" )
