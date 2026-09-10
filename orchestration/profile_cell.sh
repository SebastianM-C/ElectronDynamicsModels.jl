#!/usr/bin/env bash
# profile_cell.sh — run ONE solver cell under rocprofv3 hardware counters (AMD, ROCm ≥ 6.2) and
# merge the counters into the cell's run manifest as [gpu].rocprof_* keys. The GPM sampler does
# this in-process on NVIDIA; AMD has no in-process counter API, so the whole solver process is
# wrapped by the profiler (GPUDiagnostics.jl `rocprof_command`, under `timeout -k` because a
# counter set the hardware refuses aborts rocprofv3 with signal 6 and leaves the child hung) and
# the CSV it writes is parsed afterwards (`rocprof_counters` → scripts/rocprof_merge.jl).
#
#   bash orchestration/profile_cell.sh <set|"COUNTER …"> <outdir> <tag> [EDM_VAR=val …]
#
#   <set>     a name from GPUDiagnostics.ROCPROF_COUNTER_SETS (sq_issue, sq_waves, l1_pipe, fp64,
#             l2) or a space-separated list of rocprofv3 counter names (one pass: what the hardware
#             can collect at once — `rocprofv3 --list-avail` names them)
#   <outdir>  EDM_OUTDIR of the cell; receives run_<tag>.toml, <tag>_counter_collection.csv,
#             <tag>_kernel_trace.csv, <tag>_agent_info.csv and rocprof_<tag>.log
#   <tag>     EDM_RUN_TAG of the cell (= the rocprofv3 output prefix)
#   EDM_*     the cell's environment, exactly as run_cell.sh would receive it (the backend env
#             such as ROCR_VISIBLE_DEVICES / EDM_GPU_BACKEND=rocm goes here too)
#
# Env: SCRIPT (default scripts/inverse_thomson_scattering.jl), JL (julia launcher array, default
#      `julia --startup=no -t auto`), PROFILE_TIMEOUT (s, default 600 — the whole cell, JIT
#      included), PROFILE_KERNEL (regex selecting the profiled kernel in the CSV, default
#      forindices = the KernelAbstractions field kernel). Exit status is the merge's; the
#      profiler's own status is in the log. One counter set per run — profile the same cell once
#      per set (the counters are per dispatch, so runs are directly comparable).
set -uo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$ORCH/.." && pwd)"
[ "$#" -ge 3 ] || { sed -n '2,27p' "$0"; exit 64; }
: "${SCRIPT:=scripts/inverse_thomson_scattering.jl}"
: "${PROFILE_TIMEOUT:=600}"
: "${PROFILE_KERNEL:=forindices}"
[ -n "${JL[*]:-}" ] || JL=(julia --startup=no -t auto)
set_or_counters=$1; out=$2; tag=$3; shift 3
mkdir -p "$out"
command -v rocprofv3 >/dev/null || { echo "[profile] rocprofv3 not on PATH (ROCm ≥ 6.2)" >&2; exit 69; }

echo "[profile] $tag under rocprofv3 --pmc ($set_or_counters) → $out  (timeout ${PROFILE_TIMEOUT}s)"
( cd "$REPO" && "${JL[@]}" -t 1 --project=scripts -e '
    using GPUDiagnostics
    spec, dir, name, timeout_s = ARGS[1], ARGS[2], ARGS[3], parse(Float64, ARGS[4])
    sym = Symbol(strip(replace(spec, ":" => "")))
    counters = haskey(ROCPROF_COUNTER_SETS, sym) ? sym : String.(split(spec))
    cmd = rocprof_command(Cmd(String.(ARGS[5:end])); counters, dir, name, timeout_s)
    println("[profile] ", cmd); flush(stdout)
    run(cmd)' \
    "$set_or_counters" "$out" "$tag" "$PROFILE_TIMEOUT" \
    env "$@" EDM_OUTDIR="$out" EDM_RUN_TAG="$tag" "${JL[@]}" --project=scripts "$SCRIPT" ) \
    > "$out/rocprof_$tag.log" 2>&1 || echo "[profile] profiler exited nonzero — see $out/rocprof_$tag.log" >&2
grep -iE 'exceeds the capabilities|caught signal' "$out/rocprof_$tag.log" | head -3 >&2 || true

( cd "$REPO" && "${JL[@]}" -t 1 --project=scripts scripts/rocprof_merge.jl "$out" "$tag" --kernel="$PROFILE_KERNEL" )
