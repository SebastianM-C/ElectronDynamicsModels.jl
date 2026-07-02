#!/bin/sh
# GPU telemetry child sampler over nvidia-smi (NVML) — the CUDA counterpart of gputrace.sh;
# see there for why sampling runs out of process. Emits the same canonical TSV rows on STDOUT:
#   epoch_s <TAB> device <TAB> power_W <TAB> compute_util <TAB> mem_util <TAB> vram_used_B
# (epoch at whole-second resolution — awk systime(); fine for the >=1 s production cadence).
#
# Usage: gputrace_cuda.sh <dt_ms> <parent_pid> <stopfile> <uuid=ordinal>...
#   uuid = the GPU-... NVML uuid (stable under CUDA_VISIBLE_DEVICES); ordinal = our 1-based id
# One nvidia-smi daemon per device; all are reaped when <stopfile> appears or the parent dies.
dt_ms=$1; ppid=$2; stop=$3; shift 3

pids=""
for spec in "$@"; do
    uuid=${spec%%=*}; ord=${spec#*=}
    nvidia-smi -i "$uuid" \
        --query-gpu=power.draw,utilization.gpu,utilization.memory,memory.used \
        --format=csv,noheader,nounits -lms "$dt_ms" 2>/dev/null \
        | awk -v dev="$ord" -F', *' \
            '{ printf "%d\t%s\t%.1f\t%.2f\t%.2f\t%.0f\n", systime(), dev, $1, $2 / 100, $3 / 100, $4 * 1048576; fflush() }' &
    pids="$pids $!"
done

while [ ! -e "$stop" ] && kill -0 "$ppid" 2>/dev/null; do
    sleep 1
done
kill $pids 2>/dev/null
wait
