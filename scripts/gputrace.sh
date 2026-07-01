#!/bin/sh
# GPU telemetry child sampler over amdgpu driver sysfs — spawned by with_gpu_sampler
# (scripts/gpu_telemetry.jl). Runs OUT OF PROCESS because no in-process Julia sampler
# survives the solver: GC + libuv-timer coupling suspends sleeping tasks entirely while
# the host thread allocates (measured: 0 ticks/15 s on pure CPU churn, 1 tick/98.5 s on a
# real accumulate_field window).
#
# Emits one TSV row per device per tick on STDOUT (the parent redirects into the gputrace
# TSV next to the run outputs):
#   epoch_s <TAB> device <TAB> power_W <TAB> compute_util <TAB> mem_util <TAB> vram_used_B
#
# Usage: gputrace.sh <dt_s> <parent_pid> <stopfile> <devspec>...
#   devspec = id:power_file:busy_file:membusy_file:vram_file  (membusy_file "-" if absent)
# Exits when <stopfile> appears or the parent dies — cooperative stop, no signals, no orphans.
dt=$1; ppid=$2; stop=$3; shift 3

while [ ! -e "$stop" ] && kill -0 "$ppid" 2>/dev/null; do
    now=$(date +%s.%N)
    for spec in "$@"; do
        awk -v now="$now" -v spec="$spec" 'BEGIN {
            split(spec, a, ":")
            if ((getline p < a[2]) <= 0) exit   # transient read failure -> drop this row
            if ((getline b < a[3]) <= 0) exit
            m = "nan"
            if (a[4] != "-" && (getline mm < a[4]) > 0) m = mm / 100
            if ((getline v < a[5]) <= 0) exit
            printf "%.2f\t%s\t%.1f\t%.2f\t%s\t%.0f\n", now, a[1], p / 1e6, b / 100, m, v
        }'
    done
    sleep "$dt"
done
