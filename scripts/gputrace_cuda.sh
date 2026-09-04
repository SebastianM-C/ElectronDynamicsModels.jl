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

# mawk (Ubuntu's default awk, i.e. every RunPod CUDA image) block-buffers its INPUT from a pipe
# unless told otherwise: at ~30 B/row and 1 Hz it sat on nvidia-smi's rows for the whole field
# phase and the trace came back empty ("sampler starved: 0 ticks", 8×H200 smoke 2026-09-04).
# `-W interactive` makes mawk line-buffer reads (and unbuffer writes); gawk reads pipes
# record-by-record already and fflush() covers its output.
if awk -W version 2>/dev/null | grep -qi mawk; then AWK="awk -W interactive"; else AWK=awk; fi

# One writer per device would interleave on the shared stdout: `-W interactive` also makes
# mawk's writes unbuffered (several write() calls per row), and eight of them tore rows mid-line
# (8×H200 smoke bdcbccc5: 114/3631 rows torn, a glued VRAM+epoch number reached the manifest as a
# 3e20 B "peak"). So each device streams into its own temp file (single writer, no tearing) and
# the parent concatenates them to stdout — the only shared writer — once at stop time.
tmpd=$(mktemp -d "${TMPDIR:-/tmp}/gputrace.XXXXXX") || exit 1
trap 'rm -rf "$tmpd"' EXIT

pids=""
for spec in "$@"; do
    uuid=${spec%%=*}; ord=${spec#*=}
    nvidia-smi -i "$uuid" \
        --query-gpu=power.draw,utilization.gpu,utilization.memory,memory.used \
        --format=csv,noheader,nounits -lms "$dt_ms" 2>/dev/null \
        | $AWK -v dev="$ord" -F', *' \
            '{ printf "%d\t%s\t%.1f\t%.2f\t%.2f\t%.0f\n", systime(), dev, $1, $2 / 100, $3 / 100, $4 * 1048576; fflush() }' \
        > "$tmpd/dev_$ord.tsv" &
    pids="$pids $!"
done

while [ ! -e "$stop" ] && kill -0 "$ppid" 2>/dev/null; do
    sleep 1
done
kill $pids 2>/dev/null
wait
# Emit complete rows only (a writer killed mid-line could leave a partial last line).
for f in "$tmpd"/dev_*.tsv; do [ -e "$f" ] && awk 'NF == 6' "$f"; done
