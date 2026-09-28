# campaigns/verda_inverse_lanes.sh — helpers for the inverse-Thomson Verda lanes (verda_llps_*.sh, verda_a5_*.sh):
# several lanes on ONE multi-GPU VM, each pinned to a card, sharing one host RAM and one reducer slot.
# Sourced by a lane file AFTER its recipe (BASE/CELLS set). Lane files must sit directly in campaigns/
# (verda.sh launches campaigns/<basename>), and their stems must be unique (runs/<stem>.out|.pid).

# _lane_gpu <index>: pin the lane's solver to one card (the reducer is CPU-only and ignores it).
_lane_gpu() { BASE+=(CUDA_VISIBLE_DEVICES="$1"); }

# _lane_cells "<label> ...": keep only these cells, in the given order (the lane's run order).
_lane_cells() {
    local want=$1 l c _all=("${CELLS[@]}"); CELLS=()
    for l in $want; do for c in "${_all[@]}"; do [ "${c%%|*}" = "$l" ] && CELLS+=("$c"); done; done
    [ "${#CELLS[@]}" -eq "$(wc -w <<< "$want")" ] || { echo "_lane_cells: unknown label in '$want'" >&2; return 1; }
}

# _lane_reduce <uuid>: the REDUCE_HOOK for every lane on the VM.
#   • one reduce at a time VM-wide (flock on runs/.reduce.lock): a reduce peaks at ~1.5× cube with O_DIRECT
#   • RAM gate: start only when MemAvailable ≥ 1.5× cube + 8 GiB (a cube landing elsewhere waits it out; ≤ 30 min)
#   • oom_score_adj 1000: if two landings and a reduce ever coincide, the kernel kills the reducer, not a solver
#   • one retry after a failure (the cube is kept either way; a second failure leaves <uuid>.reduce_failed)
_lane_reduce() {
    local uuid=$1 cube need t=0 avail
    export EDM_DIRECT_READ=1
    echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
    {
        flock 9
        cube=$(find "$CAMP" -maxdepth 1 -name "field_*_${uuid}.jls" -printf '%s\n' | head -1)
        need=$(( ${cube:-0} * 3 / 2 + 8 * 2**30 ))
        for try in 1 2; do
            while avail=$(awk '/^MemAvailable:/{print $2 * 1024}' /proc/meminfo) && [ "$avail" -lt "$need" ] && [ "$t" -lt 1800 ]; do
                [ $((t % 300)) -eq 0 ] && echo "[$(date -u +%FT%TZ)] reduce $uuid waits for RAM: $((avail / 2**30)) of $((need / 2**30)) GiB" >> "$CAMP/run_${uuid}.log"
                sleep 30; t=$((t + 30))
            done
            _reduce_cell "$uuid"
            [ -f "$CAMP/${uuid}.reduce_failed" ] || break
            [ "$try" = 1 ] && { echo "[$(date -u +%FT%TZ)] reduce $uuid failed — retrying once" >> "$CAMP/run_${uuid}.log"; rm -f "$CAMP/${uuid}.reduce_failed"; t=0; }
        done
    } 9> "$REPO/runs/.reduce.lock"
}
REDUCE_HOOK='_lane_reduce "$uuid"'

# _lane_after <stem> <n>: VM-only — block until lane <stem> has finished <n> field phases (its card is then
# free) or its driver is gone. No-op on the driver (verda.sh sources lane files to read CAMPAIGN) and when
# that lane is not running. List the waiting lane AFTER <stem> on the verda.sh command line.
_lane_after() {
    local stem=$1 n=$2 out pid
    [ "$(basename "$0")" = local.sh ] || return 0
    out="$REPO/runs/$stem.out"; pid=$(cat "$REPO/runs/$stem.pid" 2>/dev/null) || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    echo "[$(date -u +%FT%TZ)] waiting for lane $stem to finish $n field phase(s)"
    while kill -0 "$pid" 2>/dev/null && [ "$(grep -cE '^  (field done|done|FAILED) ' "$out" 2>/dev/null)" -lt "$n" ]; do sleep 60; done
    echo "[$(date -u +%FT%TZ)] lane $stem released the card"
}
