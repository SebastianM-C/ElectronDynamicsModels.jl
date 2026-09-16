# campaigns/diag_validate_nvidia_counters.sh — the counter cells of diag_validate_nvidia.sh only
# (occupancy_c{1,2,4,8,16}, fp64, issue, memory, l2), for rerunning them on a kept pod under the
# same DIAG_CAMPAIGN after a collector-setup failure (the power cells stand). Same sweep ids and
# labels, so the published campaign is indistinguishable from a single pass.
#   RUNPOD_STATE=~/.config/runpod/pod_h200 DIAG_CAMPAIGN=diag_validate_h200sxm \
#     bash orchestration/backends/runpod.sh run orchestration/campaigns/diag_validate_nvidia_counters.sh
. "$(dirname "${BASH_SOURCE[0]}")/diag_validate_nvidia.sh"
_all=("${CELLS[@]}"); CELLS=()
for c in "${_all[@]}"; do case "$c" in power_c*) ;; *) CELLS+=("$c") ;; esac; done
unset _all
