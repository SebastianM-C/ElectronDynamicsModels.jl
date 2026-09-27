# Cloud LPWA queue lane: `. lpwa_queue.sh <lane>` keeps the base-recipe cells named in queues/<lane>.txt (one label per
# line, run in that order). The driver ships orchestration/ at every `run`, so a queue edited before a launch moves cells
# between lanes without re-provisioning; a cell a RUNNING lane has not started yet is released with a skip marker
# (run_cells: $CAMP/skip_<label>). Lane files live at campaigns/ top level (the cloud backends launch by basename).
_q="$(dirname "${BASH_SOURCE[0]}")/queues/${1:?usage: . lpwa_queue.sh <lane>}.txt"
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_lpwa.sh"
_all=("${CELLS[@]}"); CELLS=()
while read -r _l _; do
    [ -n "$_l" ] && [ "${_l:0:1}" != "#" ] || continue
    _hit=""; for _c in "${_all[@]}"; do [ "${_c%%|*}" = "$_l" ] && { CELLS+=("$_c"); _hit=1; }; done
    [ -n "$_hit" ] || echo "lpwa_queue: $_l (queue $_q) is not a cell of emission_ladder_lpwa — ignored" >&2
done < "$_q"
