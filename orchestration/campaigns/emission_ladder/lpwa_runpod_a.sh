# cloud lane lpwa_runpod_a (H100 NVL/H200)
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_lpwa.sh"
_keep=" a1em2 a1 a2 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%|*} "* ]] && CELLS+=("$c"); done
