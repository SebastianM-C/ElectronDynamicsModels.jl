# cloud lane lpwa_runpod_b (H100 NVL/H200): a0 ≥ 5 (n_substeps NS_HIGH)
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_lpwa.sh"
_keep=" a5 a10 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%|*} "* ]] && CELLS+=("$c"); done
