# cloud lane lpwa_verda (H100 SXM .32V): low a0, next to the local numeric twins on NVIDIA
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_lpwa.sh"
_keep=" a1em5 a1em4 a1em3 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%|*} "* ]] && CELLS+=("$c"); done
