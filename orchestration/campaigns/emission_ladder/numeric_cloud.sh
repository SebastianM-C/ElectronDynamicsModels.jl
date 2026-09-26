# cloud lane numeric_cloud: numeric a0 = 5 after the a0 = 10 sizing cell passes the window gate; drop a5 from numeric_high if this lane runs it
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a5 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%|*} "* ]] && CELLS+=("$c"); done
