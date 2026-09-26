# lane numeric_high: a0 = 2, 5 (n_substeps NS_HIGH); a0 = 5 only after the a0 = 10 sizing cell passes the window gate
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a2 a5 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
