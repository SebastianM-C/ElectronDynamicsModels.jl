# lane numeric_low_a: a0 ≤ 2 (below the coverage gate), first half
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a1em5 a1em4 a1em3 a1em2 a2 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
