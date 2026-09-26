# lane numeric_low_b: a0 0.1 … 1 (n_substeps 1)
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a1em1 a2em1 a5em1 a1 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
