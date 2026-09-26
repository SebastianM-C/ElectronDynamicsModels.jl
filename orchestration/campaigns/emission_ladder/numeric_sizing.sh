# lane numeric_sizing: full-size a0=10 sizing cell (window coverage gate)
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a10 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
