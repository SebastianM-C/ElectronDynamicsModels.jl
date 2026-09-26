# cloud lane lpwa_hotaisle (MI300X): mid a0; a5em2/a3em1 run only if those cells are in the base recipe
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_lpwa.sh"
_keep=" a5em2 a1em1 a2em1 a3em1 a5em1 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%|*} "* ]] && CELLS+=("$c"); done
