# lane: the a₀ = 5 cell alone (93.9 GiB) — needs one ≥ 105 GiB card (B200); sharding does not split the cube
. "$(dirname "${BASH_SOURCE[0]}")/../inverse_a0_ladder_g5.sh"
_keep=" a5 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
