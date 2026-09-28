# lane: the single-card a₀ cells (0.01 … 2) of inverse_a0_ladder_g5 — one NVL / PCIe
. "$(dirname "${BASH_SOURCE[0]}")/../inverse_a0_ladder_g5.sh"
_keep=" a1em2 a1em1 a3em1 a1 a2 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
