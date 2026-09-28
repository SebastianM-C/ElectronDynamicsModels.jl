# lane: the a₀ = 5 cell alone (93.9 GiB) — A5_DEVICES=0,1 shards it over both cards (2 × NVL); empty = one B200
. "$(dirname "${BASH_SOURCE[0]}")/../inverse_a0_ladder_g5.sh"
_keep=" a5 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
