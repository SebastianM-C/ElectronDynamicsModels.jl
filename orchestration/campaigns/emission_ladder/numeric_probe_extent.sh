# lane probe_extent (H100 NVL #0): full-size a0 = 20 extent probe, after the a0 = 10 gate; kept in the ladder only if its own [window].ok is true and extent_a20_diag is alias-clean to h4 at SPP 16
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a20 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
