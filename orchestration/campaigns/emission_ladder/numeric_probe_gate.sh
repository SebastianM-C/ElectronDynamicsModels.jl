# lane probe_gate (H100 NVL #0): full-size a0 = 10 probe — the window-coverage gate for a0 ≥ 5 ([window].ok must be true)
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a10 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
