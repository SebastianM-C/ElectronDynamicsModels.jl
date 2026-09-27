# lane probe_typical (H100 NVL #1): full-size a0 = 0.1 probe — stage timings, host-RAM and VRAM peaks for a typical cell
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_numeric.sh"
_keep=" a1em1 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%%|*} "* ]] && CELLS+=("$c"); done
