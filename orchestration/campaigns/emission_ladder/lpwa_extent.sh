# cloud lane lpwa_extent: LPWA twin of the a0 = 20 extent probe; launch only after the numeric a20 probe has [window].ok = true and extent_a20_diag is alias-clean to h4 at SPP 16
. "$(dirname "${BASH_SOURCE[0]}")/../emission_ladder_lpwa.sh"
_keep=" a20 "; _all=("${CELLS[@]}"); CELLS=(); for c in "${_all[@]}"; do [[ "$_keep" == *" ${c%|*} "* ]] && CELLS+=("$c"); done
