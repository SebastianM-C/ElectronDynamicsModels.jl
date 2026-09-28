# wei_gaussian_ae with a Gaussian electron bunch (Sebastian: the realistic LWFA transverse profile): 500 electrons from
# one fixed-seed 2D normal draw (MersenneTwister(20260928)), per-axis rms 1.5 or 3 μm (their bunch size is not in the
# paper; 1–3 μm is the usual LWFA range), on axis. Angular energy only (EDM_FIELD=0), head-on; divergence and energy
# spread are applied afterwards to the summed maps.
. "$(dirname "${BASH_SOURCE[0]}")/wei_gaussian_ae.sh"
CAMPAIGN=wei_gaussian_ae_gauss
SWEEP_AXES=gamma,layout,polarization
BASE+=(EDM_FIELD=0)
_pos="$(dirname "${BASH_SOURCE[0]}")/positions"
_base=("${CELLS[@]}"); CELLS=()
for _s in 15 30; do
    _p=$(<"$_pos/wei_g_gauss_s$_s.txt")
    for _c in "${_base[@]}"; do
        _o=$(sed -E 's/ ?EDM_SWEEP=[^ ]*//' <<< "${_c#*|}")
        CELLS+=("${_c%%|*}_s$_s|$_o EDM_POSITIONS=$_p EDM_LAYOUT=gauss500_rms${_s:0:1}.${_s:1}um")
    done
done
