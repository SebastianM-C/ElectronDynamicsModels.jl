# wei_lg_ae with a Gaussian electron bunch centred on the ring (ρ = √3.5 w₀): 500 electrons from the same fixed-seed
# draw, per-axis rms 1.5 or 3 μm, at 160 / 200 / 240 MeV (their 200 MeV ± 20 %, for the energy-spread average).
# Angular energy only (EDM_FIELD=0), head-on. SPP 2e6 keeps n₀(240 MeV) = 886113 inside Nyquist (no cube is made).
. "$(dirname "${BASH_SOURCE[0]}")/wei_lg_ae.sh"
CAMPAIGN=wei_lg_ae_gauss
SWEEP_AXES=gamma,layout,polarization
BASE+=(EDM_FIELD=0 EDM_SPP=2000000)
_pos="$(dirname "${BASH_SOURCE[0]}")/positions"
CELLS=()
for _s in 15 30; do
    _p=$(<"$_pos/wei_lg_gauss_s$_s.txt")
    for _e in "160 314.1121893694694 0.05093721460513033 381285,394664" \
              "200 392.39023671183674 0.04077573421315798 595000,615878" \
              "240 470.6682840542041 0.03399421746071459 856074,886113"; do
        read -r _E _g _t _h <<< "$_e"
        for _pol in linear:lp circular_plus:cp; do
            CELLS+=("${_pol#*:}_lg7_e${_E}_s$_s|EDM_LG_M=7 EDM_POL=${_pol%%:*} EDM_GAMMA=$_g EDM_TSPAN_TAU=$_t EDM_HARMONICS=$_h EDM_POSITIONS=$_p EDM_LAYOUT=ring_gauss500_rms${_s:0:1}.${_s:1}um")
        done
    done
done
