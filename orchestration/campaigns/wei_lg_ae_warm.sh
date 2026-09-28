# Warm beam: the wei_lg_ae_gauss cells, each electron with its own momentum (EDM_MOMENTA):
# θ ~ N(0, 1.2 mrad) per transverse axis, δ = γ/γ₀ − 1 ~ N(0, 0.2) truncated at 3σ — the same
# fixed-seed draw as wei_gaussian_ae_warm, scaled to the LG shots' divergence.
. "$(dirname "${BASH_SOURCE[0]}")/wei_lg_ae_gauss.sh"
CAMPAIGN=wei_lg_ae_warm
_m=$(<"$(dirname "${BASH_SOURCE[0]}")/momenta/wei_lg_warm_div1.2_s20.txt")
for _i in "${!CELLS[@]}"; do CELLS[_i]+=" EDM_MOMENTA=$_m EDM_BEAM=warm_div1.2mrad_s20"; done
