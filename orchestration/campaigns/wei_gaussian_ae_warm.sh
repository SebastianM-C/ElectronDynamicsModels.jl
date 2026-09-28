# Warm beam: the wei_gaussian_ae_gauss cells, each electron with its own momentum (EDM_MOMENTA):
# θ ~ N(0, 2.0 mrad) per transverse axis, δ = γ/γ₀ − 1 ~ N(0, 0.2) truncated at 3σ — one fixed-seed
# draw (momenta/draw_momenta.jl, MersenneTwister(20260929)), shared by every cell.
. "$(dirname "${BASH_SOURCE[0]}")/wei_gaussian_ae_gauss.sh"
CAMPAIGN=wei_gaussian_ae_warm
_m=$(<"$(dirname "${BASH_SOURCE[0]}")/momenta/wei_g_warm_div2.0_s20.txt")
for _i in "${!CELLS[@]}"; do CELLS[_i]+=" EDM_MOMENTA=$_m EDM_BEAM=warm_div2.0mrad_s20"; done
