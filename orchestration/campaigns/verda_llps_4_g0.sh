# lane (Verda 2×B200, route C″): the a₀ 10 and a₀ 100 pairs of ll_probe_s_v2 at 281² on card 0; the a₀ 30 pair and
# then the a₀ = 5 cell share card 1 (verda_llps_a30_g1.sh, verda_a5_g1.sh) ⇒ ≈ equal lanes (measured MI300X cell
# times: a₀ 10 at N 4000 ≈ a₀ 100 at N 2000 ≈ 1.1× a₀ 30). Batching as in verda_llps_g0.sh.
LLS_SCREEN=nx281
. "$(dirname "${BASH_SOURCE[0]}")/ll_probe_s_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=500); _lane_gpu 0
_lane_cells "a10_cl a10_ll a100_cl a100_ll"
