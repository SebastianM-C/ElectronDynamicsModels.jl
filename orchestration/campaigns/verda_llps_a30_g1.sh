# lane (Verda 2×B200, route C″): the a₀ 30 pair of ll_probe_s_v2 at 281² on card 1, before the a₀ = 5 cell
# (verda_a5_g1.sh). Batching as in verda_llps_4_g0.sh.
LLS_SCREEN=nx281
. "$(dirname "${BASH_SOURCE[0]}")/ll_probe_s_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=500); _lane_gpu 1
_lane_cells "a30_cl a30_ll"
