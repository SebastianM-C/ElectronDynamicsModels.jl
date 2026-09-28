# lane (Verda 2×B200, route C): ll_probe_s_v2 LL cells at 281² on card 1 — pair lane verda_llps_g0.sh; the a₀ = 5
# cell (verda_a5_g1.sh) follows on this card. Batching as in verda_llps_g0.sh.
LLS_SCREEN=nx281
. "$(dirname "${BASH_SOURCE[0]}")/ll_probe_s_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=500); _lane_gpu 1
_lane_cells "a10_ll a100_ll a30_ll"
