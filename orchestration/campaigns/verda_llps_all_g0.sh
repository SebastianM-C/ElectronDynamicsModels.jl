# lane (Verda 2×B200, route C′): all six ll_probe_s_v2 cells at 281² on card 0, while verda_a5_g1.sh runs on card 1
# from the start (≈ equal lane lengths; one card per lane, no hand-over). Batching as in verda_llps_g0.sh.
LLS_SCREEN=nx281
. "$(dirname "${BASH_SOURCE[0]}")/ll_probe_s_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=500); _lane_gpu 0
_lane_cells "a10_cl a10_ll a100_cl a100_ll a30_cl a30_ll"
