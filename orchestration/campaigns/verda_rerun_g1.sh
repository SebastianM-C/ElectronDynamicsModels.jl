# lane (Verda C″, rerun): the a₀ 10 pair of ll_probe_s_v2 on card 1 after the a₀ = 5 cell — first attempts died at the first
# CUDA call (CC GPUs "not ready" until `nvidia-smi conf-compute -srs 1`, 2026-09-28 03:38Z). Batching as verda_llps_4_g0.sh.
LLS_SCREEN=nx281
. "$(dirname "${BASH_SOURCE[0]}")/ll_probe_s_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=500); _lane_gpu 1
_lane_cells "a10_cl a10_ll"
_lane_after verda_a5_g1 1
