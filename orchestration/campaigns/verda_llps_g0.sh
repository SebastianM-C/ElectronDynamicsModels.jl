# lane (Verda 2×B200, route C): ll_probe_s_v2 classical cells at 281² on card 0 — pair lane verda_llps_g1.sh.
# EDM_ELECTRON_BATCH=500: ~98k knots/electron (TSPAN 0.16 τ·γ 100, 64 knots/period) ≈ 30 MB each, so an
# unbatched N = 4000 cell holds ~110 GiB of splines under its cube download; batched (bit-identical on one
# device) the field phase holds ≤ 30 GB and the download lands on an empty host.
LLS_SCREEN=nx281
. "$(dirname "${BASH_SOURCE[0]}")/ll_probe_s_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=500); _lane_gpu 0
_lane_cells "a10_cl a100_cl a30_cl"
