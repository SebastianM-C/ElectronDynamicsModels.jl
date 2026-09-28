# lane (Verda 2×B200): the a₀ = 5 cell of inverse_a0_ladder_g5 (93.9 GiB) on card 1, after the a₀ 30 pair
# (verda_llps_a30_g1.sh) — list this lane after it; without that lane it starts at once. EDM_ELECTRON_BATCH
# 4000 → 2000: 7.4 MB/electron at 24k knots, two batches resident while the next is solved ⇒ 59 GB → 30 GB during
# the field phase (bit-identical on one device).
. "$(dirname "${BASH_SOURCE[0]}")/inverse_a0_ladder_g5.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
BASE+=(EDM_ELECTRON_BATCH=2000); _lane_gpu 1
_lane_cells "a5"
_lane_after verda_llps_a30_g1 2
