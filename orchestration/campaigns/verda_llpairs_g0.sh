# lane (Verda 2×B200, route C″ tail): the six ll_pairs_v2 cells on card 0 once verda_llps_4_g0.sh has finished its four
# field phases — they overlap card 1's a₀ = 5 solve/reduce, so the VM's last drain is a small cube. Splines ≤ 28 GiB
# (a₀ 8 at 32 knots), cubes 13–31 GiB: no batching needed. Can join a running VM (verda.sh run on the kept state).
. "$(dirname "${BASH_SOURCE[0]}")/ll_pairs_v2.sh"
. "$(dirname "${BASH_SOURCE[0]}")/verda_inverse_lanes.sh"
_lane_gpu 0
_lane_after verda_llps_4_g0 4
