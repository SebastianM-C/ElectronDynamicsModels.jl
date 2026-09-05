# campaigns/mgpu_bench_calib.sh — throughput calibration at the PRODUCTION screen (601²) before
# phase 1, because the 201² smoke underfills the H200 (thread-fill occupancy 0.15: 40k pixel
# threads on 132 SMs × 2048) and reports a per-device throughput that does not transfer to the
# 601² cells (361k threads, saturated). Two short cells on the smoke campaign (no cube kept):
#   • calib_D1: N=100 on one device  → per-device throughput at 601² (the phase-1 gate number)
#   • calib_D8: N=800 on all eight   → same work per device: the first 8-way sharded CUDA run,
#     its field time vs calib_D1 = the weak-scaling efficiency at this size, and its maps must
#     equal an N=800 single-device run only in the sense of being the exact 800-electron sum
#     (compare_hmaps against calib_D1 is NOT meaningful — different N).
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CAMPAIGN=smoke
KEEP_CUBE=0
CELLS=(
  "calib_D1|$WEAK EDM_N=100 CUDA_VISIBLE_DEVICES=0"
  "calib_D8|$WEAK EDM_N=800 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
)
