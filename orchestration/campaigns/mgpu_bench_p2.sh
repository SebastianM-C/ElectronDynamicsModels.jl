# campaigns/mgpu_bench_p2.sh — phase 2, all 8 GPUs, launched AFTER the four phase-1 lanes are DONE.
# The 1101² capacity cell goes FIRST: its ~90 GiB cube is consumer-bound (reduce + sha + upload
# ≈ 32 min vs ≈ 15 min of field), so running it first lets the two 8-device cells overlap that
# tail instead of the pod idling on the teardown gate. Then strong D=8 and weak N=16000 (D=8).
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CELLS=(
  "cap_1101_D8|$CAPACITY CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "strong_D8|$STRONG CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "weak_N16000_D8|$WEAK EDM_N=16000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
)
