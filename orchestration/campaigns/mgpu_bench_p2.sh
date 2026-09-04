# campaigns/mgpu_bench_p2.sh — phase 2, all 8 GPUs, launched AFTER the four phase-1 lanes are DONE:
# strong D=8, weak N=16000 (D=8), then the 1101² capacity cell (~90 GiB cube; N=2000 sharded 8-way).
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CELLS=(
  "strong_D8|$STRONG CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "weak_N16000_D8|$WEAK EDM_N=16000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
  "cap_1101_D8|$CAPACITY CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
)
