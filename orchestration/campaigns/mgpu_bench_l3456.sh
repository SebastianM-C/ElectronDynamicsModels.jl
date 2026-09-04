# campaigns/mgpu_bench_l3456.sh — phase-1 lane on GPUs 3-6: strong D=4, then weak N=8000 (D=4).
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CELLS=(
  "strong_D4|$STRONG CUDA_VISIBLE_DEVICES=3,4,5,6"
  "weak_N8000_D4|$WEAK EDM_N=8000 CUDA_VISIBLE_DEVICES=3,4,5,6"
)
