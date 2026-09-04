# campaigns/mgpu_bench_l12.sh — phase-1 lane on GPUs 1-2: strong D=2, then weak N=4000 (D=2).
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CELLS=(
  "strong_D2|$STRONG CUDA_VISIBLE_DEVICES=1,2"
  "weak_N4000_D2|$WEAK EDM_N=4000 CUDA_VISIBLE_DEVICES=1,2"
)
