# campaigns/mgpu_bench_l0.sh — phase-1 lane on GPU 0: strong-scaling D=1 (the long pole, ~2.2 h on H200).
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CELLS=(
  "strong_D1|$STRONG CUDA_VISIBLE_DEVICES=0"
)
