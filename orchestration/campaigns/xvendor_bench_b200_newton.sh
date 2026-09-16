# campaigns/xvendor_bench_b200_newton.sh — the cross-vendor cells on ONE B200 with the ARTICLE's
# kernel: EDM_ACCUM_ALG=newton (n = 2) at EDM_SAMPLE_CHUNKS=5, i.e. the configuration the September
# H100/H200 cells (xvendor_rerun) actually ran. The plain recipe inherits the script defaults
# (rk4, C = 1): the first Blackwell pass on 2026-09-16 ran that kernel (973 FLOP/slot vs Newton's
# 737) and is kept as data under xvendor_bench_b200 / _c5.
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_b200_newton
CELLS=(
  "strong|$STRONG EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=5"
  "weak|$WEAK EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=5"
)
