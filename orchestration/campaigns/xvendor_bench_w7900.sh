# campaigns/xvendor_bench_w7900.sh — the cross-vendor benchmark on ONE W7900 (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_w7900
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
