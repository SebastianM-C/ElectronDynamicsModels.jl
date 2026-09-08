# campaigns/xvendor_bench_h200.sh — the cross-vendor benchmark on ONE H200 (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_h200
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
