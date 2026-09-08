# campaigns/xvendor_bench_h100.sh — the cross-vendor benchmark on ONE H100 (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_h100
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
