# campaigns/xvendor_bench_5090.sh — the cross-vendor benchmark on ONE 5090 (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_5090
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
