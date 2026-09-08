# campaigns/xvendor_bench_mi300x.sh — the cross-vendor benchmark on ONE MI300X (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_mi300x
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
