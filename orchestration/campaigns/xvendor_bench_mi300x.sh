# campaigns/xvendor_bench_mi300x.sh — the cross-vendor benchmark on ONE MI300X (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_mi300x
# Per-card kernel options measured 2026-09-09 (ledger): register cache on (1.13×) and C = 8 (the 8–24 plateau of the resolution study; 1.27×). Later assignments win
# inside a cell's override list, so these beat BASE's cache-off / C = 5 (the Hopper choice).
STRONG="$STRONG EDM_COEF_REUSE=1 EDM_SAMPLE_CHUNKS=8"
WEAK="$WEAK EDM_COEF_REUSE=1 EDM_SAMPLE_CHUNKS=8"
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
