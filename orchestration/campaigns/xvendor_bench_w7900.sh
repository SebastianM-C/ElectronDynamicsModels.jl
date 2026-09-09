# campaigns/xvendor_bench_w7900.sh — the cross-vendor benchmark on ONE W7900 (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_w7900
# Per-card kernel options measured 2026-09-09 (ledger): register cache on (1.11×) and C = 4 (1.05×; 16 is already slower). Later assignments win
# inside a cell's override list, so these beat BASE's cache-off / C = 5 (the Hopper choice).
STRONG="$STRONG EDM_COEF_REUSE=1 EDM_SAMPLE_CHUNKS=4"
WEAK="$WEAK EDM_COEF_REUSE=1 EDM_SAMPLE_CHUNKS=4"
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
