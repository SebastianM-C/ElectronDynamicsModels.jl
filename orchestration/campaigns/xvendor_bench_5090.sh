# campaigns/xvendor_bench_5090.sh — the cross-vendor benchmark on ONE 5090 (both cells).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_5090
# Per-card kernel options measured 2026-09-09 (ledger): register cache on (1.03× at one block per SM) and C = 4 (7.4 → 8 rounds of 170 block slots). Later assignments win
# inside a cell's override list, so these beat BASE's cache-off / C = 5 (the Hopper choice).
STRONG="$STRONG EDM_COEF_REUSE=1 EDM_SAMPLE_CHUNKS=4"
WEAK="$WEAK EDM_COEF_REUSE=1 EDM_SAMPLE_CHUNKS=4"
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
