# campaigns/xvendor_bench_b300_newton_strong.sh — the ARTICLE-kernel strong cell only (Newton n = 2,
# C = 5) on a B300, for a second-host check of RunPod's xvendor_bench_b300_newton strong number.
# Own campaign name (the RunPod pair is published as xvendor_bench_b300_newton); same sweep id
# xvendor_strong so the dashboard's cross-vendor table sees both points.
#   bash orchestration/backends/verda.sh run orchestration/campaigns/xvendor_bench_b300_newton_strong.sh
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_b300_newton.sh"
CAMPAIGN=xvendor_bench_b300_newton_verda
CELLS=("${CELLS[0]}")
