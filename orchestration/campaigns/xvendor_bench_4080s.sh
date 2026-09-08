# campaigns/xvendor_bench_4080s.sh — the cross-vendor benchmark on ONE RTX 4080 SUPER: strong cell
# only (the weak cell's 26.8 GiB cube is refused by the 90 % guard on 16 GB). The 12.0 GiB cube plus
# the CUDA context is close to the card's limit — run it on an idle card (no display session).
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_4080s
CELLS=(
  "strong|$STRONG"
)
