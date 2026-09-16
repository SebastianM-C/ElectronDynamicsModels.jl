# campaigns/xvendor_bench_b200.sh — the cross-vendor benchmark on ONE B200 (both cells).
# B200: full-rate FP64 Blackwell (datasheet ≈ 40 TFLOP/s), the 'does the new SM help an issue-bound kernel' point.
#   RUNPOD_DC="" RUNPOD_GPU_CANDIDATES="NVIDIA B200" bash orchestration/backends/runpod.sh run orchestration/campaigns/xvendor_bench_b200.sh
# (no Blackwell in EU-RO-1 as of 2026-09-16; the gpuTypeId is verbatim from the RunPod API)
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_b200
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
