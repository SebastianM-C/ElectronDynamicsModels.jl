# campaigns/xvendor_bench_b300.sh — the cross-vendor benchmark on ONE B300 (both cells).
# B300 (Blackwell Ultra): FP64 traded for tensor throughput, 1.2 TFLOP/s per the datasheet; expect 5090-like times, the strong cell ≈ 40 min of field time.
#   RUNPOD_DC="" RUNPOD_GPU_CANDIDATES="NVIDIA B300 SXM6 AC" bash orchestration/backends/runpod.sh run orchestration/campaigns/xvendor_bench_b300.sh
# (no Blackwell in EU-RO-1 as of 2026-09-16; the gpuTypeId is verbatim from the RunPod API)
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_b300
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
