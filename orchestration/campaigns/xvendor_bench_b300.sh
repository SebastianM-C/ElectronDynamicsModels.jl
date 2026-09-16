# campaigns/xvendor_bench_b300.sh — the cross-vendor benchmark on ONE B300 (both cells).
# B300 (Blackwell Ultra): FP64 traded for tensor throughput, 1.2 TFLOP/s per the datasheet — and that is the
# whole story: measured on RunPod 2026-09-16 (driver 595, sm_103a) the strong cell took 5349 s of field time
# (2.4× the RTX 5090's 2251 s; FMA probe 1.203 TF = datasheet, field 0.780 TF = 65 % of it, FP64-bound at
# full rate). Budget ≈ 1.5 h strong + weak ≈ 25 min. NVML GPM's FP64 pipe metric reads ~1 % during the
# saturated field on CC 10.3 — it does not see the sm_103 FP64 path; ncu counters do (diag_validate).
#   RUNPOD_DC="" RUNPOD_GPU_CANDIDATES="NVIDIA B300 SXM6 AC" bash orchestration/backends/runpod.sh run orchestration/campaigns/xvendor_bench_b300.sh
# (no Blackwell in EU-RO-1 as of 2026-09-16; the gpuTypeId is verbatim from the RunPod API)
#   VERDA_TYPES=1B300.30V bash orchestration/backends/verda.sh run orchestration/campaigns/xvendor_bench_b300.sh
# (Verda: on-demand 1×B300 $7.73/h, spot $3.86/h; the instance_type is verbatim from GET /instance-types)
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_b300
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
