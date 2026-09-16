# campaigns/diag_validate_common.sh — shared definitions of the instrumentation-validation
# campaigns (sourced by diag_validate_amd.sh / diag_validate_nvidia.sh; not a campaign itself).
#
# Purpose: validate the measurement chain on one card in about an hour of GPU time, with every
# number landing in a published manifest — the swept FP64 peak probe with the clock and power it
# ran at ([flops].peak_probe_*), the GEMM reference, the sampler's clock/power/throttle series
# under a sustained field load (the power-bound question on the MI300X), and the per-dispatch
# hardware counters (EDM_PROFILE cells → [gpu].hw_*, the collector's files in [outputs]). The
# physics is the cross-vendor STRONG cell (γ = 5, 401², 1666 samples, Newton n = 2), field-only.
#
#   • power cells: N = 2000 (about a minute on a datacenter card) at sample-chunk counts 1, 2
#     and 8 — enough sustained load for the clock to settle under the power cap; the sampler
#     (0.2 s) records the clock, power and (with GPUDiagnostics ≥ 0.4 on AMD) the throttle state
#   • counter cells: N = 32 (per-slot counts are deterministic; profiled durations are not
#     timings), one cell per counter set, plus the occupancy set at C = 1 and C = 8 so cycles per
#     slot can be compared across the chunk count while the clock moves
# Every cell measures the peak probe (the [flops] section always does), so the probe's
# repeatability across cells is a free by-product. POST_HOOK captures the driver's sysfs/proc
# state after each cell (orchestration/device_snapshot.sh) into registered outputs.
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
KEEP_CUBE=0
REDUCE_OVERLAP=0
POST_HOOK="bash orchestration/device_snapshot.sh \"\$CAMP\""
PROFILE_TIMEOUT=1800
BASE+=(EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SKIP_POSTPROCESS=1 EDM_GPU_SAMPLE_DT=0.2)
# the strong cell's physics with the N of each cell class; sweep ids name the two classes
POWER="$STRONG EDM_N=2000 EDM_SWEEP=diag_power"
COUNTERS="$STRONG EDM_N=32 EDM_SWEEP=diag_counters"
CELLS=(
  "power_c1|$POWER EDM_SAMPLE_CHUNKS=1"
  "power_c2|$POWER EDM_SAMPLE_CHUNKS=2"
  "power_c8|$POWER EDM_SAMPLE_CHUNKS=8"
  "fp64|$COUNTERS EDM_PROFILE=fp64"
  "memory|$COUNTERS EDM_PROFILE=memory"
  "issue|$COUNTERS EDM_PROFILE=issue"
  "l2|$COUNTERS EDM_PROFILE=l2"
  "occupancy_c1|$COUNTERS EDM_PROFILE=occupancy EDM_SAMPLE_CHUNKS=1"
  "occupancy_c8|$COUNTERS EDM_PROFILE=occupancy EDM_SAMPLE_CHUNKS=8"
)
