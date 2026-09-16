# campaigns/diag_validate_common.sh — shared definitions of the instrumentation-validation
# campaigns (sourced by diag_validate_amd.sh / diag_validate_nvidia.sh; not a campaign itself).
#
# Purpose: validate the measurement chain on one card and collect the per-card limiter figure
# of the thesis in about 40 minutes of GPU time, with every number landing in a published manifest — the swept FP64 peak probe with the clock and power it
# ran at ([flops].peak_probe_*), the GEMM reference, the sampler's clock/power/throttle series
# under a sustained field load (the power-bound question on the MI300X), and the per-dispatch
# hardware counters (EDM_PROFILE cells → [gpu].hw_*, the collector's files in [outputs]). The
# physics is the cross-vendor STRONG cell (γ = 5, 401², 1666 samples, Newton n = 2), field-only.
#
#   • the chunk sweep, C ∈ {1, 2, 4, 8, 16}, is the x axis of the per-card "what limits the
#     kernel" figure: at each C one N = 2000 cell under the sampler (launch time, clock, power
#     against the cap, and with GPUDiagnostics ≥ 0.4 on AMD the throttle state — about a minute
#     of sustained load on a datacenter card, enough for the clock to settle) and one N = 32
#     cell under the `occupancy` counter set (cycles per slot, achieved occupancy; per-slot
#     counts are deterministic and profiled durations are not timings). Sweep ids
#     diag_chunks_power / diag_chunks_occupancy group them on the dashboard.
#   • single-C counter cells at N = 32: `fp64`, `issue`, `memory`, `l2` (sweep id diag_counters)
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
KEEP_CUBE=0
REDUCE_OVERLAP=0
POST_HOOK="bash orchestration/device_snapshot.sh \"\$CAMP\""
PROFILE_TIMEOUT=1800
BASE+=(EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SKIP_POSTPROCESS=1 EDM_GPU_SAMPLE_DT=0.2)
# the strong cell's physics with the N of each cell class; sweep ids name the classes
POWER="$STRONG EDM_N=2000 EDM_SWEEP=diag_chunks_power"
OCC="$STRONG EDM_N=32 EDM_SWEEP=diag_chunks_occupancy EDM_PROFILE=occupancy"
COUNTERS="$STRONG EDM_N=32 EDM_SWEEP=diag_counters"
CELLS=()
for C in 1 2 4 8 16; do
  CELLS+=("power_c$C|$POWER EDM_SAMPLE_CHUNKS=$C")
done
for C in 1 2 4 8 16; do
  CELLS+=("occupancy_c$C|$OCC EDM_SAMPLE_CHUNKS=$C")
done
CELLS+=(
  "fp64|$COUNTERS EDM_PROFILE=fp64"
  "issue|$COUNTERS EDM_PROFILE=issue"
  "memory|$COUNTERS EDM_PROFILE=memory"
  "l2|$COUNTERS EDM_PROFILE=l2"
)
