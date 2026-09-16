# campaigns/diag_validate_nvidia.sh — instrumentation validation on one NVIDIA datacenter card
# (H100 SXM / H200 SXM on a rented pod, or any Hopper box for development).
#   RUNPOD_GPU_CANDIDATES="NVIDIA H100 80GB HBM3" bash orchestration/backends/runpod.sh run orchestration/campaigns/diag_validate_nvidia.sh
# The GPM sampler needs Hopper or newer; the counter cells need `ncu` reachable (set
# GPUDIAGNOSTICS_COUNTER_TOOL when it is not on PATH) and a driver that permits non-admin
# counter collection (RmProfilingAdminOnly = 0 — the snapshot records it; when refused the
# counter cells fail with ERR_NVGPUCTRPERM in their log and the power cells still stand).
. "$(dirname "${BASH_SOURCE[0]}")/diag_validate_common.sh"
# DIAG_CAMPAIGN names the campaign per card (diag_validate_h100sxm, diag_validate_b300, …) so
# output dirs and published campaign names stay separate; the recipe is the same.
CAMPAIGN="${DIAG_CAMPAIGN:-diag_validate_nvidia}"
