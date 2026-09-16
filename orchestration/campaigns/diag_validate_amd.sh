# campaigns/diag_validate_amd.sh — instrumentation validation on one AMD Instinct card (MI300X).
#   bash orchestration/backends/hotaisle.sh run orchestration/campaigns/diag_validate_amd.sh
#   RUNPOD_GPU_CANDIDATES="AMD Instinct MI300X OAM" bash orchestration/backends/runpod.sh run orchestration/campaigns/diag_validate_amd.sh
# Cloud MI300X instances are SR-IOV virtual functions: whether the VF exposes `gpu_metrics`, a
# writable performance level or the amd-smi violation counters is exactly what the snapshot
# records. Counters collect in the default performance state on Instinct parts; the fp64 set's
# instruction counts are clock-independent anyway.
. "$(dirname "${BASH_SOURCE[0]}")/diag_validate_common.sh"
CAMPAIGN=diag_validate_amd
