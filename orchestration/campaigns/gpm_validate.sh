# campaigns/gpm_validate.sh — two tiny Newton/total cells (~5 min of GPU) to validate the GPM
# counter sampler on a GPM-capable (Hopper+) pod: the manifest must carry [gpu].gpm_* (achieved SM occupancy,
# FP64 / DRAM-bandwidth utilization) beside the compile-time kernel_occupancy, and the run dir a
# gpmtrace_<uuid>.tsv beside the gputrace. Launch (cheapest GPM-capable card, any DC):
#   RUNPOD_DC="" RUNPOD_CLOUD_TYPE=COMMUNITY RUNPOD_GPU_CANDIDATES="NVIDIA H100 PCIe,NVIDIA H100 NVL,NVIDIA H100 80GB HBM3" \
#     bash orchestration/backends/runpod.sh run orchestration/campaigns/gpm_validate.sh
# Not a physics campaign: no cube kept, no sweep declared.
CAMPAIGN=gpm_validate
KEEP_CUBE=0
REDUCE_OVERLAP=0
BASE=(
  EDM_N=8 EDM_NX=400 EDM_NSAMPLES=1000
  EDM_ACCUM_ALG=newton EDM_FIELD_MODE=total
  EDM_SKIP_POSTPROCESS=1
  EDM_GPU_SAMPLE_DT=0.5   # denser ticks: the N=8 cell is JIT-dominated, its kernel-busy window is seconds
)
CELLS=(
  "gpm_newton_total|"
  "gpm_newton_total_N32|EDM_N=32"
)
