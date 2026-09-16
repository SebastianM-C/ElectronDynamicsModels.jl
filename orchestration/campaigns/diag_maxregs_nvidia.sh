# campaigns/diag_maxregs_nvidia.sh — register-cap A/B for the Newton field kernel on one NVIDIA
# card (GPUDiagnostics issue #4). Hopper runs the kernel at 128 registers/thread → theoretical
# occupancy 0.25 with the FP64 pipe idle ~80 % of the time; does capping registers raise achieved
# occupancy and per-launch speed, or do spills eat the gain? (On an FP64-saturated card removing
# work gained < 10 % — the 5090 ledger; Hopper is where the headroom is.)
#   VERDA_TYPES=1H100.80S.32V bash orchestration/backends/verda.sh run orchestration/campaigns/diag_maxregs_nvidia.sh
#   (Verda VMs allow counter collection; RunPod containers refuse it — the occupancy/fp64 cells need it)
# Per cap R ∈ {ctrl = compiler default, 96, 80, 64}, the strong cell's physics (γ = 5, 401², 1666
# samples, Newton n = 2) as three cell classes, EDM_MAXREGS=R riding every one:
#   • power:     N = 2000, C = 4 under the sampler — launch median, GPM occupancy/FP64 util, clock, power
#   • occupancy: N = 32, `occupancy` counters — achieved occupancy (sm__warps_active), cycles/slot
#   • fp64:      N = 32, `fp64` counters — dfma/dadd/dmul per slot, sm__pipe_fp64_cycles_active
# Per setting the manifest carries kernel_registers / kernel_isa_spill_* / kernel_maxregs ([gpu]),
# the launch median ([timing]/[gpu].sampler_*), GPM + hw_* counters. 12 cells ≈ 40 min on an H100.
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CAMPAIGN="${DIAG_CAMPAIGN:-diag_maxregs_nvidia}"
KEEP_CUBE=0
REDUCE_OVERLAP=0
POST_HOOK="bash orchestration/device_snapshot.sh \"\$CAMP\""
PROFILE_TIMEOUT=1800
BASE+=(EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SKIP_POSTPROCESS=1 EDM_GPU_SAMPLE_DT=0.2 EDM_SAMPLE_CHUNKS=4)
POWER="$STRONG EDM_N=2000 EDM_SWEEP=diag_maxregs_power"
OCC="$STRONG EDM_N=32 EDM_SWEEP=diag_maxregs_occupancy EDM_PROFILE=occupancy"
FP64="$STRONG EDM_N=32 EDM_SWEEP=diag_maxregs_fp64 EDM_PROFILE=fp64"
CELLS=(
  "power_ctrl|$POWER"
  "occupancy_ctrl|$OCC"
  "fp64_ctrl|$FP64"
)
for R in 96 80 64; do
  CELLS+=(
    "power_r$R|$POWER EDM_MAXREGS=$R"
    "occupancy_r$R|$OCC EDM_MAXREGS=$R"
    "fp64_r$R|$FP64 EDM_MAXREGS=$R"
  )
done
