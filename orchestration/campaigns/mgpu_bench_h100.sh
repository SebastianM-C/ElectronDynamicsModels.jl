# campaigns/mgpu_bench_h100.sh — the H100 throughput point: the weak N=2000 cell on a 1×H100 pod
# (its own STATE; launch with RUNPOD_GPU_CANDIDATES="NVIDIA H100 80GB HBM3" RUNPOD_DC="").
# Same physics as mgpu_bench/weak_N2000_D1 ⇒ no cube kept.
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CAMPAIGN=mgpu_bench_h100
KEEP_CUBE=0
CELLS=(
  "weak_N2000_D1_h100|$WEAK EDM_N=2000 EDM_SWEEP=mgpu_h100"
)
