# campaigns/mgpu_bench_l7.sh — phase-1 lane on GPU 7: weak N=2000 (D=1) = the single-device
# throughput point, directly comparable with the archived MI300X e2p5e0 cell. This lane also
# declares the weak sweep (axis N) that the other lanes' weak cells join via EDM_SWEEP=mgpu_weak.
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
SWEEP_AXES=N
SWEEP_NAME=mgpu_weak
SWEEP_LABEL="multi-GPU weak scaling = N ladder (e2p5e0 framing)"
CELLS=(
  "weak_N2000_D1|$WEAK EDM_N=2000 CUDA_VISIBLE_DEVICES=7"
)
