# campaigns/mgpu_bench_smoke.sh — 2-cell shakeout on the 8-GPU pod BEFORE phase 1 (~3 min):
# the weak framing shrunk to 201²/N=200 on one device and on two. Both manifests must agree on
# the maps to roundoff (sharding is exact) and the D=2 field time should be ~half of D=1.
# CAMPAIGN=smoke is exempt from the cube drainer + teardown gate by name.
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
CAMPAIGN=smoke
KEEP_CUBE=0
CELLS=(
  "smoke_D1|$WEAK EDM_NX=201 EDM_N=200 CUDA_VISIBLE_DEVICES=0"
  "smoke_D2|$WEAK EDM_NX=201 EDM_N=200 CUDA_VISIBLE_DEVICES=0,1"
)
