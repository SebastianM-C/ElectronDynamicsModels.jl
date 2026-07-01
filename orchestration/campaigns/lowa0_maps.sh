# campaigns/lowa0_maps.sh — low-a0 radiated-field harmonic maps.
# PURE DATA: no infra, no backend, no run-tags (the core mints UUIDs). Runs on any backend:
#   bash orchestration/backends/local.sh   orchestration/campaigns/lowa0_maps.sh
#   bash orchestration/backends/hotaisle.sh orchestration/campaigns/lowa0_maps.sh
#   sbatch orchestration/backends/slurm.sbatch orchestration/campaigns/lowa0_maps.sh
#
# Uses uniform EDM_INTERP_SAVEAT=16 — the floor fix: adaptive Vern9 output injects a spurious 2ω
# in the radiation field; uniform output recovers the physical ∝a0 2ω (see the floor investigation).
CAMPAIGN=lowa0_maps
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=400 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=10000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2 (published convention)
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em6|EDM_A0=1e-6"
  "a1em7|EDM_A0=1e-7"
)
