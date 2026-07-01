# campaigns/smoke.sh — tiny end-to-end validation cell, runnable on ANY backend.
# PURE DATA. Proves the whole toolchain (ODE solve → field accumulation → reduction →
# products) on a trivially small grid before committing to a costly campaign. Use it as the
# first thing on a fresh machine / a freshly-provisioned cloud VM:
#   bash   orchestration/backends/local.sh    orchestration/campaigns/smoke.sh
#   sbatch orchestration/backends/slurm.sbatch orchestration/campaigns/smoke.sh
#   bash   orchestration/backends/hotaisle.sh run orchestration/campaigns/smoke.sh
# A clean exit + a run_<uuid>.toml with its PNG/derived sidecars beside it == the box is good.
CAMPAIGN=smoke
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=48 EDM_N=64 EDM_NSAMPLES=512 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_A0=0.1 EDM_NSUBSTEPS=1 EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "smoke|"
)
