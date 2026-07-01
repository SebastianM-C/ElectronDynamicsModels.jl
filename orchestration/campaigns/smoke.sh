# campaigns/smoke.sh — tiny end-to-end validation cell (ANY backend): ODE solve → field accumulation →
# reduction → products on a trivial grid. Run first on a fresh machine/VM; clean exit + sidecars == good.
CAMPAIGN=smoke
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=48 EDM_N=64 EDM_NSAMPLES=512 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_A0=0.1 EDM_NSUBSTEPS=1 EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "smoke|"
)
