# campaigns/sanity_inverse.sh — rk4 numeric twins for sanity_lpwa's lower a0 rungs (γ=1).
# The a0=0.3 twin is rest_departure's `rest` cell — not duplicated here. See sanity_lpwa.sh
# for the full cross-check design and pair list.
CAMPAIGN=sanity_inverse
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=25
  EDM_GAMMA_EPS=0
  EDM_DIRECT_READ=1
)
CELLS=(
  "a1em2_rk4|EDM_A0=1e-2"
  "a1em1_rk4|EDM_A0=1e-1"
)
