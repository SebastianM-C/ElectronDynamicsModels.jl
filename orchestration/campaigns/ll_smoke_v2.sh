# campaigns/ll_smoke_v2.sh — ll_smoke re-run at current main (inverse-Thomson session probes): the Newton
# kernel end-to-end, classical and LL. Recipe unchanged from ll_smoke.sh; new campaign name, cube kept.
CAMPAIGN=ll_smoke_v2
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=system
BASE=(
  EDM_A0=1 EDM_GAMMA=10 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=100 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_TSPAN_TAU=1.6 EDM_SPP=2048 EDM_SCREEN_HW=0.4 EDM_NX=65
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3 EDM_HARMONICS=199,299,398,597
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "smoke_newton|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2"
  "smoke_ll|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SYSTEM=ll"
)
