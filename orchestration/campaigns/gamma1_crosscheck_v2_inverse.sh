# campaigns/gamma1_crosscheck_v2_inverse.sh — gamma1_crosscheck re-acquired at current main, INVERSE arm.
# Recipe unchanged from gamma1_crosscheck_inverse.sh; new campaign name only. See the direct arm's header.
CAMPAIGN=gamma1_crosscheck_v2
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=Z
BASE=(
  EDM_NX=201 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_GAMMA=1 EDM_TSPAN_TAU=8 EDM_WINDOW=full EDM_SCREEN_HW=0.4
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "inv_mz|EDM_SCREEN_ZSIGN=-1"
  "inv_pz|EDM_SCREEN_ZSIGN=1"
)
