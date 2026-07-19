# campaigns/bunched_resolved_smoke.sh — CUDA-path shakeout for bunched_resolved (minutes):
# tiny lm2 cell on the resolved-screen geometry; validates the RunPod CUDA image + EDM cuda
# backend + products before the 8-cell run pays for anything.
CAMPAIGN=bunched_resolved_smoke
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_A0=1 EDM_GAMMA=10 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=100 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_TSPAN_TAU=1.6 EDM_SPP=2048 EDM_SCREEN_HW=0.4 EDM_NX=65
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3 EDM_HARMONICS=199,299,398,597
)
CELLS=(
  "smoke_lm2|EDM_BUNCH_NB=398 EDM_BUNCH_L=-2"
)
