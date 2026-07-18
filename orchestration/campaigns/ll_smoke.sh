# campaigns/ll_smoke.sh — MI300X shakeout of the hardened Newton kernel (PR #44:
# light-front residual + bracketed step) + depot-cache warm for the LL campaign.
# Tiny cell, minutes; the cloud-GPU gate the PR's validation section calls for.
CAMPAIGN=ll_smoke
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_A0=1 EDM_GAMMA=10 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=100 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_TSPAN_TAU=1.6 EDM_SPP=2048 EDM_SCREEN_HW=0.4 EDM_NX=65
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3 EDM_HARMONICS=199,299,398,597
)
CELLS=(
  "smoke_newton|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2"
)
