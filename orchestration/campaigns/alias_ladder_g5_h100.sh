# alias_ladder_g5 (h100 half) — the halo wall at γ = 5 (n₀ = 97.99), a₀ = 0.01, ±6 w₀, production
# sunflower, N = 2000 / 4000 / 8000: does the wall collapse onto r_alias across γ in the full
# simulation, as the geometry model says (model walls 1.15, 2.05, 4.95 w₀; r_alias 1.41, 1.99, 2.82)?
# Split over the two GPUs of one node (H100: N = 8000; A100: 2000, 4000), one campaign dir.
# 441² pixels = 0.027 w₀ ≤ δr/2 at n₀ = 98. Feeds the screen-image-formation report.
CAMPAIGN=alias_ladder_g5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=N
BASE=(
  EDM_NX=441 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.01
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_ELECTRON_BATCH=500
  EDM_GAMMA_EPS=4 EDM_SPP=512 EDM_TSPAN_TAU=3.2 EDM_SCREEN_HW=6.0
  EDM_HARMONICS=97.1395,97.9898,193.6813,195.9796
)
CELLS=(
  "N8000|EDM_N=8000"
)
