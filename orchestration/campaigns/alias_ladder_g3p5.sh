# alias_ladder_g3p5 — the √N arm of the r_alias law at γ = 3.5 (n₀ = 46.98), downward from the
# image_a0_g3p5 N = 2000 cell (identical config otherwise). Prediction r_alias = 3.09·√N/n₀ w₀:
# N = 250 → 1.04, 500 → 1.47, 1000 → 2.08, 2000 → 2.94 w₀ (screen half-width 3 w₀), so the alias
# onset walks across the screen. Scored by scripts/alias_metrics.jl; feeds the
# screen-image-formation report. Sized for a 16 GB consumer GPU: electron batches bound host RAM,
# no cube kept (the hmaps hold the complex line maps).
CAMPAIGN=alias_ladder_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=N
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.01
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_ELECTRON_BATCH=250
  EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=3.0
  EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576
)
CELLS=(
  "N250|EDM_N=250"
  "N500|EDM_N=500"
  "N1000|EDM_N=1000"
)
