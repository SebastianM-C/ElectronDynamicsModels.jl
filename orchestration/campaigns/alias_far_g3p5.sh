# alias_far_g3p5 — the halo wall where the model puts it near 2 r_alias: γ = 3.5, a₀ = 0.01,
# N = 4000, ±10 w₀ (361² = 0.056 w₀ ≤ δr/2). Model wall 8.15 w₀ (r_alias 4.15, ghost distance 8.31).
# Needs an 80 GB GPU for the wide window (run on an H100). Feeds the screen-image-formation report.
CAMPAIGN=alias_far_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=N
BASE=(
  EDM_NX=361 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.01
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_ELECTRON_BATCH=500
  EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=10.0
  EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576
)
CELLS=(
  "N4000|EDM_N=4000"
)
