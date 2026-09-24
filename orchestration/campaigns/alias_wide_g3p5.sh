# alias_wide_g3p5 — the halo wall at large N with the exact (a₀ = 0.01) geometry reference.
# γ = 3.5, ±7 w₀, production sunflower, N = 4000 and 8000 (N = 2000 on the same screen is the
# grating_g3p5 sunflower cell). Ghost-edge law: wall ≈ 2 r_alias − 1.9 w₀ = 6.4 w₀ at N = 4000 and
# beyond the screen at 8000; the earlier large-N points were a₀ = 0.3 (mgpu_bench), where the plain
# model is approximate. Feeds the screen-image-formation report.
CAMPAIGN=alias_wide_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=N
BASE=(
  EDM_NX=351 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.01
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_ELECTRON_BATCH=250
  EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=7.0
  EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576
)
CELLS=(
  "N4000|EDM_N=4000"
  "N8000|EDM_N=8000"
)
