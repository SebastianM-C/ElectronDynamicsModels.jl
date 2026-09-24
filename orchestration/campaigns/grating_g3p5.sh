# grating_g3p5 — the electron layout as a grating, in the full simulation. γ = 3.5 (n₀ = 46.98),
# a₀ = 0.01, ±7 w₀: a square lattice at the density of the 2000-point sunflower (EDM_POSITIONS=square,
# 1992 sites, spacing d = 0.129 w₀) against the production sunflower. The model predicts sharp ghost
# copies of the image centred at λₙZ/d = 2 r_alias = 5.88 w₀ along the lattice axes on the square
# lattice, and a speckle ring starting ≈ 4.0 w₀ on the sunflower. Feeds the screen-image-formation report.
CAMPAIGN=grating_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=layout
BASE=(
  EDM_NX=351 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
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
  "square|EDM_POSITIONS=square EDM_LAYOUT=square"
  "sunflower|"
)
