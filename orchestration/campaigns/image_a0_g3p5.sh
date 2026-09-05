# image_a0_g3p5 — which features of the near-field ring image are geometry, and which are intensity?
# The γ = 3.5 rung (n₀ = 47, N_F = 14) of rest_departure_bridge_refix / mgpu_bench at a₀ = 0.3 shows
# inner rings the geometry-only diffraction model of the screen-image-formation report does not have,
# while the a₀ = 0.01 gamma_equiv rest run matches that model exactly. Same disk, same screen
# (±3 w₀, 401², the image size), a₀ two decades either side of the existing 0.3 point:
#   a1em2 — a₀ = 0.01  (linear response: the map should BE the geometry model)
#   a1e0  — a₀ = 1     (strong a₀² line shift: intensity reweights the disk)
# Window/bins as the refix e2p5e0 cell (0.99 n₀, n₀ and their doubles; SPP 256 ≥ 2·94).
CAMPAIGN=image_a0_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=a0
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=3.0
  EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576
)
CELLS=(
  "a1em2|EDM_A0=0.01"
  "a1e0|EDM_A0=1"
)
