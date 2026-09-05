# bunched_lens_g2 — is bunching a lens? Same γ = 2 disk as rest_departure_bridge_refix e1e0
# (n₀ = 13.93, N_F = 4.1: near-field image regime, r_alias = 9.9 w₀ so the ±4 w₀ screen is alias-free),
# three cells on ONE screen sized to the image (±4 w₀, 401² → 0.02 w₀ pixels, δr = 0.39 w₀):
#   unb  — unbunched control (the ring IMAGE of the LG disk, 2.2 w₀; reproduces e1e0 on a finer grid)
#   l0   — lens only (bunch_nb = 14 ≈ n₀, ℓ = 0): the Fourier-plane pattern, the |2| vortex ring
#   lm2  — lens + helix ℓ = −2: the winding cancels → the axial focal spot
# Prediction (screen-image-formation report, geometry model): r₅₀ ≈ 2.3 w₀ (unb) → ≈ 1.0 w₀ (l0);
# lm2 spot width ≈ δr. Window: narrow, refix bins (0.99 n₀, n₀, and their doubles), SPP 64 ≥ 2·27.9.
CAMPAIGN=bunched_lens_g2
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES="bunch_nb,bunch_l"
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_GAMMA_EPS=1 EDM_SPP=64 EDM_TSPAN_TAU=8 EDM_SCREEN_HW=4
  EDM_HARMONICS=13.7882,13.9282,27.5189,27.8564
)
CELLS=(
  "unb|EDM_BUNCH_NB=0"
  "l0|EDM_BUNCH_NB=14 EDM_BUNCH_L=0"
  "lm2|EDM_BUNCH_NB=14 EDM_BUNCH_L=-2"
)
