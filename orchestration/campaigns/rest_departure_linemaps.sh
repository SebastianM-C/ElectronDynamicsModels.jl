# campaigns/rest_departure_linemaps.sh — line-anchored harmonic maps for the ε ladder.
# The rest_departure runs mapped integer laser harmonics (the :full default 1..4 ω₁), but the
# boosted backscatter line sits BETWEEN them — up to 32% off the nearest bin (wing maps, not
# line maps). Reruns of the 8 off-line rungs with fractional EDM_HARMONICS at BOTH anchors:
#   n_th  = (1+β)/(1−β)          the "4γ²" theory line
#   n_ps  = the measured powspec peak from the original run (2026-07-30 gates; both carry the
#           common −0.8% window systematic, ~3 rfft bins apart at Ns/SPP = 375)
# plus h1 for continuity with the original chips. NEW campaign name: same-ε duplicates inside
# rest_departure would break the one-axis sweep (harmonics not γ-locked across the pair).
# KEEP_CUBE=1 + auto-drain (policy from tonight: cubes you might re-mine go to R2 — future
# re-extraction at any frequency becomes CPU-only instead of a GPU rerun).
CAMPAIGN=rest_departure_linemaps
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=25
  EDM_DIRECT_READ=1
)
CELLS=(
  "e1em3_lm|EDM_GAMMA_EPS=1e-3 EDM_HARMONICS=1,1.0936,1.0853"
  "e2em3_lm|EDM_GAMMA_EPS=2e-3 EDM_HARMONICS=1,1.1348,1.1253"
  "e5em3_lm|EDM_GAMMA_EPS=5e-3 EDM_HARMONICS=1,1.2213,1.2107"
  "e1em2_lm|EDM_GAMMA_EPS=1e-2 EDM_HARMONICS=1,1.3266,1.3173"
  "e2em2_lm|EDM_GAMMA_EPS=2e-2 EDM_HARMONICS=1,1.4908,1.48"
  "e5em2_lm|EDM_GAMMA_EPS=5e-2 EDM_HARMONICS=1,1.8773,1.8613"
  "e1em1_lm|EDM_GAMMA_EPS=1e-1 EDM_HARMONICS=1,2.4282,2.408"
  "e2em1_lm|EDM_GAMMA_EPS=2e-1 EDM_HARMONICS=1,3.4724,3.4453"
)
