# campaigns/sanity_newton_lm.sh — line-anchored rerun of the Newton sanity arm.
# The original sanity_newton cells mapped the :full integer defaults (h1 = a wing map at
# ε = 1e-3, 8.5% off-line) and ran keep=0 — no cubes to re-extract. Rerun both rungs with
# fractional anchors (theory line + measured powspec peak from the 2026-07-30 gates), the
# same anchors as rest_departure_linemaps' rk4 twins, so the kernel A/B comparison exists
# AT the line as well. Cubes kept + drained (retention default post-#69). NEW campaign name:
# same-coordinate duplicates inside sanity_newton would break its grouping — the originals
# get unpublished once these validate (same supersede rule as the ladder linemaps).
CAMPAIGN=sanity_newton_lm
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=25
  EDM_ACCUM_ALG=newton
  EDM_DIRECT_READ=1
)
CELLS=(
  "g1_newton_lm|EDM_GAMMA_EPS=0 EDM_HARMONICS=1,0.992,2"
  "e1em3_newton_lm|EDM_GAMMA_EPS=1e-3 EDM_HARMONICS=1,1.0936,1.0853"
)
