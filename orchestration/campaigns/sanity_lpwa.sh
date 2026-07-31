# campaigns/sanity_lpwa.sh — analytic reference for the rest_departure sanity cross-check
# (the June field_campaign_cm_phi0 ↔ lpwa_campaign_split_899991 comparison, redone in total
# mode on the production ±25 w₀/601² framing, with the #69 crosscheck harness as the
# verification layer). lpwa.jl is the spline-free analytic rest-electron solution — the
# ground truth the numeric pipeline is checked against. One cell: lpwa is rest-only and
# rk4-only (no EDM_GAMMA / EDM_ACCUM_ALG knobs), so the γ and kernel axes live on the
# inverse side (rest_departure rest/e1em3 = rk4; sanity_newton = the Newton pair) and meet
# in compare_mirror_runs.jl pairs:
#   lpwa↔rest(rk4)  lpwa↔g1(newton)  rest↔g1-newton  e1em3↔e1em3-newton  lpwa↔e1em3
# φ₀: both scripts define EDM_INITIAL_PHASE on the E-field carrier post-PR #62 — same value,
# no π/2 compensation (the June recipes' 0 vs −π/2 pairing is obsolete).
CAMPAIGN=sanity_lpwa
SCRIPT=scripts/lpwa.jl
# a0 ladder (linear → weakly relativistic → the campaign point): each rung pairs with an
# inverse rk4 twin — a0 ≤ 0.1 twins live in sanity_inverse, the 0.3 twin is rest_departure's
# own `rest` cell.
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_SCREEN_HW=25
  EDM_DIRECT_READ=1
  # lpwa.jl's assert_committed refuses the VM's tree: the DRIVER rsyncs its orchestration/
  # over the clone on every launch, and a driver checkout on an older branch leaves
  # orchestration files modified (solver code untouched — provenance still records dirty).
  EDM_ALLOW_DIRTY=1
)
CELLS=(
  "a1em2|EDM_A0=1e-2"
  "a1em1|EDM_A0=1e-1"
  "a3em1|EDM_A0=3e-1"
)
