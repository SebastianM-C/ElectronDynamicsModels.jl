# emission_ladder_lpwa (branch campaigns/emission-ladder) — the one-go a₀ ladder, analytic side (lpwa.jl), twin of
# emission_ladder_numeric.sh: same screen, electrons, window, sampling, φ₀ and rect reduction. lpwa.jl is RK4-only and
# spline-free (no saveat/reltol/kernel knobs); n_substeps follows the numeric side (1 for a₀ ≤ 1, NS_HIGH above).
CAMPAIGN=emission_ladder_lpwa
SCRIPT=scripts/lpwa.jl
KEEP_CUBE=1   # cubes kept for post-hoc reduction; drained to R2 (cloud backends; local: LOCAL_DRAIN_R2=1)
REDUCE_OVERLAP=1
LADDER_SIDE=lpwa
. "$(dirname "${BASH_SOURCE[0]}")/emission_ladder/post_reduce.sh"   # ladder_reduce: rect reduce + checks + diagnostics
REDUCE_HOOK='ladder_reduce "$uuid"'
SWEEP_AXES=a0
NS_HIGH_RECOMMENDED=1
NS_HIGH=$NS_HIGH_RECOMMENDED
BASE=(
  EDM_NX=400 EDM_N=10000 EDM_FIELD_MODE=total
  EDM_NSAMPLES=6128 EDM_SPP=16   # 383 periods: +8 over 375 covers the a0 ≥ 5 burst tail (Sebastian 2026-09-27; harmonics stay on-bin)
  EDM_NSUBSTEPS=1
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em4|EDM_A0=1e-4"
  "a1em3|EDM_A0=1e-3"
  "a1em2|EDM_A0=1e-2"
  "a5em2|EDM_A0=0.05"
  "a1em1|EDM_A0=0.1"
  "a13em2|EDM_A0=0.13"
  "a16em2|EDM_A0=0.16"
  "a2em1|EDM_A0=0.2"
  "a25em2|EDM_A0=0.25"
  "a3em1|EDM_A0=0.3"
  "a5em1|EDM_A0=0.5"
  "a1|EDM_A0=1"
  "a2|EDM_A0=2 EDM_NSUBSTEPS=$NS_HIGH"
  "a5|EDM_A0=5 EDM_NSUBSTEPS=$NS_HIGH"
  "a10|EDM_A0=10 EDM_NSUBSTEPS=$NS_HIGH"
  "a20|EDM_A0=20 EDM_NSUBSTEPS=$NS_HIGH"   # extent probe: joins the ladder only if window- and alias-clean
)
