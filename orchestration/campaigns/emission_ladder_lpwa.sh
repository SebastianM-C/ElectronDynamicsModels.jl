# emission_ladder_lpwa (branch campaigns/emission-ladder) — the one-go a₀ ladder, analytic side (lpwa.jl), twin of
# emission_ladder_numeric.sh: same screen, electrons, window, sampling, φ₀ and rect reduction. lpwa.jl is RK4-only and
# spline-free (no saveat/reltol/kernel knobs); n_substeps follows the numeric side (1 for a₀ ≤ 1, NS_HIGH above).
CAMPAIGN=emission_ladder_lpwa
SCRIPT=scripts/lpwa.jl
KEEP_CUBE=0
REDUCE_OVERLAP=1
REDUCE_HOOK='{ flock 9; EDM_APODIZATION=none EDM_DIRECT_READ=1 _reduce_cell "$uuid"; } 9>"/tmp/edm-reduce-$(id -un).lock"'
SWEEP_AXES=a0
NS_HIGH_RECOMMENDED=1
NS_HIGH=UNSET_run_kernel_conv_diag_first
BASE=(
  EDM_NX=400 EDM_N=10000 EDM_FIELD_MODE=total
  EDM_NSAMPLES=6000 EDM_SPP=16
  EDM_NSUBSTEPS=1
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em4|EDM_A0=1e-4"
  "a1em3|EDM_A0=1e-3"
  "a1em2|EDM_A0=1e-2"
  "a1em1|EDM_A0=0.1"
  "a2em1|EDM_A0=0.2"
  "a5em1|EDM_A0=0.5"
  "a1|EDM_A0=1"
  "a2|EDM_A0=2 EDM_NSUBSTEPS=$NS_HIGH"
  "a5|EDM_A0=5 EDM_NSUBSTEPS=$NS_HIGH"
  "a10|EDM_A0=10 EDM_NSUBSTEPS=$NS_HIGH"
)
