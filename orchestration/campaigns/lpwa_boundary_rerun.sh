# campaigns/lpwa_boundary_rerun.sh — the two ladder cells the pre-fix verify flake killed
# (lpwa.jl's forward-difference trajectory check; central-difference fix 2026-07-23).
# SAME campaign name: outputs join runs/lpwa_boundary and publish under the one campaign dir.
CAMPAIGN=lpwa_boundary
SCRIPT=scripts/lpwa.jl
KEEP_CUBE=1
BASE=(
  EDM_NX=400 EDM_FIELD_MODE=total
  EDM_N=10000
  EDM_NSAMPLES=12160 EDM_SPP=32          # matches newton_a0_high (Nyquist h16)
  EDM_INITIAL_PHASE=-1.5707963267948966  # phi0 = -pi/2, matches numeric side
)
CELLS=(
    "a5em1|EDM_A0=0.5"
    "a2|EDM_A0=2"
)
