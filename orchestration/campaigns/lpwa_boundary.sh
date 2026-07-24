# campaigns/lpwa_boundary.sh — analytic LPWA side of the emission R3 boundary ladder, matched
# cell-for-cell to newton_a0_high (grid, window, phi0, a0 values, TOTAL mode). lpwa.jl exposes
# no POL/SAVEAT/ACCUM knobs (analytic circular_minus trajectories), so BASE carries only what
# exists. Cubes kept for re-analysis (~90 GB each — R2 drain); total mode at Ns=12160 peaked at
# ~95 GB VRAM (newton_a0_high on MI300X): needs an MI300X-class card, does NOT fit the W7900.
CAMPAIGN=lpwa_boundary
SCRIPT=scripts/lpwa.jl
KEEP_CUBE=1
BASE=(
  EDM_NX=400 EDM_FIELD_MODE=total
  EDM_N=10000
  EDM_NSAMPLES=12160 EDM_SPP=32          # matches newton_a0_high (Nyquist h16)
  EDM_INITIAL_PHASE=-1.5707963267948966  # φ0 = -π/2: corpus continuity with pre-PR#62 runs (arbitrary since #62 unified the φ₀ convention)
)
CELLS=(
    "a2em1|EDM_A0=0.2"
    "a5em1|EDM_A0=0.5"
    "a1|EDM_A0=1"
    "a2|EDM_A0=2"
    "a5|EDM_A0=5"
    "a10|EDM_A0=10"
)
