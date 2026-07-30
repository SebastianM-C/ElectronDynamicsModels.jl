# campaigns/gamma1_crosscheck_direct.sh — γ=1 direct/inverse consistency check, DIRECT arm.
# Companion of gamma1_crosscheck_inverse.sh (same CAMPAIGN ⇒ one runs/ dir; launch both).
# The check: at γ=1 the inverse script's electron is at rest and only its laser direction
# differs (−z vs +z), so direct(laser +z, screen +Z) must reproduce inverse(laser −z,
# screen −Z) up to a definite transverse mirror (helicity/m are NOT parity-mapped by the
# reversed laser — the residual transform is what compare_mirror_runs.jl detects), and
# inverse(+Z vs −Z) shows the physical forward/backward asymmetry as the control.
# Geometry mirrors bunched_resolved base_a03 (a0=0.3, ±0.4 w₀ screen) at REST-FRAME
# sampling: SPP 16 (no upshift at γ=1; the bunched 2048 was for the 4γ² line), Ns 6000
# (375-period window ⊇ the 143λ signal), 201² (consistency metric, not coherence res).
# Windows align EXACTLY across the arms: both scripts reduce to
# x⁰_start = c·τi + hypot(Z, 0.4w₀ + Rmax) at these knobs — no phase offset in the diff.
CAMPAIGN=gamma1_crosscheck
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=1                      # keep cubes: a mismatch needs forensics, not a rerun
BASE=(
  EDM_NX=201 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_SCREEN_HALFW=0.4
)
CELLS=(
  "direct|"
)
