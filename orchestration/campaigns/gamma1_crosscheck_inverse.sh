# campaigns/gamma1_crosscheck_inverse.sh — γ=1 direct/inverse consistency check, INVERSE arm.
# Companion of gamma1_crosscheck_direct.sh (same CAMPAIGN ⇒ one runs/ dir) — see its header
# for the full rationale. γ=1 ⇒ electron at rest (β=0), laser reversed to −z:
#   inv_mz (screen −Z, transmission side) ↔ the direct run's +Z screen;
#   inv_pz (screen +Z, backscatter side)  — the forward/backward asymmetry control.
# EDM_TSPAN_TAU=8 restores thomson tspan parity (the bunched 1.6 fit only the γ=10
# compressed burst); EDM_WINDOW=full is the thomson-equivalent window and lands on the
# identical x⁰_start at these knobs. Tolerances matched to the direct arm explicitly.
CAMPAIGN=gamma1_crosscheck
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_NX=201 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_GAMMA=1 EDM_TSPAN_TAU=8 EDM_WINDOW=full EDM_SCREEN_HW=0.4
)
CELLS=(
  "inv_mz|EDM_SCREEN_ZSIGN=-1"
  "inv_pz|EDM_SCREEN_ZSIGN=1"
)
