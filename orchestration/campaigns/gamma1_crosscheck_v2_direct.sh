# campaigns/gamma1_crosscheck_v2_direct.sh — gamma1_crosscheck re-acquired at current main (inverse-Thomson
# session probes), DIRECT arm. Recipe unchanged from gamma1_crosscheck_direct.sh (see its header for the
# rationale: direct(+z laser, +Z screen) ↔ inverse(−z laser, −Z screen) up to a transverse mirror); only the
# campaign name is new so the published gamma1_crosscheck dir stays intact. Kernel stays RK4 (as-run arms).
# Companion: gamma1_crosscheck_v2_inverse.sh (same CAMPAIGN ⇒ one runs/ dir; launch both).
CAMPAIGN=gamma1_crosscheck_v2
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=Z         # the backfilled gamma1_crosscheck declaration's axis (signed screen Z)
BASE=(
  EDM_NX=201 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_SCREEN_HALFW=0.4
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "direct|"
)
