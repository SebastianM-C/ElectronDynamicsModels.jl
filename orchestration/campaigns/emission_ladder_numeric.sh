# emission_ladder_numeric (branch campaigns/emission-ladder) — the one-go a₀ ladder, numeric side
# (thomson_scattering.jl), for the emission-boundary figures. Configuration from the dcshelf diagnosis (2026-09-27):
# EDM e33cc1a, total mode, uniform trajectory knots, rect reduction (EDM_APODIZATION=none, applied by the deferred
# reducer: REDUCE_OVERLAP=1 AND EDM_APODIZATION=none exported in the driver env — the inline reduction is Hann-only),
# SPP 16 / Ns 6000 (Figs 7–8 use h1 and h2), φ₀ = −π/2. Kernel: RK4 (rest electrons; same kernel as lpwa.jl, which is
# RK4-only). n_substeps 1 for a₀ ≤ 1; for a₀ ≥ 2 set NS_HIGH from kernel_conv_diag before launch — until then those
# cells fail at their first parse. Twin: emission_ladder_lpwa.sh. Lane wrappers in emission_ladder/.
NS_HIGH=UNSET_run_kernel_conv_diag_first
CAMPAIGN=emission_ladder_numeric
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=0
REDUCE_OVERLAP=1
SWEEP_AXES=a0
BASE=(
  EDM_NX=400 EDM_N=10000 EDM_FIELD_MODE=total
  EDM_NSAMPLES=6000 EDM_SPP=16
  EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=1
  EDM_RELTOL=1e-12 EDM_INTERP_SAVEAT=16
  EDM_APODIZATION=none EDM_DIRECT_READ=1
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
