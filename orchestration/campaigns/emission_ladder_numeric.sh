# emission_ladder_numeric (branch campaigns/emission-ladder) — the one-go a₀ ladder, numeric side
# (thomson_scattering.jl), for the emission-boundary figures. Configuration from the dcshelf diagnosis (2026-09-27):
# EDM main (v0.4.3 pin), total mode, uniform trajectory knots, rect reduction: the inline reduction is Hann-only, so
# REDUCE_OVERLAP=1 defers it to harmonic_products.jl and REDUCE_HOOK hands that reducer EDM_APODIZATION=none +
# EDM_DIRECT_READ=1 (it reads them from its own env, not BASE) under a per-host lock (one reduce at a time per host);
# SPP 16 / Ns 6128 (Figs 7–8 use h1 and h2), φ₀ = −π/2. Kernel: RK4 (rest electrons; same kernel as lpwa.jl, which is
# RK4-only). n_substeps 1 for a₀ ≤ 1 and NS_HIGH for a₀ ≥ 2. Twin: emission_ladder_lpwa.sh. Lane wrappers in
# emission_ladder/.
# kernel_conv_diag (2026-09-27): RK4 ns1 is within 2e-5 (rel. L2, h1/h2 maps, E and B) of the converged maps even at
# a₀ = 10 (ns4 gains ~1 digit on h1 at ~1.8× field time) ⇒ NS_HIGH = 1, approved 2026-09-27.
NS_HIGH_RECOMMENDED=1
NS_HIGH=$NS_HIGH_RECOMMENDED
CAMPAIGN=emission_ladder_numeric
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=1   # cubes kept for post-hoc reduction; drained to R2 (cloud backends; local: LOCAL_DRAIN_R2=1)
REDUCE_OVERLAP=1
LADDER_SIDE=numeric
. "$(dirname "${BASH_SOURCE[0]}")/emission_ladder/post_reduce.sh"   # ladder_reduce: rect reduce + checks + diagnostics
REDUCE_HOOK='ladder_reduce "$uuid"'
SWEEP_AXES=a0
BASE=(
  EDM_NX=400 EDM_N=10000 EDM_FIELD_MODE=total
  EDM_NSAMPLES=6128 EDM_SPP=16   # 383 periods: +8 over 375 covers the a0 ≥ 5 burst tail (Sebastian 2026-09-27; harmonics stay on-bin)
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
