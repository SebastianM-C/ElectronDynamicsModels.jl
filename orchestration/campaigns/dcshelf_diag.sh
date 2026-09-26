# dcshelf_diag (branch campaigns/dcshelf-diag) — small forward-Thomson cells for the E-side "DC shelf" that leaked into
# the 2ω bin below a₀ ≈ 0.01 in the June ladder. There the shelf was the static Coulomb field TRUNCATED at the observer-
# window end (per-pixel samples past the solved trajectory got no field; fixed by the window-end coverage, EDM d1d8c99).
# These cells check the fix at HEAD: flat Coulomb level, no tail zeros/steps, and the E-vs-B 2ω floor.
# Levers: total vs split (E_far isolates the acceleration part), window length (Ns 6000 → 12000), a₀ 1e-5 vs 1e-3.
# Cubes kept for the per-pixel traces and offline rect/Hann spectra.
CAMPAIGN=dcshelf_diag
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=a0,mode,N_samples
SWEEP_DESIGN=oat
BASE=(
  EDM_NX=64 EDM_N=400 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em5_split|EDM_A0=1e-5 EDM_FIELD_MODE=split"
  "a1em5_ns12k|EDM_A0=1e-5 EDM_NSAMPLES=12000"
  "a1em3|EDM_A0=1e-3"
)
