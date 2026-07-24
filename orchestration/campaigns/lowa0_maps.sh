# campaigns/lowa0_maps.sh — low-a0 radiated-field harmonic maps.
# Uniform EDM_INTERP_SAVEAT=16 (the floor fix): recovers the physical ∝a0 2ω vs adaptive-Vern9's spurious 2ω.
CAMPAIGN=lowa0_maps
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=400 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=10000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2: corpus continuity with pre-PR#62 runs (arbitrary since #62 unified the φ₀ convention)
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em6|EDM_A0=1e-6"
  "a1em7|EDM_A0=1e-7"
)
