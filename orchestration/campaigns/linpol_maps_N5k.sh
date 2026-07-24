# campaigns/linpol_maps_N5k.sh — linear-pol harmonic maps at N_electrons=5000 (vs 400 in
# linpol_maps.sh): ~√(5000/400) ≈ 3.5× lower shot-noise floor for the ∝a0 2ω at a0=1e-5. N is VRAM-free.
CAMPAIGN=linpol_maps_N5k
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=200 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=5000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_POL=linear                       # linear polarization (ξ = (1,0)); default is circular_minus
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2: corpus continuity with pre-PR#62 runs (arbitrary since #62 unified the φ₀ convention)
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em3|EDM_A0=1e-3"
  "a1e1|EDM_A0=0.1"
)
