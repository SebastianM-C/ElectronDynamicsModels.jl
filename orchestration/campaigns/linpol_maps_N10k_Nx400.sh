# campaigns/linpol_maps_N10k_Nx400.sh — hi-res linear-pol harmonic maps: EDM_NX=400 (vs 200 in
# linpol_maps_N5k) and N_electrons=10000 (vs 5000). Nx doubles → ~4× the field-accumulation work;
# N doubles → ~2× → ~8× the per-cell cost of linpol_maps_N5k. Bump EDM_N to 15000 for a ~1.22× lower
# shot-noise floor (√N) at ~1.5× the cost. Same physics as N5k (linear pol, φ0=-π/2, uniform saveat).
CAMPAIGN=linpol_maps_N10k_Nx400
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=400 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=10000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_POL=linear                       # linear polarization (ξ = (1,0))
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2 (published convention)
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em3|EDM_A0=1e-3"
  "a1e1|EDM_A0=0.1"
)
