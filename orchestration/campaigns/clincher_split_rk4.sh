# campaigns/clincher_split_rk4.sh — ONE cell on BOTH vendors at the SAME commit (Hot Aisle
# MI300X + RunPod H200): the caveat-free cross-vendor ratio. Config = the June production cell
# (field_campaign_cm_899976) + EDM_INTERP_SAVEAT=16 (the floor fix); the delta vs the June H200
# numbers is the method evolution, and the kept cube doubles as a floor-fix robustness check on
# the published a0=0.1 maps.
CAMPAIGN=clincher_split_rk4
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_NX=400 EDM_FIELD_MODE=split
  EDM_N=10000 EDM_RELTOL=1e-12
  EDM_NSAMPLES=6000 EDM_SPP=16
  EDM_INTERP_SAVEAT=16
  EDM_ACCUM_ALG=rk4
  EDM_POL=circular_minus
  EDM_INITIAL_PHASE=0.0
)
CELLS=(
    "a1em1|EDM_A0=0.1"
)
