# campaigns/linpol_maps.sh — linear-polarization radiated-field harmonic maps (Nx=200, N=400).
# PURE DATA: no infra, no backend, no run-tags (the core mints UUIDs). Runs on any backend:
#   bash orchestration/backends/local.sh   orchestration/campaigns/linpol_maps.sh
#   bash orchestration/backends/hotaisle.sh orchestration/campaigns/linpol_maps.sh
#   sbatch orchestration/backends/slurm.sbatch orchestration/campaigns/linpol_maps.sh
#
# Same harmonic-map recipe as lowa0_maps (uniform EDM_INTERP_SAVEAT=16 floor fix, φ0=-π/2 published
# convention) but with EDM_POL=linear instead of the default circular_minus, so it is directly
# comparable to the circular runs at matched a0. a0 spans 1e-5 (deep-linear ∝a0) to 0.1.
CAMPAIGN=linpol_maps
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=200 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=400 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_POL=linear                       # linear polarization (ξ = (1,0)); default is circular_minus
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2 (published convention)
)
CELLS=(
  "a1em5|EDM_A0=1e-5"
  "a1em3|EDM_A0=1e-3"
  "a1e1|EDM_A0=0.1"
)
