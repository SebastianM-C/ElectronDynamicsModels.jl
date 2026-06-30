# campaigns/smalla0_floor.sh — small-a0 2ω floor investigation (OAT / star design).
# PURE DATA. One baseline + independent 1-D arms (a0, interp_saveat, N, reltol, n_substeps). The
# dashboard renders this as a hub-and-arms star (see the OAT family/hub support in the builder).
# Run on any backend, e.g.:  bash orchestration/backends/local.sh orchestration/campaigns/smalla0_floor.sh
# (cuda vs rocm is a BACKEND choice — the old "baseWS" rocm cross-check is just this baseline on rocm.)
CAMPAIGN=smalla0_floor
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=200 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_A0=1e-5 EDM_N=400 EDM_RELTOL=1e-12 EDM_NSUBSTEPS=1
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "base|"                              # the hub
  "a1e4|EDM_A0=1e-4"  "a1e3|EDM_A0=1e-3"  "a1e2|EDM_A0=1e-2"  "a1e1|EDM_A0=0.1"      # a0 arm
  "sa16|EDM_INTERP_SAVEAT=16" "sa32|EDM_INTERP_SAVEAT=32"                            # saveat arm (the key test)
  "sa64|EDM_INTERP_SAVEAT=64" "sa128|EDM_INTERP_SAVEAT=128"
  "N1600|EDM_N=1600" "N3200|EDM_N=3200" "N6400|EDM_N=6400" "N12800|EDM_N=12800"      # N arm (shot-noise)
  "rt14|EDM_RELTOL=1e-14"                                                            # solve-accuracy recheck
  "ns8|EDM_NSUBSTEPS=8"                                                              # march recheck
)
