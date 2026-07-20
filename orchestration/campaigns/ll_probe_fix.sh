# campaigns/ll_probe_fix.sh — makeup for a10_cl (host-OOM at cell start) and a5_ll (cube never reached the volume — pod thrashed during its end-phase serialize)
# (2026-07-19: an ssh ControlPath collision ran a second campaign's warm + cell startup on the
# same pod while a backgrounded reduce held ~100 GB — LLVM ERROR: out of memory at cell start).
CAMPAIGN=ll_probe
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=32
  EDM_SPP=8032000 EDM_NX=33 EDM_SCREEN_HW=0.27
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_HARMONICS=800000,1600000,2400000,3200000,3996000,4000000,4004000
)
CELLS=(
  "a10_cl|EDM_A0=10"
  "a5_ll|EDM_A0=5 EDM_SYSTEM=ll"
)
