# campaigns/ll_probe_rerun.sh — the 5 ll_probe cells lost to the 2026-07-19 disk-overflow
# incident (120 GB pod disk, drainer uploads but never deletes locally ⇒ cube 2 truncated at
# the quota, cells 3–6 got 8 KiB stubs; cell a2_cl survived and is archived). Same physics as
# ll_probe — run on a pod with a NETWORK VOLUME (RUNPOD_VOLUME_GB≈1400) so cubes land off the
# container disk entirely; the volume drains via orchestration/drain.sh after the campaign.
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
  "a2_ll|EDM_A0=2 EDM_SYSTEM=ll"
  "a10_cl|EDM_A0=10"
  "a10_ll|EDM_A0=10 EDM_SYSTEM=ll"
  "a5_cl|EDM_A0=5"
  "a5_ll|EDM_A0=5 EDM_SYSTEM=ll"
)
