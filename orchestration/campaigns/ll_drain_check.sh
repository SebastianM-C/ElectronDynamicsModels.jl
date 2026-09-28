# campaigns/ll_drain_check.sh — radiation-reaction drain check at the ll_probe_s configuration, shrunk to a
# probe: the a₀ = 100 classical|LL pair at N = 500 on a 33² screen. Everything else is ll_probe_s (γ = 100,
# SPP 170000, ±0.00619 w₀, narrow window, Newton), so the γ(τ)/γ₀ traces and the emission-time chips are
# the ll_probe_s ones at 1/4 the statistics; the screen is too coarse for speckle, by design (drain only).
# Cube ≈ 1.3 GiB (Ns 26677 × 33²). Pair chip (gammatau) needs both cells in the same dir.
CAMPAIGN=ll_drain_check
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=system
BASE=(
  EDM_GAMMA=100 EDM_TSPAN_TAU=0.16 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=500 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=64
  EDM_SPP=170000 EDM_NX=33 EDM_SCREEN_HW=0.00619
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_HARMONICS=8000,16000,24000,32000,39898,39998,40098,60000,79896,79996,80096
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
  EDM_A0=100
)
CELLS=(
  "a100_cl|"
  "a100_ll|EDM_SYSTEM=ll"
)
