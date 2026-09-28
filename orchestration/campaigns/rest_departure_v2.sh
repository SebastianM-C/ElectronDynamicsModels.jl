# campaigns/rest_departure_v2.sh — the bridge redo at production statistics: γ = 4 (N 8000), 5 (N 16000),
# 10 (N 32000) with the rest_departure_bridge_refix conventions (601², narrow burst-centred window, rect
# reduction, TSPAN·γ = 16, SPP ≳ 2.5× the line, exact fractional anchors) and Newton, cubes kept. The γ = 10
# screen widens from the bridge's ±0.8 to ±1.0 w₀ (inverse_probe_g10 checks it first). Near-rest rungs:
# rest_departure_v2_near.sh (separate lane and campaign dir: full window, SPP 16).
# Sizes (48 B·Ns·601²): γ4 Ns 1265 → 20.4 GiB; γ5 Ns 1713 → 27.7 GiB; γ10 Ns 3413 → 55.1 GiB (NVL/PCIe only).
# EDM_ELECTRON_BATCH=4000 bounds the host splines at N ≥ 16000 (~150 GiB unbatched at N = 32000).
CAMPAIGN=rest_departure_v2
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
REDUCE_HOOK='export EDM_DIRECT_READ=1; _reduce_cell "$uuid"'   # overlap reducer reads ITS env, not BASE: O_DIRECT keeps the reduce at ~1.5× cube
SWEEP_AXES=gamma_eps  # SPP/tspan/hw/N co-vary with γ; declared so the dir stays one ladder
BASE=(
  EDM_NX=601 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "e3e0|EDM_GAMMA_EPS=3 EDM_N=8000 EDM_SPP=256 EDM_TSPAN_TAU=4 EDM_SCREEN_HW=5.3 EDM_HARMONICS=61,61.9843,63,123.9686"
  "e4e0|EDM_GAMMA_EPS=4 EDM_N=16000 EDM_ELECTRON_BATCH=4000 EDM_SPP=512 EDM_TSPAN_TAU=3.2 EDM_SCREEN_HW=3.3 EDM_HARMONICS=97.1395,97.9898,193.6813,195.9796"
  "e9e0|EDM_GAMMA_EPS=9 EDM_N=32000 EDM_ELECTRON_BATCH=4000 EDM_SPP=2048 EDM_TSPAN_TAU=1.6 EDM_SCREEN_HW=1.0 EDM_HARMONICS=397,397.9975,399,795.995"
)
