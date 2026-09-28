# campaigns/rest_departure_v2_near.sh — the nine near-rest rungs (γ = 1 + ε, ε = 1e-4 … 0.2) re-acquired at 401²
# with the full window, as the near-rest lane of the bridge redo (rest_departure_v2.sh). Framing is
# rest_departure / rest_departure_linemaps (±25 w₀ production screen, Ns 6000 at SPP 16, tspan 8 τ, N 2000),
# the harmonics are the linemaps' line anchors (h1, n_th, n_ps; ε = 1e-4 has no measured n_ps: n_th only).
# Changes vs as-run: 401² (was 601²), Newton (was RK4), rect reduction, cubes kept. Cube 43.1 GiB (NVL/PCIe).
CAMPAIGN=rest_departure_v2_near
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
REDUCE_HOOK='export EDM_DIRECT_READ=1; _reduce_cell "$uuid"'   # overlap reducer reads ITS env, not BASE: O_DIRECT keeps the reduce at ~1.5× cube
SWEEP_AXES=gamma_eps
BASE=(
  EDM_NX=401 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=25
  EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "e1em4|EDM_GAMMA_EPS=1e-4 EDM_HARMONICS=1,1.0287"
  "e1em3|EDM_GAMMA_EPS=1e-3 EDM_HARMONICS=1,1.0936,1.0853"
  "e2em3|EDM_GAMMA_EPS=2e-3 EDM_HARMONICS=1,1.1348,1.1253"
  "e5em3|EDM_GAMMA_EPS=5e-3 EDM_HARMONICS=1,1.2213,1.2107"
  "e1em2|EDM_GAMMA_EPS=1e-2 EDM_HARMONICS=1,1.3266,1.3173"
  "e2em2|EDM_GAMMA_EPS=2e-2 EDM_HARMONICS=1,1.4908,1.48"
  "e5em2|EDM_GAMMA_EPS=5e-2 EDM_HARMONICS=1,1.8773,1.8613"
  "e1em1|EDM_GAMMA_EPS=1e-1 EDM_HARMONICS=1,2.4282,2.408"
  "e2em1|EDM_GAMMA_EPS=2e-1 EDM_HARMONICS=1,3.4724,3.4453"
)
