# campaigns/inverse_probe_g10.sh — γ = 10 inverse probe for the bridge redo's top rung: a₀ = 0.3, N = 2000,
# ±1 w₀ at 401², SPP 2048 (n₀ = 398, h2 inside Nyquist 1024), narrow window, Newton. Checks the ±1 w₀ screen
# (the bridge e9e0 used ±0.8) and the window before the N = 32000 cell. Ns ≈ 3413, cube ≈ 24.5 GiB (fits the
# 5090's 90 % guard with ~4 GiB to spare). Harmonics: n₀ ± 1 (integer), n_th, 2n_th.
CAMPAIGN=inverse_probe_g10
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=a0
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
  EDM_GAMMA_EPS=9 EDM_SPP=2048 EDM_TSPAN_TAU=1.6 EDM_SCREEN_HW=1.0
  EDM_HARMONICS=397,397.9975,399,795.995
)
CELLS=(
  "g10_a03|EDM_A0=0.3"
)
