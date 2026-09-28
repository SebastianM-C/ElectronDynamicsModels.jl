# campaigns/alias_ladder_g5_a03.sh — the Fig. 16 N ladder at the mgpu STRONG framing: γ = 5 (γ_eps 4, n₀ = 97.99),
# a₀ = 0.3, ±3 w₀ at 401², SPP 512, N = 2000 / 4000 / 8000. Same framing as the published N = 16000 strong-scaling
# maps (mgpu_bench, sweep mgpu_strong), so the four N share one screen and one line anchor set; the alias_ladder_g5
# (a₀ = 0.01, ±6 w₀, 441²) dir stays intact. Newton kernel at production defaults (the mgpu recipe's article
# kernel knobs — no coefficient reuse, five sample chunks, device reduce — are benchmark settings and are dropped).
# Ns = 1666 (narrow window, N-independent) ⇒ cube 12.0 GiB per cell: fits the A100 (36 GiB guard).
# Harmonics: n_ps (the measured powspec peak), n_th ≈ 98, and their doubles — the mgpu STRONG set.
CAMPAIGN=alias_ladder_g5_a03
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
REDUCE_HOOK='export EDM_DIRECT_READ=1; _reduce_cell "$uuid"'   # overlap reducer reads ITS env, not BASE: O_DIRECT keeps the reduce at ~1.5× cube
SWEEP_AXES=N
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
  EDM_GAMMA_EPS=4 EDM_SPP=512 EDM_TSPAN_TAU=3.2 EDM_SCREEN_HW=3.0
  EDM_HARMONICS=97.1395,98,193.6813,196
)
CELLS=(
  "N2000|EDM_N=2000"
  "N4000|EDM_N=4000"
  "N8000|EDM_N=8000"
)
