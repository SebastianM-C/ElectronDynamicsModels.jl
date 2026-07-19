# campaigns/ll_a0_ladder.sh — RR a₀-scaling at γ=1000 (chained after ll_gamma_ladder).
# Pairs at a₀ {0.5, 5} fill the a₀²γ axis around the γ-ladder's a₀=2 point (RR ∝ a₀²γ:
# 2.5e2 / 2.5e4 ≈ 0.001% / 0.9% per pulse); the γ=2000 pair extends the γ-ladder between
# 1000 and 4000. saveat 32 on a₀=5 (high-a₀ spline knot density; floor lesson). Cubes kept.
CAMPAIGN=ll_a0_ladder
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_SPP=2048 EDM_NX=401 EDM_SCREEN_HW=5 EDM_HARMONICS=1,2,3,4
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
)
CELLS=(
  "a05_cl|EDM_A0=0.5"
  "a05_ll|EDM_A0=0.5 EDM_SYSTEM=ll"
  "a5_cl|EDM_A0=5 EDM_INTERP_SAVEAT=32"
  "a5_ll|EDM_A0=5 EDM_INTERP_SAVEAT=32 EDM_SYSTEM=ll"
  "g2000_cl|EDM_A0=2 EDM_GAMMA=2000 EDM_TSPAN_TAU=0.008"
  "g2000_ll|EDM_A0=2 EDM_GAMMA=2000 EDM_TSPAN_TAU=0.008 EDM_SYSTEM=ll"
)
