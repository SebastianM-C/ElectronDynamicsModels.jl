# campaigns/gamma1_ladder_frame.sh — the γ = 1 inverse cell in the emission-ladder framing: inverse script at
# rest (EDM_GAMMA=1, laser along −z), a₀ = 0.3, N = 10000, 400², full window Ns 6128 at SPP 16, ±25 w₀, φ₀ = −π/2,
# transmission-side screen (EDM_SCREEN_ZSIGN=−1 ↔ the ladder's +Z screen, up to the gamma1_crosscheck mirror).
# The partner is emission_ladder_numeric a3em1 (5c87d965). Rect reduction inline (the inverse script honours
# EDM_APODIZATION=none; the ladder needed its REDUCE_HOOK for that), so the ladder's sentinel checks don't run.
# Split mode (E, B, E_far, B_far): cube 87.7 GiB ⇒ refused by every local card's 90 % guard (NVL 84 GiB), and
# electron sharding does not help (each device holds a full buffer set). Total mode (E and B, 43.8 GiB) fits one
# card and still gives the E + B check (Sebastian: total). The ladder partner ran RK4; this cell runs Newton.
# FIELD_MODE picks the cell: total (one card, chosen) | split (needs a ≥ 98 GiB card).
FIELD_MODE=${FIELD_MODE:-total}
CAMPAIGN=gamma1_ladder_frame
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
REDUCE_HOOK='export EDM_DIRECT_READ=1; _reduce_cell "$uuid"'   # overlap reducer reads ITS env, not BASE: O_DIRECT keeps the reduce at ~1.5× cube
SWEEP_AXES=Z
BASE=(
  EDM_NX=400 EDM_N=10000
  EDM_NSAMPLES=6128 EDM_SPP=16
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_NSUBSTEPS=1
  EDM_RELTOL=1e-12 EDM_INTERP_SAVEAT=16
  EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_GAMMA=1 EDM_TSPAN_TAU=8 EDM_WINDOW=full EDM_SCREEN_HW=25
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "g1_a03_mz|EDM_A0=0.3 EDM_SCREEN_ZSIGN=-1 EDM_FIELD_MODE=$FIELD_MODE"
)
