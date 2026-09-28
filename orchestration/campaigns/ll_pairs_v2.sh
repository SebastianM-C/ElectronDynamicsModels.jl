# campaigns/ll_pairs_v2.sh — the three classical|LL pairs kept from the old LL ladders (inverse-Thomson session LL
# rerun set), at the old ladder framing: N 2000, 401², SPP 2048, tspan 16/γ, Newton 2 iterations, knots 16 per
# Doppler period, total mode, narrow window with 0.3 λ margins, cubes kept.
#   (a₀ 2, γ 1000, ±5 w₀) — ll_gamma_ladder g1000;  (a₀ 2, γ 4000, ±1 w₀) — ll_gamma_ladder g4000;
#   (a₀ 8, γ 4000, ±1 w₀) — new corner (a₀²γ = 2.6e5; χ ≈ 0.19, past the χ ≲ 0.1 classical-LL range).
# The harmonic maps are BEAM PATTERNS, not line maps: 4γ² = 4e6 / 6.4e7 ω₁ is far above Nyquist (1024 ω₁), so the
# extracted bins are the laser-harmonic content of the aliased signal (h1…h4, as the old ladders). The physics
# products are the γ(τ)/γ₀ drain traces (pair chip) and the per-electron emission-time/-end chips.
# Harmonics: 1,2,3,4 as the old ladders set explicitly — the script's :narrow default (n₀ ± 1, 2n₀ ≈ 4γ²) would
# trip the Nyquist guard at these γ. Rect reduction (EDM_APODIZATION=none), as the rest of the session.
# Sizes: γ 1000 Ns 4297 → 30.9 GiB; γ 4000 Ns 1856 → 13.3 GiB (both fit the H100 PCIe).
A8_ISA=${A8_ISA:-32}   # knots per Doppler period for the a₀ 8 pair (Sebastian: 32, as the old a₀ 5 LL cells)
CAMPAIGN=ll_pairs_v2
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
REDUCE_HOOK='export EDM_DIRECT_READ=1; _reduce_cell "$uuid"'   # overlap reducer reads ITS env, not BASE: O_DIRECT keeps the reduce at ~1.5× cube
SWEEP_AXES=system
BASE=(
  EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_SPP=2048 EDM_NX=401
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_HARMONICS=1,2,3,4
  EDM_APODIZATION=none
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
G1000="EDM_A0=2 EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_SCREEN_HW=5"
G4000A2="EDM_A0=2 EDM_GAMMA=4000 EDM_TSPAN_TAU=0.004 EDM_SCREEN_HW=1"
G4000A8="EDM_A0=8 EDM_GAMMA=4000 EDM_TSPAN_TAU=0.004 EDM_SCREEN_HW=1 EDM_INTERP_SAVEAT=$A8_ISA"
CELLS=(
  "g1000_a2_cl|$G1000"
  "g1000_a2_ll|$G1000 EDM_SYSTEM=ll"
  "g4000_a2_cl|$G4000A2"
  "g4000_a2_ll|$G4000A2 EDM_SYSTEM=ll"
  "g4000_a8_cl|$G4000A8"
  "g4000_a8_ll|$G4000A8 EDM_SYSTEM=ll"
)
