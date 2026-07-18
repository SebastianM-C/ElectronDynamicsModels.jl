# campaigns/ll_gamma_ladder.sh — Landau–Lifshitz radiation reaction through the old Float64 wall.
#
# LL-vs-classical pairs at a₀=2 on a γ-ladder {10, 100, 1000, 4000}; the pre-#44 light-front
# fix capped the kernel at γ≈2900 (a0-gamma-envelope report), so the γ=4000 pair is physics
# that did not exist before the hardened kernel. χ = 2γa₀·ħω/mc² ≈ 0.05 at the flagship —
# safely classical, LL valid. RR significance a₀²γ = 1.6e4 there.
#
# Research mode (no fixed-bin scoring): cubes KEPT everywhere; harmonic lists are only
# feasibility placeholders at high γ (the 4γ² line is unresolvable above γ≈100 at any
# realizable SPP — observables are worldline energy loss, powspec envelopes, and the kept
# cubes). Each pair shares the sunflower disk ⇒ the common-disk complex diff isolates the
# RR imprint (bunched_diff_products.jl pattern; pairs group by (a0, kernel) — here by system).
# Per-γ scaling: EDM_TSPAN_TAU ∝ 1/γ (proper-time span); screen half-width tracks the ~1/γ
# beaming cone at Z = 2e5λ (γ≤1000: default 5 w₀ covers it; γ=4000: cone ≈ 0.7 w₀ ⇒ hw=1).
# n_iters mini-ladder at γ=1000 (it2 = the pair cell) answers the open iteration-count
# question on the bracketed step.
CAMPAIGN=ll_gamma_ladder
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_A0=2 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_SPP=2048 EDM_NX=401 EDM_SCREEN_HW=5
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
)
CELLS=(
  "g10_cl|EDM_GAMMA=10 EDM_TSPAN_TAU=1.6 EDM_HARMONICS=199,299,398,597"
  "g10_ll|EDM_GAMMA=10 EDM_TSPAN_TAU=1.6 EDM_HARMONICS=199,299,398,597 EDM_SYSTEM=ll"
  "g100_cl|EDM_GAMMA=100 EDM_TSPAN_TAU=0.16 EDM_HARMONICS=1,2,3,4"
  "g100_ll|EDM_GAMMA=100 EDM_TSPAN_TAU=0.16 EDM_HARMONICS=1,2,3,4 EDM_SYSTEM=ll"
  "g1000_cl|EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_HARMONICS=1,2,3,4"
  "g1000_ll|EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_HARMONICS=1,2,3,4 EDM_SYSTEM=ll"
  "g1000_ll_it1|EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_HARMONICS=1,2,3,4 EDM_SYSTEM=ll EDM_NEWTON_ITERS=1"
  "g1000_ll_it3|EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_HARMONICS=1,2,3,4 EDM_SYSTEM=ll EDM_NEWTON_ITERS=3"
  "g4000_cl|EDM_GAMMA=4000 EDM_TSPAN_TAU=0.004 EDM_HARMONICS=1,2,3,4 EDM_SCREEN_HW=1"
  "g4000_ll|EDM_GAMMA=4000 EDM_TSPAN_TAU=0.004 EDM_HARMONICS=1,2,3,4 EDM_SCREEN_HW=1 EDM_SYSTEM=ll"
)
