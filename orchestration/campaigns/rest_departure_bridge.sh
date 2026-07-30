# campaigns/rest_departure_bridge.sh — segment 3: bridge γ from the near-rest ladder to the
# boosted regime, ε = 0.5 → 9 (γ = 1.5 → 10, backscatter line n0 ≈ 6.9 → 398 ω₀).
#
# NEW campaign name — a deliberate sweep split from rest_departure: the frozen full-window
# SPP=16 framing runs out of Nyquist above ε = 0.2, so the bridge switches to the burst-centred
# EDM_WINDOW=narrow with per-cell SPP ≳ 2.5× the line (h2 of the γ ≥ 3 rungs stays beyond
# Nyquist — weak at a0 = 0.3, accepted for exploration). N_samples is sized by the narrow
# window itself: bursts shrink ∝ 1/n0, so cubes stay far below segment 1's 97 GiB at the same
# 601²/±1.2 w₀ screen. TSPAN_TAU follows the γ-ladder convention TSPAN·γ = 16 (γ-free knot
# count and cost); ε = 9 lands exactly on the bunched_resolved reference framing
# (SPP 2048, tspan 1.6τ — run b35964ec, the config this whole family descends from).
# RELTOL 1e-13 per the boosted-solver anti-skip requirement; ABSTOL = the script's boosted
# default 1e-11. Untested corner: narrow-window sizing at n0 ~ 7 (ε = 0.5) — the burst is only
# mildly compressed there; truncation leakage in the powspec flags it if the window clips.
CAMPAIGN=rest_departure_bridge
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_DIRECT_READ=1
)
# Screen shrinks with the boost (per-cell): the forward cone narrows ∝ 1/γ, so hw = 25/γ w₀
# keeps the pattern envelope filling the frame — continuous with segment 1/2's production
# ±25 w₀ at γ→1, and still 6× wider than the γ=10 reference's ±0.4 detail zoom at the top.
CELLS=(
  "e5em1|EDM_GAMMA_EPS=0.5 EDM_SPP=32 EDM_TSPAN_TAU=10.666666666666666 EDM_SCREEN_HW=16.7"
  "e1e0|EDM_GAMMA_EPS=1 EDM_SPP=64 EDM_TSPAN_TAU=8 EDM_SCREEN_HW=12.5"
  "e2e0|EDM_GAMMA_EPS=2 EDM_SPP=128 EDM_TSPAN_TAU=5.333333333333333 EDM_SCREEN_HW=8.3"
  "e4e0|EDM_GAMMA_EPS=4 EDM_SPP=512 EDM_TSPAN_TAU=3.2 EDM_SCREEN_HW=5"
  "e9e0|EDM_GAMMA_EPS=9 EDM_SPP=2048 EDM_TSPAN_TAU=1.6 EDM_SCREEN_HW=2.5"
)
