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
# Screen shrinks with the boost (per-cell), bounded by SPECKLE RESOLUTION, not just the cone:
# the smallest screen feature is the disk-aperture grain at the UPSHIFTED wavelength,
# grain′ = λZ/(2Rmax)/n0 = 5.47 w₀/n0 — so hw = min(25/γ, 60·grain′) keeps ≥5 px/grain at
# 601². The γ=10 reference's ±0.4 w₀ was exactly this constraint in disguise (6.9 px/grain).
# The two lowest rungs carry exact fractional line anchors: the :narrow integer default
# (n0±1, 2n0) rounds 2.1% / 0.5% off their lines — half a linewidth at γ=1.5. γ ≥ 2's
# integers are ≤0.1% off; defaults suffice there. (Extraction is lab-frame n·ω₁ throughout.)
# SEGMENT 2 (user, 2026-07-30): densify the n0 = 14 → 98 gap where the decoherence knee
# may sit — γ = 2.5, 3.5, 4. Same per-cell discipline: SPP ≳ 2.5× the line (h2 inside
# Nyquist), TSPAN·γ = 16, hw = min(25/γ, 60·grain′), exact fractional anchors.
CELLS=(
  "e1p5e0|EDM_GAMMA_EPS=1.5 EDM_SPP=128 EDM_TSPAN_TAU=6.4 EDM_SCREEN_HW=10 EDM_HARMONICS=22,22.9564,24,45.9128"
  "e2p5e0|EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=7.0 EDM_HARMONICS=46,46.9788,48,93.9576"
  "e3e0|EDM_GAMMA_EPS=3 EDM_SPP=256 EDM_TSPAN_TAU=4 EDM_SCREEN_HW=5.3 EDM_HARMONICS=61,61.9843,63,123.9686"
)
