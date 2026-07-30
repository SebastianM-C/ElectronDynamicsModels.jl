# campaigns/rest_departure_seg2.sh — segment 2 of the γ→1⁺ ladder: ε continued UPWARD.
# Same CAMPAIGN as segment 1 ⇒ same runs/ dir, same union campaign, and — because BASE is
# byte-identical — all ten runs group into ONE dashboard gamma_eps sweep (any knob change
# would fork the canonical params and split the sweep; that is why the windowing is frozen).
#
# Rung rationale (backscatter line at ω′/ω₀ = (1+β)/(1−β)):
#   ε = 2e-2 → 1.49 ω₀ (+49%)     ε = 5e-2 → 1.88 ω₀ (+88%)
#   ε = 1e-1 → 2.43 ω₀ (+143%)    ε = 2e-1 → 3.47 ω₀ (+247%)
# Windowing headroom that sets the ε ≤ 2e-1 cap: SPP=16 ⇒ Nyquist 8 ω₀; the top rung's
# 2nd harmonic sits at 6.94 ω₀ — still inside. One decade higher (ε=5e-1 → line 6.85 ω₀,
# h2 aliased) needs SPP≥32 + a NEW campaign name (segment-3 territory, deliberately split).
# Spectral resolution stays 1/(Ns/SPP = 375 periods) ≈ 0.27% — far finer than any rung gap.
# Cell cost unchanged from segment 1 (cube geometry identical): ~30 min/cell on MI300X.
CAMPAIGN=rest_departure
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=1.2
  EDM_DIRECT_READ=1
)
CELLS=(
  "e2em2|EDM_GAMMA_EPS=2e-2"
  "e5em2|EDM_GAMMA_EPS=5e-2"
  "e1em1|EDM_GAMMA_EPS=1e-1"
  "e2em1|EDM_GAMMA_EPS=2e-1"
)
