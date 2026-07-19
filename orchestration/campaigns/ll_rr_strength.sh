# campaigns/ll_rr_strength.sh — into the ~10% radiation-reaction regime (chained last).
# γ=4000 × a₀ {4, 6, 8}: ΔE/E per pulse ≈ 3.5e-7·a₀²γ ⇒ 2% / 5% / 9%; χ = 0.10/0.15/0.20 —
# the ladder walks LL from exact (χ≈0.05) to its first quantum-correction territory (χ≈0.2).
# The γ=2000, a₀=8 pair (χ=0.10, ΔE/E≈4.5%) separates a₀- from χ-dependence. a₀=8 sits at the
# measured solver-envelope edge (clean a₀≈6–9) — envelope stress-test by construction.
# saveat 32 everywhere (a₀ ≥ 4); expect the 4γ² line to WALK ~20% during the pulse at a₀=8
# (γ-decay chirp) — powspec + kept cubes are the observables, bins are placeholders.
CAMPAIGN=ll_rr_strength
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_GAMMA=4000 EDM_TSPAN_TAU=0.004 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=32
  EDM_SPP=2048 EDM_NX=401 EDM_SCREEN_HW=1 EDM_HARMONICS=1,2,3,4
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
)
CELLS=(
  "a4_cl|EDM_A0=4"
  "a4_ll|EDM_A0=4 EDM_SYSTEM=ll"
  "a6_cl|EDM_A0=6"
  "a6_ll|EDM_A0=6 EDM_SYSTEM=ll"
  "a8_cl|EDM_A0=8"
  "a8_ll|EDM_A0=8 EDM_SYSTEM=ll"
  "g2000a8_cl|EDM_A0=8 EDM_GAMMA=2000 EDM_TSPAN_TAU=0.008 EDM_SCREEN_HW=2"
  "g2000a8_ll|EDM_A0=8 EDM_GAMMA=2000 EDM_TSPAN_TAU=0.008 EDM_SCREEN_HW=2 EDM_SYSTEM=ll"
  # Burst-resolved SPECTRAL probes (LAST: experimental guards may trip; main cells land first).
  # Temporal analogue of the 7.2 recipe: arrival spread matched to the burst (hw 0.27 w0 ->
  # ~1e-3 T), SPP 8e6 clears the 4g^2 = 4e6 w Nyquist, margins at burst scale, NX 65 keeps the
  # cube ~8 GB. Goal: the RR line-walk (4g^2 line dropping as gamma decays) in the spectrum.
  "probe_cl|EDM_A0=2 EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_SCREEN_HW=0.27 EDM_NX=65 EDM_SPP=8000000 EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002 EDM_HARMONICS=3996000,4000000,4004000"
  "probe_ll|EDM_A0=2 EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_SCREEN_HW=0.27 EDM_NX=65 EDM_SPP=8000000 EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002 EDM_HARMONICS=3996000,4000000,4004000 EDM_SYSTEM=ll"
)
