# campaigns/rest_departure_bridge_refix.sh — re-acquire the 5 clipped bridge rungs (γ = 1.5 … 3.5)
# after the burst-centering window fix (PR #82). The pre-fix :narrow window opened at Z − lead,
# treating Z as the burst START; the burst is CENTERED there, so its leading half was never
# sampled — 39/28/17/8/2.5% of the center-pixel burst energy at γ = 1.5/2/2.5/3/3.5
# (narrow-window audit 2026-08-30; dashboard known-issue "narrow-window"). The γ ≥ 4 rungs were
# clip-free (≤0.5%): they keep their archived cubes and get a re-reduce only, not a re-acquire.
#
# Reduction taper: EDM_APODIZATION=none — with the burst contained by the fixed window the bare
# rfft is unbiased (synthetic γ=2: 98.5% of the true line amplitude, leakage skirt 2e-7 of the
# line) and imprints NO radial taper mask on the maps, unlike the shared-axis Hann whose weight
# varied 0.02–0.83 across the screen. The power spectrum is un-windowed by design either way.
#
# Cells reproduce the AS-RUN rows of rest_departure_bridge cells.tsv (label ⇆ overrides,
# including the e2e0 SPP=256 + line-anchored EDM_HARMONICS refinements that superseded the
# original recipe rows). Same BASE as rest_departure_bridge otherwise.
CAMPAIGN=rest_departure_bridge_refix
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
)
CELLS=(
  "e5em1|EDM_GAMMA_EPS=0.5 EDM_SPP=32 EDM_TSPAN_TAU=10.666666666666666 EDM_SCREEN_HW=16.7 EDM_HARMONICS=6,6.8542,8,13.7084"
  "e1e0|EDM_GAMMA_EPS=1 EDM_SPP=64 EDM_TSPAN_TAU=8 EDM_SCREEN_HW=12.5 EDM_HARMONICS=13,13.9282,15,27.8564"
  "e1p5e0|EDM_GAMMA_EPS=1.5 EDM_SPP=128 EDM_TSPAN_TAU=6.4 EDM_SCREEN_HW=10 EDM_HARMONICS=22,22.9564,24,45.9128"
  "e2e0|EDM_GAMMA_EPS=2 EDM_SPP=256 EDM_TSPAN_TAU=5.333333333333333 EDM_SCREEN_HW=8.3 EDM_HARMONICS=33,33.9706,35,67.9412"
  "e2p5e0|EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=7.0 EDM_HARMONICS=46,46.9788,48,93.9576"
)
