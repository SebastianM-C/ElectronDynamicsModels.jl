# campaigns/inverse_bunched.sh — helical-prebunching pilot (the coherent counterfactual to the
# inverse_gamma_scan g10 speckle baseline; see the inverse-speckle-tomography dashboard report).
#
# Design: identical physics to the g10 cell (a0=1, γ=10, N0=398) at N=2000, with a per-electron
# longitudinal start offset Δz = (1+β)/2·[ρ²/2Z + ℓθ/2π·λ/n_b]:
#   • the ρ² term is achromatic phased-array FOCUSING — it cancels the transverse path spread
#     (the speckle source), so the on-axis coherent amplitude grows ∝ N instead of √N
#     (expected on-axis intensity gain vs `base`: ~N = 2000, i.e. +33 dB at bin 398);
#   • the ℓ term winds the array into a helix at n_b = 398 — MEASURED (W7900, 2026-07-17):
#     observed winding = 2+ℓ in this code's convention, so lm2 is the AXIAL SPOT cell (on-axis
#     metric valid; 13×rms at N=10³), l0 the |2| vortex ring at ~1λ, lp2 the |4| ring at ~2.2λ
#     (score ring cells by radial profile, NOT on-axis). Coherence is full (weight-limited) at
#     a₀ ≤ 0.3 and ~0.65× at a₀=1 — no chirp compensation needed (EDM_BUNCH_CHIRP measured
#     harmful; leave unset).
# N=2000 (not 10⁴): the coherent term needs no statistics — contrast over residual speckle is
# already ~√N·b ≈ 45× in amplitude — and kernel time scales ∝ N (~1–1.5 h/cell on MI300X-class).
#
# Run on Hot Aisle (from the driving machine):
#   HOTAISLE_BRANCH=inverse-gamma-scan bash orchestration/backends/hotaisle.sh run \
#       orchestration/campaigns/inverse_bunched.sh
# (the branch override matters: the EDM_BUNCH_* knobs exist only on inverse-gamma-scan / PR #40).
# SMOKE FIRST on the VM: prepend a cell run with EDM_N=100 EDM_NX=64 (minutes) and check
# (a) the l0 smoke's on-axis h398 amplitude ≫ base smoke's, (b) which of lm2/lp2 kills the
# winding in inverse_thomson_phasePolar*_h398. Cubes are ~36 GiB each and NOT kept (KEEP_CUBE
# default 0) — products (hmaps/PNGs/powspec/manifests) are the deliverable; nothing big to
# download alongside the other transfers.
CAMPAIGN=inverse_bunched
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_A0=1 EDM_GAMMA=10 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_TSPAN_TAU=1.6 EDM_SPP=2048 EDM_SCREEN_HW=5 EDM_NX=400
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3 EDM_HARMONICS=199,299,398,597
)
CELLS=(
  "base|"
  "l0|EDM_BUNCH_NB=398"
  "lm2|EDM_BUNCH_NB=398 EDM_BUNCH_L=-2"
  "lp2|EDM_BUNCH_NB=398 EDM_BUNCH_L=2"
)
