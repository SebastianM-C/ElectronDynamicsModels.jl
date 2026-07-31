# campaigns/rest_departure.sh — inverse-Thomson γ→1⁺ ladder: departure from the rest case.
# Reference config: bunched_resolved run b35964ec (γ=10, a0=0.3, LG p=2 m=−2, screen ±0.4 w₀
# @401²) — but that screen was sized for the γ=10 boosted cone; near rest the pattern fills
# the PRODUCTION direct-scattering frame, so the screen is ±25 w₀ at 601² (finer pitch than
# production 400², directly comparable to every thomson chip). Screen size is VRAM-free —
# only Nx² counts; earlier ±1.2 w₀ variants were a boosted-cone-anchored over-zoom.
# Physics: backscatter line at ω′/ω₀ = (1+β)/(1−β), β=√(1−γ⁻²) ⇒ shifts vs rest of
#   +2.9% / +9.4% / +13.5% / +22.1% / +32.7% for ε = 1e-4 / 1e-3 / 2e-3 / 5e-3 / 1e-2 (γ = 1+ε).
# Spectral resolution ≈ 1/(Ns/SPP = 375 periods) ≈ 0.27% ⇒ all rungs well separated; the
# ladder can extend to ε ~ 1e-5 (+0.9%) before resolution matters.
# EDM_GAMMA_EPS (γ = 1+ε, exact gamma_eps manifest key) landed on main (PR #68); the dashboard
# groups by the gamma_eps axis (dashboard PR #72).
# γ≈1 conventions follow gamma1_crosscheck (tspan/window/tolerances). Screen side: main's
# default = the backscatter side (same as the reference run — it resolved the n0 line there);
# EDM_SCREEN_ZSIGN is NOT on main yet (gamma1-crosscheck branch only), so the transmission-side
# control cell must wait for that merge or run locally from that branch.
# VRAM/RAM sizing (MI300X 192 GB, ~172 GiB after the 90% guard; total mode = 48·Ns·Nx² B):
#   601²/Ns6000 → cube 97 GiB, reduce 145 GiB (direct read) — REDUCE_OVERLAP-safe at 283 GiB RAM;
#   769²/Ns6000 → cube 159 GiB max (pixel pitch ≈ the reference at ±0.8 w₀) — fits VRAM but
#   NO overlap (reduce+next read 397 GiB > RAM) and needs RUNPOD_DISK_GB≥250.
CAMPAIGN=rest_departure
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=25
  EDM_DIRECT_READ=1
)
CELLS=(
  "rest|EDM_GAMMA_EPS=0"
  "e1em4|EDM_GAMMA_EPS=1e-4"
  "e1em3|EDM_GAMMA_EPS=1e-3"
  "e2em3|EDM_GAMMA_EPS=2e-3"
  "e5em3|EDM_GAMMA_EPS=5e-3"
  "e1em2|EDM_GAMMA_EPS=1e-2"
)
