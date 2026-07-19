# campaigns/ll_probe_s.sh — S config: speckle-resolved backscatter probes at γ=100, the
# a₀-driven deep-RR corner of the (a₀, γ) envelope. Carrier-resolved AND grain-resolved:
# ω_bs = 4γ² ≈ 4e4 ω₁ makes the carrier 47× cheaper than the γ=1000 probes, refunding the
# pixels — SPP=170000 reaches 2.13 ω_bs (fundamental + first harmonic, 6% margin), and the
# 361² screen at hw = 0.464λ (0.00619 w₀) gives 2 px per 2ω_bs speckle grain, 4 px per ω_bs
# grain, ~90×90-grain field of view (window is Rmax-dominated ⇒ screen extent is time-free).
# Odd NX ⇒ a pixel exactly on axis (the bunched_resolved node-at-origin lesson).
#
# Pairs at a₀ {10, 30, 100}: ΔE/E ≈ 0.35% / 3% / 35% (the a₀=100 pair probes where the linear
# 3.5e-7·a₀²γ law must bend), line walk ≈ 280 / 2500 / ~23000 ω₁ at ~6.4 ω₁/bin; χ ≤ 0.0625
# (classical, LL valid). Asymmetric statistics: the signal scales as a₀², so the weak a₀=10
# pair gets N=4000 while a₀ {30,100} keep the production N=2000 — line position (the walk) is
# N-independent, and each pair stays common-disk internally. Extremes first.
#
# Cube: N_samples ≈ 26,700 × 6 × 361² × 8 B ≈ 155 GiB (81% of the MI300X guard; host peak
# ~1.06× post chunked-permute 5abd239). Run on a NETWORK VOLUME pod (RUNPOD_VOLUME_GB≈1400,
# shared with ll_probe_rerun) — cubes bypass the container disk; drain.sh empties the volume
# after the campaign. Bins: plateau ladder + 4γ² edge triplet + mid-gap + first-harmonic
# triplet (all ≤ Nyquist 85000 ω₁); the kept cube holds the full spectrum regardless.
CAMPAIGN=ll_probe_s
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_GAMMA=100 EDM_TSPAN_TAU=0.16 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=64
  EDM_SPP=170000 EDM_NX=361 EDM_SCREEN_HW=0.00619
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_HARMONICS=8000,16000,24000,32000,39898,39998,40098,60000,79896,79996,80096
)
CELLS=(
  "a10_cl|EDM_A0=10 EDM_N=4000"
  "a10_ll|EDM_A0=10 EDM_N=4000 EDM_SYSTEM=ll"
  "a100_cl|EDM_A0=100"
  "a100_ll|EDM_A0=100 EDM_SYSTEM=ll"
  "a30_cl|EDM_A0=30"
  "a30_ll|EDM_A0=30 EDM_SYSTEM=ll"
)
