# campaigns/ll_probe.sh — carrier-resolved backscatter probes: does the 4γ² line WALK under RR?
#
# Re-run of the crashed ll_rr_strength probe cells. Two design faults fixed: (1) the Nyquist
# guard (EDM_SPP=8000000 < 2·4004000 = 8008000 — both cells died at startup, the driver counted
# them done); (2) the as-designed 65² screen would have hit the memory guard anyway — the
# :narrow window is corner_spread-dominated (full 3.25 w₀ disk ⇒ (√2·hw + Rmax)²/2Z ≈ 0.19λ ⇒
# N_samples ≈ 1.5e6 at this SPP), so cube memory = N_samples·6·Nx²·8 B pins the screen at 33²
# (~80 GB) on one 192 GB MI300X. Physics is production-identical (full disk, same LG mode).
#
# a₀ ladder {2, 5, 10} at γ=1000, LL/classical common-disk pairs: predicted line walk
# 2δγ/γ (ΔE/E ≈ 3.5e-7·a₀²γ) ≈ 11 / 70 / 280 bins at the window's 1000 ω₁/bin resolution.
# a₀=10 sits at a₀²γ = 1e5 — past the √2 field-diff decorrelation ceiling; the spectral walk
# is the observable that should survive there. χ ≤ 0.0625 (classical regime, LL valid).
# Extremes first (a2 anchor, a10 max signal), a5 last: a budget runout costs the ladder's
# interior, not its ends. Bins: the 4γ² edge triplet + a coarse plateau ladder (the nonlinear
# redshift pushes spectral weight down toward 4γ²/(1+a₀²/2)); the kept cube holds the full
# spectrum regardless.
CAMPAIGN=ll_probe
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_GAMMA=1000 EDM_TSPAN_TAU=0.016 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=32
  EDM_SPP=8032000 EDM_NX=33 EDM_SCREEN_HW=0.27
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_HARMONICS=800000,1600000,2400000,3200000,3996000,4000000,4004000
)
CELLS=(
  "a2_cl|EDM_A0=2"
  "a2_ll|EDM_A0=2 EDM_SYSTEM=ll"
  "a10_cl|EDM_A0=10"
  "a10_ll|EDM_A0=10 EDM_SYSTEM=ll"
  "a5_cl|EDM_A0=5"
  "a5_ll|EDM_A0=5 EDM_SYSTEM=ll"
)
