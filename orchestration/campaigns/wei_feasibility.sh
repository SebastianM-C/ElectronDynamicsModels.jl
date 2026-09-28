# campaigns/wei_feasibility.sh — feasibility probe for the high-γ, high-ℓ inverse configuration: γ = 391, a₀ = 0.27,
# circular polarization (circular_plus first — Sebastian, to test whether that handedness matches the observed broadening;
# the circular_minus pair is staged below as a one-line addition), waist w₀ = 25 λ (20 μm at 800 nm), two laser modes:
# Gaussian (LG p = 0, m = 0) and LG ℓ = 7 (p = 0, m = 7; intensity ring at w₀√(ℓ/2) = 1.87 w₀, inside the 3.25 w₀ disc).
# Worldline-first: the purpose is the ODE side at this γ and mode (trajectory solve, γ(τ) trace) and the per-electron
# radiated energy (emission-time + emission-end chips), before an angular-energy reduction script is written. The
# screen is a token 33² on axis (±0.02 w₀), so the field products are a by-product, not a result.
# tspan ∝ 1/γ (TSPAN·γ = 16, as ll_probe_s); knots 64 per Doppler period (ll_probe_s). n₀ = 611522 (4γ²),
# n_a = n₀/(1 + a₀²) = 569971; SPP 1.3e6 puts both inside Nyquist (650000) — fundamental only.
# Narrow window with ll_probe_s-style 0.002 λ margins: Ns ≈ 27336 ⇒ cube 1.33 GiB. χ ≈ 6e-4 (classical).
# Needs the EDM_LG_P / EDM_LG_M / EDM_W0_LAMBDA knobs (this branch).
CAMPAIGN=wei_feasibility
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=lg_m
BASE=(
  EDM_GAMMA=391 EDM_A0=0.27 EDM_POL=circular_plus
  EDM_W0_LAMBDA=25 EDM_LG_P=0
  EDM_TSPAN_TAU=0.04092071611253197 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=500 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=64
  EDM_SPP=1300000 EDM_NX=33 EDM_SCREEN_HW=0.02
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_HARMONICS=569971,611522
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "gauss|EDM_LG_M=0"
  "lg7|EDM_LG_M=7"
  # handedness comparison (uncomment both): own named sweep, since pol lives in [laser], not [config]
  # "gauss_cm|EDM_LG_M=0 EDM_POL=circular_minus EDM_SWEEP=wei_feasibility_cm"
  # "lg7_cm|EDM_LG_M=7 EDM_POL=circular_minus EDM_SWEEP=wei_feasibility_cm"
)
