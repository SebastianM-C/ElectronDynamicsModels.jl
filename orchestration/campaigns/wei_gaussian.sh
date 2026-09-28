# campaigns/wei_gaussian.sh — the Gaussian collider of Wei et al. (arXiv:2503.18843) in the inverse geometry, the case
# their classical Liénard–Wiechert calculation reproduces (their Fig. 4a,b), as the validation partner of the LG runs.
# Their numbers: 260 mJ, 8 μm FWHM focus ⇒ w₀ = 8/√(2 ln 2) μm = 8.49 λ at 800 nm; peak ≈ 5e18 W/cm² ⇒ a₀ ≈ 1.5 (LP;
# at equal energy EDM's circular a₀ is the same number, the LP field being √2 higher); electrons 160 MeV (γ 314.1)
# with 15–25 % spread, here as a γ ladder 128 / 160 / 192 MeV. Pulse: 25 fs FWHM intensity ASSUMED from the driver
# (the paper states the driver's duration only) ⇒ ωτ = 50. Their collision angle is 135°; the inverse script is
# head-on only (photon energy ×1/0.85 here, axially symmetric). Electron divergence (2 mrad) not modelled.
# Worldline-first like wei_feasibility: token 33² screen on axis; the products are the γ(τ) trace and the emission
# chips. Harmonics: n₀ = (1+β)/(1−β), n_a = n₀/(1 + a₀²/2) LP or n₀/(1 + a₀²) CP, and 2n_a (LP); SPP 1.3e6 keeps
# them inside Nyquist (650000); higher LP harmonics alias (token screen). Narrow Ns ≈ 7900 ⇒ cube 0.38 GiB.
CAMPAIGN=wei_gaussian
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=gamma
BASE=(
  EDM_A0=1.5 EDM_POL=linear
  EDM_W0_LAMBDA=8.493218002880191 EDM_LG_P=0 EDM_LG_M=0 EDM_OMEGA_TAU=50
  EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=500 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=64
  EDM_SPP=1300000 EDM_NX=33 EDM_SCREEN_HW=0.02
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
# TSPAN_TAU = 16/γ (±16τ lab, as ll_probe_s / wei_feasibility)
CELLS=(
  "lp_e128|EDM_GAMMA=251.49 EDM_TSPAN_TAU=0.06362088277892108 EDM_HARMONICS=119052,238105,252986"
  "lp_e160|EDM_GAMMA=314.11 EDM_TSPAN_TAU=0.05093721460513033 EDM_HARMONICS=185724,371448,394664"
  "lp_e192|EDM_GAMMA=376.73 EDM_TSPAN_TAU=0.042470213362320715 EDM_HARMONICS=267159,534319,567714"
  # CP: own named sweep (pol lives in [laser], not [config])
  "cp_e128|EDM_POL=circular_plus EDM_SWEEP=wei_gaussian_cp EDM_GAMMA=251.49 EDM_TSPAN_TAU=0.06362088277892108 EDM_HARMONICS=77842,252986"
  "cp_e160|EDM_POL=circular_plus EDM_SWEEP=wei_gaussian_cp EDM_GAMMA=314.11 EDM_TSPAN_TAU=0.05093721460513033 EDM_HARMONICS=121435,394664"
  "cp_e192|EDM_POL=circular_plus EDM_SWEEP=wei_gaussian_cp EDM_GAMMA=376.73 EDM_TSPAN_TAU=0.042470213362320715 EDM_HARMONICS=174681,567714"
)
