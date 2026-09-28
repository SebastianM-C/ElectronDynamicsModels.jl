# campaigns/wei_lg.sh — the LG ℓ = 7 collider of Wei et al. (arXiv:2503.18843) in the inverse geometry, partner of
# wei_gaussian.sh. Their focus: ring between the 27 and 41 μm diameters ⇒ ring radius ≈ 17 μm = √3.5 w₀ ⇒ w₀ = 11.36 λ at
# 800 nm; measured peak 1.5e17 W/cm² ⇒ a_peak = 0.855·0.8·√0.15 = 0.2649, converted to EDM's a0 with
# a0_from_peak(0.2649; mode = (p = 0, m = 7)) = 0.2649/1945.48 (EDM's a0 scales the underlying Gaussian E₀). Electrons
# 200 MeV (γ 392.39). Pulse 25 fs FWHM intensity ASSUMED from the driver ⇒ ωτ = 50. Head-on here (theirs is 135°);
# electron spread and divergence not modelled. LP and CP (their Fig. 4c,d). Worldline-first, token 33² screen.
# Harmonics: n₀ = (1+β)/(1−β) = 615878 and n_a = n₀/(1 + a²/2) LP, n₀/(1 + a²) CP, inside Nyquist 650000 at SPP 1.3e6.
# Narrow Ns ≈ 9800 ⇒ cube 0.48 GiB.
CAMPAIGN=wei_lg
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
SWEEP_AXES=lg_m
BASE=(
  EDM_GAMMA=392.39023671183674 EDM_A0=1.3616767107703034e-4 EDM_POL=linear
  EDM_W0_LAMBDA=11.358602781278035 EDM_LG_P=0 EDM_OMEGA_TAU=50
  EDM_TSPAN_TAU=0.04077573421315798 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=500 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=64
  EDM_SPP=1300000 EDM_NX=33 EDM_SCREEN_HW=0.02
  EDM_WINDOW_LEAD=0.002 EDM_WINDOW_TAIL=0.002
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
)
CELLS=(
  "lp_lg7|EDM_LG_M=7 EDM_HARMONICS=595000,615878"
  # CP: own named sweep (pol lives in [laser], not [config])
  "cp_lg7|EDM_LG_M=7 EDM_POL=circular_plus EDM_SWEEP=wei_lg_cp EDM_HARMONICS=575491,615878"
)
