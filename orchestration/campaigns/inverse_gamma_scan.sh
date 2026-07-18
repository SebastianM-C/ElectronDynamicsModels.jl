# campaigns/inverse_gamma_scan.sh — inverse-Thomson γ-ladder at a0 = 1 (first test past the
# validated γ=10, a0=0.1 point of PR #29).
#
# Design notes (why each knob moves with γ):
#   • EDM_TSPAN_TAU ∝ 1/γ — tspan is PROPER time, the interaction is Doppler-compressed to ~τ/γ;
#     1.6 / 0.32 / 0.16 keeps a constant ±16τ lab window (the 1%-envelope overlap is |t| ≲ 1.1τ,
#     so ~15× margin; the removed coast is force-free and deep inside z_R — measured inert to
#     float noise on the g10 smoke) and a γ-free ~24k saveat knots/electron at ISA=16.
#   • EDM_INTERP_SAVEAT=16 — the a0=1 quiver needs it: the measured ISA 8-vs-16 A/B on g10 hmaps
#     is 2% max/0.26% rms of peak (ISA=16 residual ~0.13% by h⁴). RAM: 24k knots ≈ 4.7 MB/e⁻
#     → ~47 GB at N=10⁴ (+ ~36 GiB cube at serialize) — fits the 128 GB box; ISA=16 at the old
#     ±80τ window would need ~240 GB.
#   • EDM_SPP ≳ 2·N0 — the a0=1 circular line CHIRPS across [N0/(1+a0²), N0] = [2γ², 4γ²]
#     (ponderomotive redshift under the Gaussian envelope), so the default N0±1 bins are wrong;
#     EDM_HARMONICS below spans the band explicitly (N0 = 398 / 9998 / 39998; g10 adds the
#     2nd-harmonic band mid at 597 — h2 is an off-axis ring for circular pol).
#   • EDM_NX·hw shrink with γ (cube = 2·N_s·3·Nx²·8 B, N_s ∝ SPP ∝ γ² → guard < 43.2 GiB on the
#     W7900): 41.9 / 32.7 / 37.6 GiB. Finest transverse feature ≈ √(λ_n·Z) = 224λ/γ; g50/g100 are
#     vortex-core crops at ~7 / ~4 px per feature (g10 reference ≈ 12).
#   • EDM_WINDOW_LEAD/TAIL 0.15λ on g50/g100 — the fixed margins are the γ-free part of the
#     window and dominate N_samples at high γ. g10 uses 0.3λ (still ~10× the measured arrival
#     offset): the device reports 45.0 GiB total, so the 0.5λ-margin 41.9 GiB cube trips the 90%
#     guard — 0.3λ → N_s=5035 → 36.0 GiB, keeping the full Nx=400 reference map.
# γ=1000 deliberately excluded: needs the light-front residual kernel first (εZ floor ≈ 3.6
# samples at the SPP it requires — see the a0-gamma-envelope report).
CAMPAIGN=inverse_gamma_scan
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_A0=1 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=10000 EDM_NSUBSTEPS=1
  EDM_INTERP_SAVEAT=16                 # uniform proper-time knots; ISA 8-vs-16 A/B at a0=1: 2% max map error at 8
)
CELLS=(
  "g10|EDM_GAMMA=10 EDM_TSPAN_TAU=1.6 EDM_SPP=2048 EDM_SCREEN_HW=5 EDM_NX=400 EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3 EDM_HARMONICS=199,299,398,597"
  "g50|EDM_GAMMA=50 EDM_TSPAN_TAU=0.32 EDM_SPP=20480 EDM_SCREEN_HW=1 EDM_NX=240 EDM_WINDOW_LEAD=0.15 EDM_WINDOW_TAIL=0.15 EDM_HARMONICS=4999,6665,8332,9998"
  "g100|EDM_GAMMA=100 EDM_TSPAN_TAU=0.16 EDM_SPP=81920 EDM_SCREEN_HW=0.5 EDM_NX=140 EDM_WINDOW_LEAD=0.15 EDM_WINDOW_TAIL=0.15 EDM_HARMONICS=19999,26665,33332,39998"
)
