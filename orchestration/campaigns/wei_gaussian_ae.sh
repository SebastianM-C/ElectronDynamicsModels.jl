# wei_gaussian with the incoherent angular-energy map (EDM_ANGULAR_ENERGY, PR #160): trajectories are not cached, so θγ
# needs the cells re-run. ±15 mrad (γ 251: 1/γ ≈ 4 mrad, a₀/γ ≈ 6 mrad). CP anchors corrected to n₀/(1 + a₀²/2) (EDM's
# same-intensity a₀: |a|² = a₀²/2 for CP too), i.e. the LP values.
. "$(dirname "${BASH_SOURCE[0]}")/wei_gaussian.sh"
CAMPAIGN=wei_gaussian_ae
BASE+=(EDM_ANGULAR_ENERGY=1 EDM_AE_THETA_MAX_MRAD=15)
CELLS=(
  "lp_e128|EDM_GAMMA=251.49 EDM_TSPAN_TAU=0.06362088277892108 EDM_HARMONICS=119052,238105,252986"
  "lp_e160|EDM_GAMMA=314.11 EDM_TSPAN_TAU=0.05093721460513033 EDM_HARMONICS=185724,371448,394664"
  "lp_e192|EDM_GAMMA=376.73 EDM_TSPAN_TAU=0.042470213362320715 EDM_HARMONICS=267159,534319,567714"
  "cp_e128|EDM_POL=circular_plus EDM_SWEEP=wei_gaussian_ae_cp EDM_GAMMA=251.49 EDM_TSPAN_TAU=0.06362088277892108 EDM_HARMONICS=119052,252986"
  "cp_e160|EDM_POL=circular_plus EDM_SWEEP=wei_gaussian_ae_cp EDM_GAMMA=314.11 EDM_TSPAN_TAU=0.05093721460513033 EDM_HARMONICS=185724,394664"
  "cp_e192|EDM_POL=circular_plus EDM_SWEEP=wei_gaussian_ae_cp EDM_GAMMA=376.73 EDM_TSPAN_TAU=0.042470213362320715 EDM_HARMONICS=267159,567714"
)
