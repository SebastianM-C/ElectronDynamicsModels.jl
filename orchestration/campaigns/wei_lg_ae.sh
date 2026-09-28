# wei_lg with the incoherent angular-energy map (PR #160), ±10 mrad (γ 392: 1/γ ≈ 2.5 mrad). CP anchor corrected to
# n₀/(1 + a²/2) = the LP value (EDM's same-intensity a₀).
. "$(dirname "${BASH_SOURCE[0]}")/wei_lg.sh"
CAMPAIGN=wei_lg_ae
BASE+=(EDM_ANGULAR_ENERGY=1 EDM_AE_THETA_MAX_MRAD=10)
CELLS=(
  "lp_lg7|EDM_LG_M=7 EDM_HARMONICS=595000,615878"
  "cp_lg7|EDM_LG_M=7 EDM_POL=circular_plus EDM_SWEEP=wei_lg_ae_cp EDM_HARMONICS=595000,615878"
)
