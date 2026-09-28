# campaigns/wei_feasibility_peak.sh — the two LG ℓ = 7 cells of wei_feasibility rerun at EQUAL PEAK FIELD with the
# Gaussian cells. EDM's LaguerreGaussLaser a0 scales the underlying Gaussian E₀, and the (p 0, m 7) ring peak is
# √(7!)·7^3.5·e^(−3.5) = 1945.48 × E₀ (at ρ = √3.5 w₀ = 1.87 w₀), so a₀ = 0.27 / 1945.48 = 1.3878e-4 puts the ring at
# the peak |E| of the a₀ = 0.27 Gaussian (Wei et al. quote one peak intensity). The as-run lg7 cells (a₀ 0.27 ⇒
# ring 1945× the Gaussian) stay in wei_feasibility. Everything else as wei_feasibility.
. "$(dirname "${BASH_SOURCE[0]}")/wei_feasibility.sh"
CAMPAIGN=wei_feasibility_peak
BASE+=(EDM_A0=1.3878292694031262e-4)
CELLS=(
  "lg7|EDM_LG_M=7"
  "lg7_cm|EDM_LG_M=7 EDM_POL=circular_minus EDM_SWEEP=wei_feasibility_peak_cm"
)
