# campaigns/sanity_newton.sh — Newton-kernel arm of the rest_departure sanity cross-check.
# Identical to rest_departure's rest/e1em3 cells except EDM_ACCUM_ALG=newton (GPUKernelNewton
# light-cone root solve, n_iters=2 default) — the kernel A/B lands as compare_mirror_runs
# pairs against the rk4 twins and the analytic sanity_lpwa reference (see its header for the
# full pair list). Expected: kernel A/B diffs at the numerics floor; lpwa↔numeric diffs at
# the mirror-detection floor at γ=1 and the +9.4%-line departure signature at ε=1e-3.
CAMPAIGN=sanity_newton
SCRIPT=scripts/inverse_thomson_scattering.jl
BASE=(
  EDM_NX=601 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12 EDM_ABSTOL=1.7e-9
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_TSPAN_TAU=8 EDM_WINDOW=full
  EDM_SCREEN_HW=25
  EDM_ACCUM_ALG=newton
  EDM_DIRECT_READ=1
)
CELLS=(
  "g1_newton|EDM_GAMMA_EPS=0"
  "e1em3_newton|EDM_GAMMA_EPS=1e-3"
)
