# kernel_conv_diag (branch campaigns/emission-ladder) — is the RK4 accumulation kernel converged against Newton at
# the ladder's high end? lpwa.jl (e33cc1a) is RK4-only, so its n_substeps must be chosen before the LPWA side of the
# emission ladder runs. Same trajectories (numeric, uniform knots) through Newton (3 iterations) and RK4 at
# n_substeps 1 and 4, at a₀ = 1 and 10; rect reduction like the ladder. Compare the h1/h2 maps.
CAMPAIGN=kernel_conv_diag
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=0
REDUCE_OVERLAP=1
SWEEP_AXES=a0,accumulation_alg,n_substeps
BASE=(
  EDM_NX=64 EDM_N=400 EDM_FIELD_MODE=total
  EDM_NSAMPLES=6000 EDM_SPP=16
  EDM_RELTOL=1e-12 EDM_INTERP_SAVEAT=16
  EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "a1_newton|EDM_A0=1 EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3"
  "a1_rk4ns1|EDM_A0=1 EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=1"
  "a1_rk4ns4|EDM_A0=1 EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=4"
  "a10_newton|EDM_A0=10 EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3"
  "a10_rk4ns1|EDM_A0=10 EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=1"
  "a10_rk4ns4|EDM_A0=10 EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=4"
)
