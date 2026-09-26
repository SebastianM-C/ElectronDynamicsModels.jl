# alias_spp_diag (branch campaigns/dcshelf-diag) — does SPP = 16 alias the high-a₀ spectrum into the h1–h4 maps?
# a₀ ∈ {5, 10} × SPP ∈ {16, 32, 64} at a fixed 375-period window (Ns = 375·SPP). Newton kernel: its per-sample
# light-cone solve does not depend on the sample spacing, so SPP changes only the sampling (RK4's march would change
# too). Compare the h1–h4 maps and the power spectrum near Nyquist across SPP.
CAMPAIGN=alias_spp_diag
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=a0,samples_per_period
BASE=(
  EDM_NX=64 EDM_N=400 EDM_FIELD_MODE=total
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3
  EDM_RELTOL=1e-12 EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "a5_spp16|EDM_A0=5 EDM_SPP=16 EDM_NSAMPLES=6000"
  "a5_spp32|EDM_A0=5 EDM_SPP=32 EDM_NSAMPLES=12000"
  "a5_spp64|EDM_A0=5 EDM_SPP=64 EDM_NSAMPLES=24000"
  "a10_spp16|EDM_A0=10 EDM_SPP=16 EDM_NSAMPLES=6000"
  "a10_spp32|EDM_A0=10 EDM_SPP=32 EDM_NSAMPLES=12000"
  "a10_spp64|EDM_A0=10 EDM_SPP=64 EDM_NSAMPLES=24000"
)
