# extent_a20_diag (branch campaigns/emission-ladder) — does the emission ladder extend to a₀ = 20? Small 64² cells,
# run right after the a₀ = 10 sizing verdict: (i) aliasing of h1–h4 at SPP 16/32/64 on the Newton kernel (its light-cone
# solve is independent of the sample spacing; fixed 375-period window, Ns = 375·SPP), like alias_spp_diag at a₀ = 5, 10;
# (ii) RK4 n_substeps 1 and 4 against that Newton reference at SPP 16 — the ladder's kernel choice, unproven past a₀ = 10.
# Each cell also gets the shelf sentinel ([window].ok, Coulomb level, tail zeros) through the ladder's post-reduce hook.
CAMPAIGN=extent_a20_diag
SCRIPT=scripts/thomson_scattering.jl
KEEP_CUBE=0
REDUCE_OVERLAP=1
. "$(dirname "${BASH_SOURCE[0]}")/emission_ladder/post_reduce.sh"
REDUCE_HOOK='ladder_reduce "$uuid"'
SWEEP_AXES=samples_per_period,accumulation_alg,n_substeps
BASE=(
  EDM_NX=64 EDM_N=400 EDM_FIELD_MODE=total EDM_A0=20
  EDM_RELTOL=1e-12 EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
)
CELLS=(
  "newton_spp16|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3 EDM_SPP=16 EDM_NSAMPLES=6000"
  "newton_spp32|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3 EDM_SPP=32 EDM_NSAMPLES=12000"
  "newton_spp64|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3 EDM_SPP=64 EDM_NSAMPLES=24000"
  "rk4ns1_spp16|EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=1 EDM_SPP=16 EDM_NSAMPLES=6000"
  "rk4ns4_spp16|EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=4 EDM_SPP=16 EDM_NSAMPLES=6000"
)
