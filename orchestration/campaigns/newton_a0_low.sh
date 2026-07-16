# campaigns/newton_a0_low.sh — GPUKernelNewton shakeout: low-a0 half of the a0 sweep.
# Circular_minus only (the production/lpwa-compare convention; the high-a0 half is circular too).
# Solver A/B is self-contained: the _rk4 control cell shares BASE + hardware with the Newton
# cells, so any drift is the retarded-time solver alone — no reliance on old cross-hardware runs.
# Second axis: n_iters ∈ {2, 1} — the _it1 twins test whether a single Newton correction already
# sits at the accuracy floor at low a0 (kernel_newton.jl says yes at a0=0.1 on the potential
# path; this is the field-path confirmation + same-GPU timing).
CAMPAIGN=newton_a0_low
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=400 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=split
  EDM_N=10000 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2 (published convention)
  EDM_GPU_SOLVER=newton EDM_NEWTON_ITERS=2
)
CELLS=(
  "circ_a1em3|EDM_POL=circular_minus EDM_A0=1e-3"
  "circ_a1e1|EDM_POL=circular_minus EDM_A0=0.1"
  # n_iters=1 twins (same physics cells; is one Newton correction enough at low a0?)
  "circ_a1em3_it1|EDM_POL=circular_minus EDM_A0=1e-3 EDM_NEWTON_ITERS=1"
  "circ_a1e1_it1|EDM_POL=circular_minus EDM_A0=0.1 EDM_NEWTON_ITERS=1"
  # RK4 controls — the same-hardware solver A/B baseline at both a0 decades (the 1e-3 one
  # doubles as a Newton-vs-RK4 check in the spline-floor regime). No _it1 twins for rk4:
  # n_iters doesn't exist there, they'd be duplicate runs.
  "circ_a1e1_rk4|EDM_POL=circular_minus EDM_A0=0.1 EDM_GPU_SOLVER=rk4 EDM_NSUBSTEPS=1"
  "circ_a1em3_rk4|EDM_POL=circular_minus EDM_A0=1e-3 EDM_GPU_SOLVER=rk4 EDM_NSUBSTEPS=1"
)
