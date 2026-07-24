# campaigns/newton_a0_high.sh — GPUKernelNewton high-a0 (nonlinear Thomson) half of the sweep.
# n_iters=3: strongly-relativistic forward scattering is where the warm start needs the extra
# correction (kernel_newton.jl header / newton_lightcone report).
# Sampling at high a0: the arrival-time window is NSAMPLES/SPP laser periods — keep ≥ ~380 to
# cover the ±8τ pulse crossing (the light-front crossing time is a0-independent, so the window
# LENGTH is fine; it's the Nyquist harmonic = SPP/2 that high a0 outruns). Raising SPP at fixed
# window ⇒ NSAMPLES = 380·SPP, and cell cost scales ∝ NSAMPLES (~56 min at 6000 on MI300X).
CAMPAIGN=newton_a0_high
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=400 EDM_FIELD_MODE=total
  EDM_N=10000 EDM_RELTOL=1e-12
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the floor fix)
  EDM_POL=circular_minus
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2: corpus continuity with pre-PR#62 runs (arbitrary since #62 unified the φ₀ convention)
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3
  EDM_NSAMPLES=12160 EDM_SPP=32         # UNIFORM across cells (Nyquist h16, 380-period window):
                                        # keeps the a0 sweep 1-D on the dashboard and the maps
                                        # spectrally comparable; check a2's power spectrum before
                                        # a5/a10 run — if content nears Nyquist, requeue those at
                                        # SPP=48/NSAMPLES=18240 (VRAM caps NSAMPLES ≲ 22k at NX=400)
)
CELLS=(
    "a2em1|EDM_A0=0.2"
    "a5em1|EDM_A0=0.5"
    "a1|EDM_A0=1"
    "a2|EDM_A0=2"
    "a5|EDM_A0=5"
    "a10|EDM_A0=10"
)
