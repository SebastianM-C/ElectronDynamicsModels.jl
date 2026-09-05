# campaigns/mgpu_bench_common.sh — shared definitions for the multi-GPU scaling benchmark
# (sourced by every mgpu_bench_*.sh lane file; not a campaign by itself).
#
# Purpose: GPU-count scaling curves for an article (weak + strong scaling of the electron-sharded
# accumulate_field on an 8×H200 pod, plus a 1×H100 throughput point) where EVERY benchmark cell
# is also a physics point:
#   • weak scaling  = N ladder 2000→16000 (N ∝ devices) on the e2p5e0 bridge rung, EXACTLY the
#     published framing (601², ±7 w₀) — tests the lattice-alias reading of the halo: the clean
#     radius r_alias ≈ 3.09·√N/n₀ w₀ must grow from 2.9 (N=2000) to 8.3 w₀ (N=16000, off-screen).
#   • strong scaling = fixed N=16000 on the γ=5 rung (n₀≈98) with the screen corrected to the
#     Fresnel-regime image size (±3 w₀ at 401²; the bridge's 25/γ rule undersized it) — at
#     N=16000 the clean radius is 4 w₀, so the LG image should appear intact where the
#     N=2000 map showed only alias speckle.
#   • capacity = 1101² at the weak framing: a ~90 GiB cube (fits H200's 141 GB, refused by
#     H100's 80 GB under the solver's 90% guard).
# Lanes: each mgpu_bench_l*.sh is one sequential lane pinned to a device subset through the
# per-cell CUDA_VISIBLE_DEVICES override; runpod.sh launches the phase-1 files concurrently
# (RUNPOD_GPU_COUNT=8). All lanes share CAMPAIGN=mgpu_bench; cells name their sweep with
# EDM_SWEEP. See orchestration/README.md § "Multi-GPU lanes" for the launch sequence.
CAMPAIGN=mgpu_bench
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1          # cubes archived by the R2 drainer (the N=16000 maps are the physics product)
BASE=(
  EDM_NX=601 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.3
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
)
# e2p5e0 as published (rest_departure_bridge_refix; harmonics = the PR #84 {n_ps, n_th, 2n_ps, 2n_th} set)
WEAK="EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=7.0 EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576 EDM_SWEEP=mgpu_weak"
# γ=5 rung (rest_departure_bridge e4e0) at the corrected screen; harmonics = that run's {n_ps, n_th, 2n_ps, 2n_th}
STRONG="EDM_GAMMA_EPS=4 EDM_SPP=512 EDM_TSPAN_TAU=3.2 EDM_SCREEN_HW=3.0 EDM_NX=401 EDM_N=16000 EDM_HARMONICS=97.1395,98,193.6813,196 EDM_SWEEP=mgpu_strong"
CAPACITY="EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=7.0 EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576 EDM_NX=1101 EDM_N=2000 EDM_SWEEP=mgpu_capacity"
