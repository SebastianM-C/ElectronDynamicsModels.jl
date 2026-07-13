# Radiation cube for the 3D animation (animation/ env work, PR #35): exact LW
# far-field |E_far|² sampled at the animation frame times as a z-slice stack.
# NOT a thomson_scattering.jl cell: no manifest, no cube reduce — the output is
# radiation_cube_<uuid>.jls (picked up by the default product rsync; it is not
# matched by the field_*.jls exclude/delete globs).
# n_substeps = 4 validated against n = 14 (global L2 2.8e-3, worst-case slice).
# NOTE: needs the animation/ scripts on the pod — run with RUNPOD_BRANCH=animation
# until PR #35 merges.
CAMPAIGN=radiation_cube
SCRIPT=animation/precompute_radiation.jl
# No manifest/cube to reduce (the script serializes its own product); the hook
# keeps REDUCE_OVERLAP backends from running the default harmonic reducer, but
# must still drop the .reduced marker reap_reduces checks — a bare no-op gets
# counted as a failed reduce (cosmetic, but noisy).
REDUCE_HOOK='touch "$CAMP/${uuid}.reduced"'
BASE=(
  EDM_NSUBSTEPS=4
)
CELLS=(
  # hero-v2: extends the slice stack to the detector plane (+16λ — the cube
  # meets the screen with no gap) at the same ~6.7 slices/λ density, and stores
  # the signed Ex_far alongside |E_far|² (phase-striped radiation shells).
  # 480 frames × 192 slices × 128² × 2 arrays ≈ 12 GB Float32.
  "hero_v2|EDM_RAD_NT=128 EDM_RAD_NSLICES=192 EDM_RAD_ZMAX_LAMBDA=16 EDM_RAD_BATCH=16"
)
