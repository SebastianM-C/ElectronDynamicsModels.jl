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
# No manifest/cube to reduce (the script serializes its own product); a no-op
# hook keeps REDUCE_OVERLAP backends from running the default harmonic reducer
# against a manifest that doesn't exist.
REDUCE_HOOK='true'
BASE=(
  EDM_NSUBSTEPS=4
)
CELLS=(
  # hero-res: 128² transverse × 192 slices × 480 frames ≈ 6 GB Float32
  "hero|EDM_RAD_NT=128 EDM_RAD_NSLICES=192"
)
