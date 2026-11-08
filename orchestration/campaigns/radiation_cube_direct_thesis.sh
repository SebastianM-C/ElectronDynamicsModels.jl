# The direct ("electrons at rest") cube in the thesis mode (p = 2, m = −2, circular_minus), with the thesis disk:
# 2000 electrons out to 3.25 w₀ in a ±3.5 w₀ box, so the disk covers all three rings (the outer one peaks at √5 w₀).
# Otherwise the hero_v2 direct cube of radiation_cube.sh: 480 frames, 192 slices to the screen at +16λ, signed Ex_far
# and |E_far|², n_substeps 4; 224² keeps ~7 px per λ across the wider box. About 37 GB in 24 batches of 8 slices (3 per GPU on 8 GPUs), drained to R2.
CAMPAIGN=radiation_cube_direct_thesis
SCRIPT=animation/radiation_multigpu.jl
KEEP_CUBE=1
REDUCE_HOOK='touch "$CAMP/${uuid}.reduced"'
BASE=(
  EDM_ANIM_GEOMETRY=direct EDM_ANIM_MODE=thesis EDM_ANIM_N=2000 EDM_ANIM_RMAX=3.25 EDM_ANIM_HALF_W=3.5
  EDM_RAD_NT=224 EDM_RAD_NSLICES=192 EDM_RAD_ZMAX_LAMBDA=16 EDM_RAD_BATCH=8 EDM_NSUBSTEPS=4
)
CELLS=(
  "direct_thesis|"
)
