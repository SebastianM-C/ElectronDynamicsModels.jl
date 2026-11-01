# Benchmark for the INVERSE radiation cube (animation/setup_inverse.jl): per-slice cost of
# precompute_radiation.jl with one process per GPU running concurrently (bench_multigpu.jl),
# to price the cube variants before any long run. No cube is written (EDM_BENCH=1 exits after
# timing the mid slice twice; the 2nd run excludes kernel compile).
#
# The cost per slice scales ~linearly with frames (EDM_ANIM_NFRAMES) and electrons, and
# ~NT² only at large screens (small screens are launch-bound), so NT 128 and 64 are both
# measured. NSLICES only scales the printed full-cube estimate (1000 ≈ a γ = 1.5 box with
# dense forward slices).
#
# Run: VERDA_TYPES=4A6000.40V VERDA_LOCATIONS=FIN-01 VERDA_BRANCH=<branch> \
#      bash orchestration/backends/verda.sh run orchestration/campaigns/radiation_cube_inverse_bench.sh
CAMPAIGN=radiation_cube_inverse_bench
SCRIPT=animation/bench_multigpu.jl
# Nothing to reduce; drop the .reduced marker so REDUCE_OVERLAP backends don't count a failure.
REDUCE_HOOK='touch "$CAMP/${uuid}.reduced"'
BASE=(
  EDM_ANIM_GEOMETRY=inverse EDM_ANIM_GAMMA=1.5 EDM_ANIM_A0=1.0 EDM_ANIM_MODE=hero
  EDM_ANIM_NFRAMES=480 EDM_RAD_NSLICES=1000
)
CELLS=(
  "nt128_x4|EDM_RAD_NT=128"
  "nt128_x1|EDM_RAD_NT=128 EDM_BENCH_GPUS=1"
  "nt64_x4|EDM_RAD_NT=64"
  "nt128_sub4_x4|EDM_RAD_NT=128 EDM_NSUBSTEPS=4"
)
