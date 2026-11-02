# Benchmark for the INVERSE radiation cube (animation/setup_inverse.jl): per-slice cost of
# precompute_radiation.jl with one process per GPU running concurrently (radiation_multigpu.jl),
# to price the cube variants before any long run. No cube is written (EDM_BENCH=1 exits after
# timing the mid slice twice; the 2nd run excludes kernel compile).
#
# The cost per slice scales ~linearly with frames (EDM_ANIM_NFRAMES) and electrons, and
# ~NT² only at large screens (small screens are launch-bound), so NT 128 and 64 are both
# measured. NSLICES only scales the printed full-cube estimate (1000 ≈ a γ = 1.5 box with
# dense forward slices).
#
# Run (sized for a 2-GPU box; every cell pins its GPU count):
#      VERDA_TYPES=2A100.44V VERDA_LOCATIONS=FIN-01 VERDA_SPOT=1 VERDA_BRANCH=<branch> \
#      bash orchestration/backends/verda.sh run orchestration/campaigns/radiation_cube_inverse_bench.sh
CAMPAIGN=radiation_cube_inverse_bench
SCRIPT=animation/radiation_multigpu.jl
# Nothing to reduce; drop the .reduced marker so REDUCE_OVERLAP backends don't count a failure.
REDUCE_HOOK='touch "$CAMP/${uuid}.reduced"'
# Declared as a BENCHMARK (dashboard /benchmarks: timing/diagnostics, no reduce owed) whose card
# is the device ladder: device_count = the declared-only hardware axis read off
# [gpu].device_count. Each GPU computes its OWN slice, so the work grows with the devices:
# a WEAK ladder, whose per-device efficiency says how much the concurrent processes slow
# each other. (A declared scaling only reaches whole grid cards, never OAT arms, hence the
# ladder is the declaration and the Nx / n_substeps cells ride the card's extras row.)
SWEEP_AXES="device_count"
SWEEP_BENCHMARK=1
SWEEP_SCALING=weak
SWEEP_LABEL="inverse radiation cube: per-slice cost"
BASE=(
  EDM_ANIM_GEOMETRY=inverse EDM_ANIM_GAMMA=1.5 EDM_ANIM_A0=1.0 EDM_ANIM_MODE=hero
  EDM_ANIM_NFRAMES=480 EDM_RAD_NSLICES=1000 EDM_BENCH=1
)
CELLS=(
  "nt128_x2|EDM_RAD_NT=128 EDM_RAD_GPUS=2"
  "nt128_x1|EDM_RAD_NT=128 EDM_RAD_GPUS=1"
  "nt64_x2|EDM_RAD_NT=64 EDM_RAD_GPUS=2"
  "nt128_sub4_x2|EDM_RAD_NT=128 EDM_NSUBSTEPS=4 EDM_RAD_GPUS=2"
)
