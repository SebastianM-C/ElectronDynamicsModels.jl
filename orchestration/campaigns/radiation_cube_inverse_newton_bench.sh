# Per-slice cost of the INVERSE radiation cube with the Newton light-cone kernel, against the RK4
# baseline of radiation_cube_inverse_bench.sh, across sample chunks and Newton iterations. The cube
# is launch-starved at animation screens (128² = 16k threads per launch), so the chunk arm finds
# where the device saturates; the iteration arm prices J = 3, 4 in case J = 2 does not converge
# at these frame spacings. No cube is written (EDM_BENCH=1).
#
# Run: VERDA_TYPES=1H200.141S.44V VERDA_LOCATIONS=FIN-02 VERDA_BRANCH=<branch> \
#      bash orchestration/backends/verda.sh run orchestration/campaigns/radiation_cube_inverse_newton_bench.sh
CAMPAIGN=radiation_cube_inverse_newton_bench
SCRIPT=animation/radiation_multigpu.jl
REDUCE_HOOK='touch "$CAMP/${uuid}.reduced"'
SWEEP_AXES="accumulation_alg newton_iters sample_chunks"
SWEEP_DESIGN=oat
SWEEP_BENCHMARK=1
SWEEP_LABEL="inverse radiation cube: Newton vs RK4 per-slice cost"
BASE=(
  EDM_ANIM_GEOMETRY=inverse EDM_ANIM_GAMMA=1.5 EDM_ANIM_A0=1.0 EDM_ANIM_MODE=hero
  EDM_ANIM_NFRAMES=480 EDM_RAD_NSLICES=1000 EDM_RAD_NT=128 EDM_BENCH=1 EDM_RAD_GPUS=1
)
CELLS=(
  "rk4_s14_c1|EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=14"
  "rk4_s14_c2|EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=14 EDM_SAMPLE_CHUNKS=2"
  "newton_n2_c1|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2"
  "newton_n2_c4|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=4"
  "newton_n2_c8|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=8"
  "newton_n2_c16|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=16"
  "newton_n3_c8|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3 EDM_SAMPLE_CHUNKS=8"
  "newton_n4_c8|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=4 EDM_SAMPLE_CHUNKS=8"
)
