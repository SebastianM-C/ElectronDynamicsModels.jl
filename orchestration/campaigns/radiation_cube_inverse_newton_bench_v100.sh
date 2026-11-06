# The Newton cube benchmark on Tesla V100 (no earlier data on this code), plus one cell with four
# concurrent GPUs to see what the host costs them. V100 needs the proprietary-driver image:
# VERDA_TYPES=4V100.20V VERDA_LOCATIONS=FIN-01 VERDA_IMAGE=24.04.cuda12.9-v100 VERDA_BRANCH=<branch> \
#   bash orchestration/backends/verda.sh run orchestration/campaigns/radiation_cube_inverse_newton_bench_v100.sh
CAMPAIGN=radiation_cube_inverse_newton_bench_v100
SCRIPT=animation/radiation_multigpu.jl
REDUCE_HOOK='touch "$CAMP/${uuid}.reduced"'
SWEEP_AXES="accumulation_alg sample_chunks device_count"
SWEEP_DESIGN=oat
SWEEP_BENCHMARK=1
SWEEP_LABEL="inverse radiation cube: Newton vs RK4 per-slice cost, V100"
BASE=(
  EDM_ANIM_GEOMETRY=inverse EDM_ANIM_GAMMA=1.5 EDM_ANIM_A0=1.0 EDM_ANIM_MODE=hero
  EDM_ANIM_NFRAMES=480 EDM_RAD_NSLICES=1000 EDM_RAD_NT=128 EDM_BENCH=1 EDM_RAD_GPUS=1
)
CELLS=(
  "rk4_s14_c1|EDM_ACCUM_ALG=rk4 EDM_NSUBSTEPS=14"
  "newton_n2_c1|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2"
  "newton_n2_c4|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=4"
  "newton_n2_c8|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=8"
  "newton_n2_c16|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=16"
  "newton_n3_c8|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=3 EDM_SAMPLE_CHUNKS=8"
  "newton_n2_c8_4gpu|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2 EDM_SAMPLE_CHUNKS=8 EDM_RAD_GPUS=4"
)
