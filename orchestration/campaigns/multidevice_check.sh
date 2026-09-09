# Multi-device reduce self-check on a ≥2-GPU host (scripts/multidevice_check.jl): one "cell", no products
# beyond its log. CAMPAIGN=smoke is exempt from the cube drainer + teardown gate by name.
CAMPAIGN=smoke
SCRIPT=scripts/multidevice_check.jl
KEEP_CUBE=0
BASE=()
CELLS=(
  "multidevice_check|CUDA_VISIBLE_DEVICES=0,1"
)
