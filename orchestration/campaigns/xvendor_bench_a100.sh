# xvendor_bench_a100.sh — the cross-vendor pair (strong + weak cell) on an A100 PCIe 40 GB.
#
# Same cells and kernel settings as xvendor_bench_h100.sh (Newton n = 2, C = 5 from the shared
# BASE). Both cubes fit the 40 GB card (12 GiB strong, 26.8 GiB weak); the strong cell needs
# ~110 GB of host RAM for the trajectory splines. On a host with several NVIDIA cards, pin the
# device from the launch environment with CUDA_VISIBLE_DEVICES=<GPU-uuid>: CUDA enumerates
# devices fastest-first, so an index does not select the A100 next to a faster card.
. "$(dirname "${BASH_SOURCE[0]}")/xvendor_bench_common.sh"
CAMPAIGN=xvendor_bench_a100
CELLS=(
  "strong|$STRONG"
  "weak|$WEAK"
)
