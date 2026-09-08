# campaigns/xvendor_bench_common.sh — shared definitions for the cross-vendor single-device
# benchmark (sourced by every xvendor_bench_<device>.sh lane file; not a campaign by itself).
#
# Purpose: the SAME single-device cells on every card we can reach — RTX 4080 SUPER (16 GB),
# RTX 5090 (32 GB), W7900 (48 GB), H100 (80 GB), H200 (141 GB), MI300X (192 GB) — at ONE commit,
# so [timing].field, [flops] (per-slot FLOP model × the measured FP64 peak) and [gpu] telemetry
# are directly comparable: the clincher's two-vendor ratio generalized to six devices. The cells
# are the mgpu_bench strong/weak cells VERBATIM (same physics as mgpu_bench/strong_D1 and
# weak_N2000_D1 — those cubes are archived, so nothing is kept here), only the sweep ids differ:
#   • strong = γ=5 rung, 401², N=16000, total mode: 12.0 GiB cube → passes the solver's 90 %
#     VRAM guard on every card down to 16 GB (the common point; the H200 ran it at thread-fill
#     occupancy 0.59, so it slightly underfills the big cards — hence the second cell)
#   • weak   = e2p5e0 rung, 601², N=2000: 26.8 GiB cube → cards ≥ 32 GB only (the saturated
#     point H100/H200/MI300X were calibrated at; ~7 % more throughput than 401² on the H200)
# Per-device files pick CAMPAIGN (one output dir per card) and the cell subset the card fits.
# Reference times for the strong cell: H200 SXM 10287 s (mgpu_bench), MI300X ≈ 25 min (3.0e9
# e·s·px/s), W7900 ≈ 2.7 h (4.4e8 measured on image_a0_g3p5 at 401²).
# HOST RAM: the strong cell keeps N=16000 trajectory splines on the host (~75 GB) plus the 12 GiB
# cube; with the inline reduce the process peaks at ~105–110 GB (measured 2026-09-08: 104 GB on the
# RTX 5090 box, OOM-killed at 109 GB on a 128 GB desktop with a 13 GB baseline). On a ≤128 GB host
# launch the local lane with REDUCE_OVERLAP=1 so the reduce runs in its own process after the
# field process has exited. Cloud pods (251 GB) are unaffected.
. "$(dirname "${BASH_SOURCE[0]}")/mgpu_bench_common.sh"
KEEP_CUBE=0
# Later assignments win inside a cell's override list (the mgpu_bench_h100 pattern).
STRONG="$STRONG EDM_SWEEP=xvendor_strong"
WEAK="$WEAK EDM_N=2000 EDM_SWEEP=xvendor_weak"
