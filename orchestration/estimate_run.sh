#!/usr/bin/env bash
# estimate_run.sh — pre-flight memory + pipeline-balance estimate for ONE campaign cell.
#
# From a cell's grid parameters alone (no run needed) it reports:
#   • the field/potential cube size on disk,
#   • the peak RAM the backgrounded reduce needs (O_DIRECT vs plain read),
#   • whether reduce-peak + a concurrent next-cube read fits a memory limit — the REDUCE_OVERLAP
#     cgroup-OOM guard (the 2026-07-19 thrash),
#   • the consumer time (reduce + sha256 + R2 upload), and given a measured cell wall-time,
#     whether the cell is producer- or consumer-bound plus the N that rebalances it.
#     (Pipeline-balance rule: consumer time scales with cube BYTES / N-independent, producer with
#     N ⇒ raising N in a consumer-bound cell buys extra statistics at zero added wall time.)
#   • the SOLVER-side host RAM of the cell itself — the N-proportional trajectory splines
#     (~knots × 196 B per electron; the dominant term at N ≳ 10⁴) plus the one host cube the
#     streamed multi-device reduce keeps resident — and, with --devices/--vram-gb, whether the
#     per-device cube passes the solver's 90 % VRAM guard;
#   • the producer (field-phase) time from a per-device throughput anchor (--thr, in
#     electron·sample·pixel/s: MI300X ≈ 3.0e9 total-mode/RK4 2026-08; H200 ≈ 5.5e8 = MI300X/5.5,
#     the clincher + June 2×H200 ratio) divided by the device count.
#   The last line sums what ONE running cell charges the host while its predecessor's reduce
#   overlaps (splines + accumulator + reduce peak): multiply by the concurrent lanes on a
#   multi-GPU pod and compare with the pod's RAM (runpod.sh logs "pod shape" after the grab).
#
# usage:
#   estimate_run.sh --nx 361 --ns 26677 --n 2000 --mode total \
#       [--direct 0|1] [--ram-gb 263] [--bw-mbs 162] [--cell-min 29] [--reduce-factor 1.5] \
#       [--knots 24000] [--devices 1] [--vram-gb 192] [--thr 3.0e9]
#
# Throughput/limit anchors are MI300X / R2-162MB/s (2026-07); recalibrate as hardware changes.
set -euo pipefail

NX=; NY=; NS=; N=; MODE=total; DIRECT=0; RAMGB=263; BW=162; CELLMIN=; RF=1.5
KNOTS=24000; DEV=1; VRAMGB=; THR=
while [ $# -gt 0 ]; do case "$1" in
  --nx) NX=$2; shift 2;; --ny) NY=$2; shift 2;; --ns) NS=$2; shift 2;;
  --n) N=$2; shift 2;; --mode) MODE=$2; shift 2;; --direct) DIRECT=$2; shift 2;;
  --ram-gb) RAMGB=$2; shift 2;; --bw-mbs) BW=$2; shift 2;; --cell-min) CELLMIN=$2; shift 2;;
  --reduce-factor) RF=$2; shift 2;;
  --knots) KNOTS=$2; shift 2;; --devices) DEV=$2; shift 2;; --vram-gb) VRAMGB=$2; shift 2;; --thr) THR=$2; shift 2;;
  -h|--help) sed -n '2,18p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 64;;
esac; done
: "${NX:?--nx required}"; : "${NS:?--ns required}"; NY=${NY:-$NX}

awk -v nx="$NX" -v ny="$NY" -v ns="$NS" -v n="${N:-0}" -v mode="$MODE" \
    -v direct="$DIRECT" -v ramgb="$RAMGB" -v bw="$BW" -v cellmin="${CELLMIN:-0}" -v rf="$RF" \
    -v knots="$KNOTS" -v dev="$DEV" -v vramgb="${VRAMGB:-0}" -v thr="${THR:-0}" 'BEGIN {
  GiB = 1024^3; F64 = 8; MB = 1000^2
  SHA_MBS = 420          # sha256 throughput, measured
  RED_MBS = 90           # reduce throughput on the MI300X node (155 GiB harmonic reduce ~30 min); CPU-bound, recalibrate per box

  # Cube: comp = Float64s stored per (screen-pixel × time-sample) by mode —
  #   potential (Aμ)=4 · field total (E,B)=6 · field split (E,B,E_far,B_far)=12.
  comp = (mode == "split" ? 12 : (mode == "potential" ? 4 : 6))
  cube = ns * nx * ny * comp * F64

  # Reduce footprint: harmonic_products deserializes the whole cube (= cube) plus spectral/working
  # buffers — rf × cube (default 1.5; measured ~1.35, padded) with EDM_DIRECT_READ=1. WITHOUT
  # O_DIRECT the plain read ALSO charges ~1× cube of page cache to the cgroup (the O_DIRECT doubling).
  reduce_peak = cube * rf + (direct == 1 ? 0 : cube)

  # consumer stage times (s) — all scale with cube bytes, N-independent
  reduce_s = cube/MB / RED_MBS
  sha_s    = cube/MB / SHA_MBS
  up_s     = cube/MB / bw
  consumer_s = reduce_s + sha_s + up_s

  # REDUCE_OVERLAP memory guard: reduce peak + page cache of a concurrent next-cube read
  overlap_peak = reduce_peak + cube

  printf "cube (%s, %dx%d x %d, %d comp):        %8.1f GiB\n", mode, nx, ny, ns, comp, cube/GiB
  printf "reduce peak RAM (direct=%d):              %8.1f GiB\n", direct, reduce_peak/GiB
  printf "overlap peak (reduce + next read):       %8.1f GiB  vs %d GiB limit  ->  %s\n", \
         overlap_peak/GiB, ramgb, (overlap_peak/GiB > ramgb ? "OOM RISK" : "ok")
  printf "consumer time (reduce %.0f + sha %.0f + up %.0f):  %.0f min\n", \
         reduce_s/60, sha_s/60, up_s/60, consumer_s/60
  if (cellmin > 0) {
    printf "cell wall-time %.0f min  ->  %s-bound\n", cellmin, (cellmin*60 < consumer_s ? "CONSUMER" : "producer")
    if (cellmin*60 < consumer_s && n > 0)
      printf "  rebalance: raise N %d -> ~%d  (free extra statistics, ~same wall-time)\n", \
             n, int(n * consumer_s / (cellmin*60))
  }

  # ── solver side: host RAM of the running cell, per-device VRAM, producer time ──
  # Splines: uniform-saveat CubicSpline knots per trajectory (24k = the TSPAN_TAU·γ=16 campaign
  # convention at EDM_INTERP_SAVEAT=16; scripts/inverse_thomson_scattering.jl RAM note) × ~196 B
  # (8 state + 4 acceleration Float64 per knot, plus the spline coefficient arrays). N-proportional.
  SPL_B = 196
  splines = n * knots * SPL_B
  # Host accumulator: the streamed sharded reduce keeps ONE host cube resident whatever the device
  # count (src/gpu/multidevice.jl); the un-sharded path downloads the same one cube. Split mode
  # returns E,B,E_far,B_far = the full 12-comp cube; total returns the 6-comp cube.
  accum = cube
  cell_host = splines + accum
  printf "solver host RAM: splines %.1f GiB (N=%d × %d knots) + host cube %.1f GiB = %.1f GiB\n", \
         splines/GiB, n, knots, accum/GiB, cell_host/GiB
  # Per-device VRAM: the kernel device buffers = the cube components (total: E,B; split:
  # E,B,E_far,B_far — the same bytes the cube serializes); each sharded device holds a full set.
  if (vramgb > 0)
    printf "per-device VRAM: %.1f GiB cube vs 90%% of %d GiB = %.1f GiB  ->  %s (x%d devices)\n", \
           cube/GiB, vramgb, 0.9*vramgb, (cube > 0.9*vramgb*GiB ? "REFUSED by the solver guard" : "ok"), dev
  if (thr > 0 && n > 0) {
    field_s = n * ns * nx * ny / (thr * dev)
    printf "producer (field) time at %.2e e·s·px/s/device × %d device(s):  %.0f min\n", thr, dev, field_s/60
    if (field_s < consumer_s) printf "  CONSUMER-bound: reduce+sha+upload (%.0f min) outlasts the field phase\n", consumer_s/60
  }
  printf "one running cell while the previous reduce overlaps: %.1f GiB host  (× concurrent lanes vs pod RAM)\n", \
         (cell_host + reduce_peak)/GiB
}'
