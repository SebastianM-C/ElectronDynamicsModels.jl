#!/usr/bin/env bash
# run_diagnostics.sh — post-download validity diagnostics for every run of a campaign, on the
# driver box (CPU only; manifests suffice, the cubes are NOT needed):
#   • per-electron E(t) pixel traces (scripts/plot_pixel_traces.jl): are the fields sampled
#     properly — carrier + envelope resolved at the run's SPP, per electron and in the coherent
#     sum (EDM_TRACE_TOTAL=1, default; the full-N CPU sum scales with N — 8× at N=16000 vs 2000).
#   • worldlines + final-position histograms (scripts/analyze_trajectories.jl, chips mode):
#     were electrons kicked out of the disk by the interaction — transverse displacement
#     rms/max and the fraction beyond EDM_TRAJ_KICK_W0 (0.1 w₀) land in the sidecar.
# Both attach derived_* sidecars + PNGs next to each run_<uuid>.toml, so a re-publish shows them
# as chips on the run's dashboard card. Idempotent per run: skips a run whose sidecars exist.
#
#   bash orchestration/run_diagnostics.sh ~/campaign_out/<campaign> [more dirs…]
#   env: EDM_TRACE_TOTAL=0 to skip the coherent-sum overlay; EDM_COORDS=absolute|displacement
#        (default displacement — the kick is what the histograms are for); EDM_N to cap the
#        re-solved electrons; DIAG_SKIP_TRACES=1 / DIAG_SKIP_TRAJ=1 to run one of the two.
set -uo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$ORCH/.." && pwd)"
JL=(julia +"${JULIA_CHANNEL:-release}" --startup=no -t "${DIAG_JL_THREADS:-auto}" --project="$REPO/scripts")
[ "$#" -ge 1 ] || { sed -n '2,18p' "$0"; exit 64; }
export EDM_COORDS="${EDM_COORDS:-displacement}"
fail=0
for dir in "$@"; do
    for m in "$dir"/run_*.toml; do
        [ -e "$m" ] || continue
        id=$(basename "$m" .toml); id=${id#run_}; id8=${id:0:8}
        if [ "${DIAG_SKIP_TRACES:-0}" != 1 ]; then
            if [ -e "$dir/derived_pixeltraces_$id8.toml" ]; then echo "[diag] traces exist: $id8"
            else echo "[diag] pixel traces ← $m"
                 ( cd "$REPO" && "${JL[@]}" scripts/plot_pixel_traces.jl "$m" ) || { echo "[diag] FAILED traces $id8"; fail=1; }
            fi
        fi
        if [ "${DIAG_SKIP_TRAJ:-0}" != 1 ]; then
            ctag=$([ "$EDM_COORDS" = displacement ] && echo disp || echo abs)
            if [ -e "$dir/derived_traj_zx_${ctag}_$id8.toml" ]; then echo "[diag] worldlines exist: $id8"
            else echo "[diag] worldlines ← $m"
                 ( cd "$REPO" && EDM_TRAJ_CHIPS=1 EDM_SOURCE_CAMPAIGN="$dir" EDM_SOURCE_RUN="$id" \
                       "${JL[@]}" scripts/analyze_trajectories.jl ) || { echo "[diag] FAILED worldlines $id8"; fail=1; }
            fi
        fi
    done
done
exit $fail
