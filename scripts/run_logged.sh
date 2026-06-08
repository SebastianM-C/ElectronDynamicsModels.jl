#!/usr/bin/env bash
# Run a solver script with a stable run tag and a captured log, so the log travels with the
# run's artifacts (run_<tag>.log alongside the .jls / PNGs / run_<tag>.toml). Essential on
# ephemeral VMs, where the box and its console output vanish on teardown — so always run →
# publish (logs included) → THEN destroy.
#
# The tag is the join key: this wrapper mints it and exports EDM_RUN_TAG, the solver reads
# EDM_RUN_TAG (falling back to a fresh uuid only if unset), and names its .jls/PNGs/manifest
# with it — so log, data, and manifest all share one id. Reuse a known tag (e.g. to re-emit
# a run's outputs) by exporting EDM_RUN_TAG before calling.
#
# Usage (env knobs passed through as usual; run from the repo root):
#   EDM_OUTDIR=runs/foo EDM_A0=0.1 scripts/run_logged.sh scripts/lpwa.jl
set -uo pipefail

: "${EDM_OUTDIR:?set EDM_OUTDIR (the run output dir)}"
mkdir -p "$EDM_OUTDIR"

export EDM_RUN_TAG="${EDM_RUN_TAG:-$(uuidgen)}"
log="$EDM_OUTDIR/run_${EDM_RUN_TAG}.log"
echo "run tag $EDM_RUN_TAG → $log"

# tee so the run is watchable live AND captured; pipefail propagates Julia's exit status
# (not tee's) so a failed run is still detectable by the caller.
julia +"${JULIA_CHANNEL:-release}" --project=scripts -t auto "$@" 2>&1 | tee "$log"
