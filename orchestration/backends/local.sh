#!/usr/bin/env bash
# LOCAL backend — run a campaign's cells on THIS machine's GPU.
#   bash orchestration/backends/local.sh orchestration/campaigns/<campaign>.sh
# Detached (survives logout):
#   setsid nohup bash orchestration/backends/local.sh orchestration/campaigns/<c>.sh \
#       > /tmp/<c>.out 2>&1 < /dev/null & echo "pid $!"
# Reads orchestration/config.env: LOCAL_BACKEND, LOCAL_PREENV, LOCAL_JL_THREADS, JULIA_CHANNEL, EDM_REPO.
set -uo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ORCH/run_cell.sh"                                   # sources config.env + defines run_cell/run_cells
CAMPAIGN_FILE="${1:?usage: local.sh <campaign.sh>}"
. "$CAMPAIGN_FILE"                                      # sets CAMPAIGN, SCRIPT, BASE, CELLS

BACKEND="${LOCAL_BACKEND:-cuda}"
# Local runs keep their cubes by default: the cube enables post-hoc analysis (new harmonic
# bins, forensics) without a GPU rerun, and the only cost is local disk — archive big ones
# to a bulk store outside runs/ (a PUBLISH_HOOK, if configured, should cap oversized
# uploads). Campaigns and the environment can still override (KEEP_CUBE=0); the cloud
# backends keep their drain-gated retention flow.
KEEP_CUBE="${KEEP_CUBE:-1}"
JL=(julia +"${JULIA_CHANNEL:-release}" --startup=no -t "${LOCAL_JL_THREADS:-auto}")
PREENV=(); [ -n "${LOCAL_PREENV:-}" ] && read -r -a PREENV <<< "$LOCAL_PREENV"
CAMP="$REPO/runs/$CAMPAIGN"
mkdir -p "$CAMP"
# Opt-in R2 drain (config.env LOCAL_DRAIN_R2=1, creds in ~/.config/edm-r2.env): one cube_drain_r2.sh per campaign
# uploads each reduced cube + .sha256 and, with LOCAL_DRAIN_DELETE=1 (default), frees it here; the archive-side
# R2 puller archives it. The drainer outlives the campaign (it idles on an empty dir) — stop it by hand when done.
if [ "$KEEP_CUBE" = 1 ] && [ "${LOCAL_DRAIN_R2:-0}" = 1 ] &&
   ! pgrep -u "$(id -un)" -f "[c]ube_drain_r2\.sh $CAMPAIGN\$" >/dev/null; then
    DRAIN_RUNS_ROOT="$REPO/runs" DRAIN_DELETE_LOCAL="${LOCAL_DRAIN_DELETE:-1}" \
        setsid nohup bash "$ORCH/cube_drain_r2.sh" "$CAMPAIGN" >> "$REPO/runs/drain_r2_$CAMPAIGN.log" 2>&1 < /dev/null &
    echo "[local] R2 drainer started for $CAMPAIGN (log: runs/drain_r2_$CAMPAIGN.log)"
fi
echo "[local] campaign=$CAMPAIGN backend=$BACKEND cells=${#CELLS[@]} threads=${LOCAL_JL_THREADS:-auto} -> $CAMP"
run_cells
echo "[local] $CAMPAIGN DONE ($(date -u +%FT%TZ))"
