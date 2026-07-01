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
JL=(julia +"${JULIA_CHANNEL:-release}" --startup=no -t "${LOCAL_JL_THREADS:-auto}")
PREENV=(); [ -n "${LOCAL_PREENV:-}" ] && read -r -a PREENV <<< "$LOCAL_PREENV"
CAMP="$REPO/runs/$CAMPAIGN"
mkdir -p "$CAMP"
echo "[local] campaign=$CAMPAIGN backend=$BACKEND cells=${#CELLS[@]} threads=${LOCAL_JL_THREADS:-auto} -> $CAMP"
run_cells
echo "[local] $CAMPAIGN DONE ($(date -u +%FT%TZ))"
