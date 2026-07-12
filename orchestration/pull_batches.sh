#!/usr/bin/env bash
# Async batch puller: rsync completed radiation slice batches from a running
# campaign VM while later slices still compute (overlaps download with GPU
# time). Exits when the _done marker has been pulled and no new files arrive.
# Usage: pull_batches.sh <user@ip> <remote_campaign_dir> <local_dir> [interval_s]
set -euo pipefail
REMOTE="$1"; RDIR="$2"; LDIR="$3"; IVL="${4:-60}"
mkdir -p "$LDIR"
while true; do
    rsync -a --include='radiation_*' --exclude='*' --exclude='*.tmp' \
        "$REMOTE:$RDIR/" "$LDIR/" 2>/dev/null || true
    if ls "$LDIR"/radiation_*_done >/dev/null 2>&1; then
        # one final sweep, then stop
        rsync -a --include='radiation_*' --exclude='*' --exclude='*.tmp' \
            "$REMOTE:$RDIR/" "$LDIR/" 2>/dev/null || true
        echo "done marker pulled — all batches local"
        break
    fi
    sleep "$IVL"
done
