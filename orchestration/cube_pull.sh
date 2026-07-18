#!/usr/bin/env bash
# cube_pull.sh — run on the archive machine (workstation): pull drained cubes out of the
# rsync-jailed store into the pool, verify (checksum dry-run transfers nothing), then free the
# store slot — the store is a ~1 TB BUFFER between the paid VMs and the archive, not a warehouse:
# cube_drain.sh (on the VMs) fills it while campaigns run; this empties it with no VM billing.
#
# Env: PULL_STORE (user@host of the jailed store), PULL_DEST (archive dir),
#      PULL_KEY (default ~/.ssh/id_ed25519_depot; must be jailed to the store),
#      PULL_PORT (default 22; 23 + sub-account for direct Hetzner storage-box access).
set -u
STORE="${PULL_STORE:?set PULL_STORE (user@host of the jailed store)}"
DEST="${PULL_DEST:?set PULL_DEST (archive dir on this machine)}"
KEY="${PULL_KEY:-$HOME/.ssh/id_ed25519_depot}"
PORT="${PULL_PORT:-22}"
RS() { rsync -e "ssh -p $PORT -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" "$@"; }
log() { echo "[pull $(date -u +%FT%TZ)] $*"; }
EMPTY=$(mktemp -d)
mkdir -p "$DEST"

while :; do
    # store layout (written by cube_drain.sh): cubes/<campaign>/<uuid>/field_*.jls
    RS -r --list-only "$STORE:cubes/" 2>/dev/null | awk 'NF>=5 && $NF ~ /\.jls$/ {print $NF}' |
    while IFS= read -r rel; do
        camp=${rel%%/*}; rest=${rel#*/}; uuid=${rest%%/*}; base=${rest#*/}
        [ -e "$DEST/$camp/.verified_$base" ] && continue
        mkdir -p "$DEST/$camp"
        log "$camp/$base ← store ($uuid)"
        RS -t --partial "$STORE:cubes/$rel" "$DEST/$camp/" || { log "pull failed $base — next sweep"; continue; }
        # checksum dry-run against the store copy: -i itemizes only mismatches ⇒ silence = verified
        if RS -nci -t "$DEST/$camp/$base" "$STORE:cubes/$camp/$uuid/" | grep -q "$base"; then
            log "VERIFY FAILED $base — keeping store copy, retrying next sweep"
        else
            touch "$DEST/$camp/.verified_$base"
            RS -r --delete "$EMPTY/" "$STORE:cubes/$camp/$uuid/" && log "verified + freed store slot: $camp/$base"
        fi
    done
    sleep 300
done
