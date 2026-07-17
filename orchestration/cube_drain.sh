#!/usr/bin/env bash
# cube_drain.sh — run ON a cloud VM: continuously push reduced field cubes to the rsync-jailed
# store, overlapping uploads with the campaign's GPU compute (needs KEEP_CUBE=1 so cubes survive
# their reduce). Local copies are KEPT (cheap on the VM's NVMe) so a bad upload is never fatal;
# the store is a staging buffer the workstation later drains at its leisure (no VM billing).
#
# A cube is eligible once its <uuid>.reduced marker exists — the reduce read the whole cube, so
# it is complete and consistent on disk. Verify = a second checksum dry-run pass transfers
# nothing; only then is the .drained sentinel written. Failures just retry next sweep.
#
# Env: DRAIN_STORE (user@host of the jailed store), DRAIN_KEY (default ~/.ssh/depot_key),
#      DRAIN_PORT (default 22; Hetzner storage-box DIRECT access = port 23 + sub-account —
#      each machine then gets its own stream instead of sharing the VPS sshfs write pipe).
set -u
STORE="${DRAIN_STORE:?set DRAIN_STORE (user@host of the jailed store)}"
KEY="${DRAIN_KEY:-$HOME/.ssh/depot_key}"
PORT="${DRAIN_PORT:-22}"
RS() { rsync -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes" "$@"; }
log() { echo "[drain $(date -u +%FT%TZ)] $*"; }

log "watching $HOME/EDM/runs (store: $STORE)"
while :; do
    for cube in "$HOME"/EDM/runs/*/field_*.jls; do
        [ -e "$cube" ] || continue
        dir=$(dirname "$cube"); camp=$(basename "$dir"); base=$(basename "$cube")
        uuid=${base%.jls}; uuid=${uuid##*_}          # field_..._<uuid>.jls
        [ "$camp" = smoke ] && continue              # smoke cubes are worthless
        [ -e "$dir/$uuid.reduced" ] || continue      # cube not yet proven complete
        [ -e "$dir/.drained_$base" ] && continue
        log "$camp/$base → store ($(du -h "$cube" | cut -f1))"
        # plain rsync's implicit mkdir is not -p: push a local skeleton first so every
        # parent of cubes/<camp>/<uuid>/ exists (harmless no-op when it already does)
        skel=$(mktemp -d); mkdir -p "$skel/cubes/$camp/$uuid"
        RS -r "$skel/cubes" "$STORE:"; rm -rf "$skel"
        if RS -t --partial "$cube" "$STORE:cubes/$camp/$uuid/"; then
            # checksum dry-run: -i itemizes ONLY files that would re-transfer ⇒ any mention = mismatch
            if RS -nci -t "$cube" "$STORE:cubes/$camp/$uuid/" | grep -q "$base"; then
                log "VERIFY FAILED $base — retrying next sweep"
            else
                touch "$dir/.drained_$base"; log "verified $base"
            fi
        else
            log "push failed for $base — retrying next sweep"
        fi
    done
    sleep 60
done
