#!/usr/bin/env bash
# cube_pull_r2.sh — archive-machine side of the R2 cube pipeline (see cube_drain_r2.sh).
# Runs on the workstation: lists the bucket, downloads each cube with multi-thread streams,
# verifies END-TO-END against the sha256 the VM computed from its own disk, and only then
# frees the bucket slot. R2 egress is free, so failed verifies re-pull at zero cost.
#
# Run from the archive box (institute line) — residential ISPs may block neither R2 (443)
# nor anything here, unlike the storage box's port 23; R2 needs only HTTPS.
#
# Env: CUBE_R2_ENV (default ~/.config/edm-r2.env — same file as the drain side),
#      R2_BUCKET (default simulation-storage), PULL_DEST (archive dir),
#      R2_PULL_STREAMS (default 8 multi-thread download streams).
set -u
ENVF="${CUBE_R2_ENV:-$HOME/.config/edm-r2.env}"
[ -f "$ENVF" ] || { echo "[pull-r2] $ENVF missing — refusing to start"; exit 1; }
. "$ENVF"
BUCKET="${R2_BUCKET:-simulation-storage}"
DEST="${PULL_DEST:?set PULL_DEST (archive dir on this machine)}"
log() { echo "[pull-r2 $(date -u +%FT%TZ)] $*"; }
mkdir -p "$DEST"

while :; do
    rclone lsf -R --files-only "r2:$BUCKET/cubes/" 2>/dev/null | grep '\.jls$' |
    while IFS= read -r rel; do   # rel = <campaign>/<uuid>/<file>.jls
        camp=${rel%%/*}; rest=${rel#*/}; uuid=${rest%%/*}; base=${rest#*/}
        [ -e "$DEST/$camp/.verified_$base" ] && continue
        mkdir -p "$DEST/$camp"
        log "$camp/$base ← r2"
        rclone copyto --multi-thread-streams "${R2_PULL_STREAMS:-8}" \
            "r2:$BUCKET/cubes/$rel" "$DEST/$camp/$base" || { log "download failed $base — next sweep"; continue; }
        want=$(rclone cat "r2:$BUCKET/cubes/$camp/$uuid/$base.sha256" 2>/dev/null | cut -d' ' -f1)
        have=$(sha256sum "$DEST/$camp/$base" | cut -d' ' -f1)
        if [ -n "$want" ] && [ "$want" = "$have" ]; then
            touch "$DEST/$camp/.verified_$base"
            rclone purge "r2:$BUCKET/cubes/$camp/$uuid" && log "verified end-to-end + freed: $camp/$base"
        else
            log "VERIFY FAILED $base (want=${want:-none} have=$have) — keeping bucket copy"
        fi
    done
    sleep 300
done
