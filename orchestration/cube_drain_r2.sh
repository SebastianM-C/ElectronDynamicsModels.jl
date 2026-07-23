#!/usr/bin/env bash
# cube_drain_r2.sh — R2 (object storage) successor to cube_drain.sh. Run ON a cloud VM with
# KEEP_CUBE=1: pushes reduced cubes to a Cloudflare R2 bucket with multipart-parallel uploads.
#
# WHY R2 over the rsync store: every ssh-based path from a GPU VM is pinned to ~12–16 MB/s per
# stream (OpenSSH's ~2 MB channel window × 116 ms RTT — measured 2026-07-17, provider-agnostic),
# so an 86 GB cube costs ~2.4 h and campaigns pay an upload tail after compute ends. Multipart
# HTTPS sidesteps the window entirely: 16 concurrent 128 MB parts exercise the real NIC.
# R2 specifics: zero egress fees (cubes leave the buffer again!), ~$0.015/GB-month prorated
# (a 1 TB campaign parked 3 days ≈ $1.50), S3-compatible API.
#
# Integrity: a <cube>.sha256 sidecar object is uploaded alongside; the puller verifies AFTER
# download and only then frees the bucket slot — end-to-end VM-disk→archive verification,
# replacing the old full re-read verify pass (~30 min/cube).
#
# Same eligibility contract as cube_drain.sh: a cube drains once its <uuid>.reduced marker
# exists; .drained_<basename> sentinels make the two drainers interchangeable.
#
# Setup (once per account — user steps in the Cloudflare dashboard):
#   1. R2 → create bucket (default name: simulation-storage); add a lifecycle rule (e.g. delete after
#      14 days) as a cost safety-net for forgotten objects.
#   2. R2 API token scoped to THAT bucket, Object Read & Write → gives access key id + secret.
#   3. Write ~/.config/edm-r2.env (the warm step ships it to VMs like the depot key):
#        export RCLONE_CONFIG_R2_TYPE=s3
#        export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
#        export RCLONE_CONFIG_R2_ACCESS_KEY_ID=...
#        export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=...
#        export RCLONE_CONFIG_R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
#   4. rclone on the VM: curl https://rclone.org/install.sh | sudo bash   (or unzip the static
#      binary into ~/bin — no root needed).
#
# Auto-start: the hotaisle/runpod backends copy this script to the VM's $HOME (out of the repo
# clone, so a branch sync can't yank it mid-run) and nohup it when the campaign sets KEEP_CUBE=1.
#
# Env: CUBE_R2_ENV (default ~/.config/edm-r2.env), R2_BUCKET (default simulation-storage),
#      R2_CONCURRENCY (default 16), R2_CHUNK (default 128M), RCLONE (default rclone from PATH).
set -u
PATH="$HOME/bin:$PATH"   # non-interactive shells miss ~/bin, where the ad-hoc rclone lives
# Self-bootstrap a current static rclone into ~/bin when absent: fresh Hot Aisle VMs ship
# none, and some pod images ship one too old to know provider=Cloudflare (both bit
# 2026-07-23, stranding every cube of the boundary batch). Explicit RCLONE= overrides skip it.
if [ ! -x "$HOME/bin/rclone" ] && [ -z "${RCLONE:-}" ]; then
    mkdir -p "$HOME/bin" \
        && curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip \
        && python3 -m zipfile -e /tmp/rclone.zip /tmp/rclone-x \
        && mv /tmp/rclone-x/rclone-*/rclone "$HOME/bin/rclone" && chmod +x "$HOME/bin/rclone" \
        || echo "[drain-r2] rclone self-bootstrap failed — trying PATH rclone" >&2
fi
RCLONE="${RCLONE:-rclone}"
command -v "$RCLONE" >/dev/null 2>&1 || { echo "[drain-r2] $RCLONE not found (PATH=$PATH) — install rclone or set RCLONE=/path/to/rclone"; exit 1; }
ENVF="${CUBE_R2_ENV:-$HOME/.config/edm-r2.env}"
[ -f "$ENVF" ] || { echo "[drain-r2] $ENVF missing — refusing to start (fall back to cube_drain.sh)"; exit 1; }
. "$ENVF"
BUCKET="${R2_BUCKET:-simulation-storage}"
RC() { "$RCLONE" --s3-no-check-bucket --s3-upload-concurrency "${R2_CONCURRENCY:-16}" --s3-chunk-size "${R2_CHUNK:-128M}" "$@"; }
log() { echo "[drain-r2 $(date -u +%FT%TZ)] $*"; }

log "watching $HOME/EDM/runs (bucket: $BUCKET)"
while :; do
    for cube in "$HOME"/EDM/runs/*/field_*.jls; do
        [ -e "$cube" ] || continue
        dir=$(dirname "$cube"); camp=$(basename "$dir"); base=$(basename "$cube")
        uuid=${base%.jls}; uuid=${uuid##*_}
        [ "$camp" = smoke ] && continue
        [ -e "$dir/$uuid.reduced" ] || continue
        [ -e "$dir/.drained_$base" ] && continue
        log "$camp/$base → r2 ($(du -h "$cube" | cut -f1))"
        # local hash first (~3 min for 86 GB) — becomes the end-to-end reference the puller checks
        sha=$(sha256sum "$cube" | cut -d' ' -f1) || continue
        if RC copyto "$cube" "r2:$BUCKET/cubes/$camp/$uuid/$base" &&
           echo "$sha  $base" | RC rcat "r2:$BUCKET/cubes/$camp/$uuid/$base.sha256"; then
            touch "$dir/.drained_$base"; log "uploaded $base (sha256 $sha)"
            # Small-disk pods (e.g. RunPod's default 120 GB container disk): free the local copy
            # once the upload + sha sidecar are in the bucket — the puller verifies end-to-end
            # against that sha before archiving. Without this, cubes accumulate and the NEXT
            # cell's write truncates at the quota (2026-07-19: cube 2 died at 44/74 GiB, cells
            # 3-6 got 8 KiB stubs — the container quota is enforced lazily, so the writes
            # "succeeded" and the corruption only surfaced at reduce time as EOFError).
            if [ "${DRAIN_DELETE_LOCAL:-0}" = 1 ]; then
                rm -f "$cube" && log "local freed: $base (R2 copy authoritative until archived)"
            fi
        else
            log "upload failed for $base — retrying next sweep"
        fi
    done
    sleep 60
done
