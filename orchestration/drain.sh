#!/usr/bin/env bash
# DRAIN — pull a campaign's reduced products (and optionally the big field cubes) from a RunPod
# network VOLUME to THIS machine over the S3 API. Data-plane only: needs the [runpod] S3 profile in
# ~/.aws + awscli — NOT the RunPod API token, a pod, or SSH — so it runs from ANY configured box
# (poincare / workstation / VPS). The volume persists after teardown, so drain any time, repeatedly.
#
#   bash orchestration/drain.sh <campaign> [dest] [--cubes]
#     --cubes   also download field_*.jls cubes (present only if the run used KEEP_CUBE=1)
#
# config.env: RUNPOD_DC (→ S3 endpoint), RUNPOD_VOLUME_ID (optional; auto-resolved if exactly one),
#             RUNPOD_S3_PROFILE (default 'runpod'), RUNPOD_OUT (default dest). Creds: ~/.aws/credentials.
# NOTE: use s3api get-object, not `aws s3 cp` — the latter's HeadObject precheck 403s on RunPod's S3.
set -Eeuo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ORCH/config.env" ] && . "$ORCH/config.env"

CAMPAIGN="${1:?usage: drain.sh <campaign> [dest] [--cubes]}"; shift
DEST="${RUNPOD_OUT:-$HOME/campaign_out}"; CUBES=0
for a in "$@"; do case "$a" in --cubes|--all) CUBES=1 ;; *) DEST="$a" ;; esac; done

DC="${RUNPOD_DC:-EU-RO-1}"
ENDPOINT="https://s3api-${DC,,}.runpod.io/"
PROFILE="${RUNPOD_S3_PROFILE:-runpod}"
AWS="${AWS_BIN:-aws}"; command -v "$AWS" >/dev/null 2>&1 || AWS="$HOME/.venvs/awscli/bin/aws"
S3() { "$AWS" --profile "$PROFILE" --region "$DC" --endpoint-url "$ENDPOINT" "$@"; }

# Volume id: explicit config wins; else auto-resolve via ListBuckets (bucket name == volume id).
VOL="${RUNPOD_VOLUME_ID:-}"
if [ -z "$VOL" ]; then
    mapfile -t _v < <(S3 s3api list-buckets --query 'Buckets[].Name' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d')
    case "${#_v[@]}" in
        1) VOL="${_v[0]}"; echo "[drain] auto-resolved volume: $VOL" ;;
        0) echo "[drain] no volume auto-found (ListBuckets empty/unsupported) — set RUNPOD_VOLUME_ID" >&2; exit 1 ;;
        *) echo "[drain] ${#_v[@]} volumes — set RUNPOD_VOLUME_ID to pick one: ${_v[*]}" >&2; exit 1 ;;
    esac
fi

PREFIX="runs/$CAMPAIGN/"
echo "[drain] s3://$VOL/$PREFIX → $DEST/$CAMPAIGN  (cubes=$CUBES, $ENDPOINT)"
mkdir -p "$DEST/$CAMPAIGN"; got=0
while read -r key; do
    [ -n "$key" ] || continue
    rel="${key#"$PREFIX"}"
    [ "$CUBES" -eq 1 ] || case "$(basename "$rel")" in field_*.jls) continue ;; esac
    mkdir -p "$DEST/$CAMPAIGN/$(dirname "$rel")"
    echo "  ← $rel"
    S3 s3api get-object --bucket "$VOL" --key "$key" "$DEST/$CAMPAIGN/$rel" >/dev/null && got=$((got+1))
done < <(S3 s3 ls "s3://$VOL/$PREFIX" --recursive | awk '{print $4}')
echo "[drain] done: $got file(s) → $DEST/$CAMPAIGN"
