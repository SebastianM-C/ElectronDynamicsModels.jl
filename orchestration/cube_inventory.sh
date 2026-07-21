#!/usr/bin/env bash
# Cube inventory for status tracking. Runs on the machine that holds the durable cube archive (the
# only one that can see it) and reports what's there, so a collector on another host can answer
# "where is cube X" / "is it safely archived?" without access to that filesystem. READ-ONLY over
# the archive: it never deletes or moves a cube.
#
# Emits one row per cube — uuid, campaign, file, bytes, verified — from the layout the pull scripts
# write (<archive>/<campaign>/field_*_<uuid>.jls + a sibling .verified_<file> marker, see
# cube_pull.sh / cube_pull_r2.sh), then optionally rsyncs that one small TSV to the collector host.
#
# Paths/hosts are machine-specific — set them in config.env (gitignored), not here:
#   INV_STORE   cube archive dir (the pull scripts' PULL_DEST)
#   INV_OUT     output TSV path (default: $INV_STORE/cube_inventory.tsv)
#   INV_PUSH    optional rsync target (user@host:/path) to ship the TSV to the collector host
set -eu

STORE="${INV_STORE:?set INV_STORE (cube archive dir = PULL_DEST from the pull scripts)}"
OUT="${INV_OUT:-$STORE/cube_inventory.tsv}"
UUID_RE='[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

# Same-dir temp so the final mv is an atomic rename — the collector never reads a half-written
# inventory (the same commit-point rule as the .reduced marker).
tmp="$(mktemp "$OUT.XXXXXX")"
host=$(hostname); now=$(date -u +%FT%TZ)
printf 'uuid\tcampaign\tfile\tbytes\tverified\thost\tscanned_at\n' > "$tmp"

# Layout: $STORE/<campaign>/field_*_<uuid>.jls (one level under STORE, per the pull scripts).
find "$STORE" -mindepth 2 -maxdepth 2 -name 'field_*.jls' -printf '%h\t%f\t%s\n' 2>/dev/null |
while IFS=$'\t' read -r dir file bytes; do
    uuid=$(printf '%s' "$file" | grep -oE "$UUID_RE" | tail -1)
    [ -n "$uuid" ] || continue                       # skip anything not carrying a uuid
    if [ -e "$dir/.verified_$file" ]; then verified=1; else verified=0; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$uuid" "$(basename "$dir")" "$file" "$bytes" "$verified" "$host" "$now" >> "$tmp"
done

mv "$tmp" "$OUT"
n=$(($(wc -l < "$OUT") - 1))
echo "[inventory] $n cubes → $OUT"

[ -n "${INV_PUSH:-}" ] && rsync -t "$OUT" "$INV_PUSH" && echo "[inventory] pushed → $INV_PUSH"
