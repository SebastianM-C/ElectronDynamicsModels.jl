#!/usr/bin/env bash
# hotaisle_availability.sh — one Hot Aisle offerings poll → availability TSV (+ ntfy when 1×MI300X returns).
#
#   bash orchestration/hotaisle_availability.sh     # driven every 5 min by edm-hotaisle-availability.timer
#
# GET /virtual_machines/available/ is READ-ONLY — a free offerings list; polling never provisions.
# Appends one row per offering to $AVAIL_LOG (default ~/.local/share/edm-availability/availability.tsv):
#   ts_utc  model  gpu_count  price_cents_h  min_reserve_min      (price = cents per GPU-hour)
# model=none  → poll OK, list empty (capacity exhausted — distinct from watcher downtime);
# model=error → poll failed (curl / bad JSON), so error stretches stay visible in the data.
# A 1×MI300X absent→available transition (vs the previous poll) fires ntfy through the existing
# campaign creds ($NTFY_ENV): price + how long the spec was gone. Otherwise silent on failures.
# Team handle stays out of git: $HOTAISLE_TEAM, else config.env beside this script, else
# ~/.config/hotaisle/env, else ~/.config/hotaisle/team. The token reaches curl only via a -K
# header file (never argv). Render the log with orchestration/availability_report.sh.
set -euo pipefail

LOG="${AVAIL_LOG:-$HOME/.local/share/edm-availability/availability.tsv}"
TOKF="${HOTAISLE_TOKEN_FILE:-$HOME/.config/hotaisle/token}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[ -z "${HOTAISLE_TEAM:-}" ] && [ -f "$HERE/config.env" ] && . "$HERE/config.env"
# shellcheck disable=SC1091
[ -z "${HOTAISLE_TEAM:-}" ] && [ -f "$HOME/.config/hotaisle/env" ] && . "$HOME/.config/hotaisle/env"
TEAM="${HOTAISLE_TEAM:-$(cat "$HOME/.config/hotaisle/team" 2>/dev/null || true)}"
[ -n "$TEAM" ] || { echo "no team handle (HOTAISLE_TEAM / config.env / ~/.config/hotaisle/{env,team})" >&2; exit 1; }
URL="${HOTAISLE_API:-https://admin.hotaisle.app/api/teams}/$TEAM/virtual_machines/available/"

NTFY_ENV="${NTFY_ENV:-$HOME/.config/ntfy/edm-campaigns.env}"
notify() {   # notify <tags> <priority> <title> <message> — no-op without creds
    [ -f "$NTFY_ENV" ] || return 0
    # shellcheck disable=SC1090
    . "$NTFY_ENV"
    [ -n "${NTFY_URL:-}" ] || return 0
    curl -sS --max-time 10 ${NTFY_CERT:+--cert "$NTFY_CERT"} ${NTFY_KEY:+--key "$NTFY_KEY"} \
        ${NTFY_TOKEN:+-H "Authorization: Bearer $NTFY_TOKEN"} \
        -H "Tags: $1" -H "Priority: $2" -H "Title: $3" -d "$4" "$NTFY_URL" >/dev/null 2>&1 \
        || echo "[warn] ntfy post failed: $3" >&2
}
fmt_dur() {   # seconds → 47m / 3h12m / 2d5h
    local s=$1
    if [ "$s" -lt 3600 ]; then echo "$((s / 60))m"
    elif [ "$s" -lt 86400 ]; then echo "$((s / 3600))h$(((s % 3600) / 60))m"
    else echo "$((s / 86400))d$(((s % 86400) / 3600))h"; fi
}

mkdir -p "$(dirname "$LOG")"
[ -f "$LOG" ] || printf 'ts_utc\tmodel\tgpu_count\tprice_cents_h\tmin_reserve_min\n' > "$LOG"
exec 9>>"$LOG"
flock -n 9 || exit 0   # previous poll still running

# previous-poll state, read BEFORE appending
prev_up=$(awk -F'\t' 'FNR>1 { if ($1 != ts) { ts = $1; up = 0 }; if ($2 == "MI300X" && $3 == 1) up = 1 }
                      END { print up + 0 }' "$LOG")
last_up_ts=$(awk -F'\t' 'FNR>1 && $2 == "MI300X" && $3 == 1 { ts = $1 } END { print ts }' "$LOG")

umask 077
hk=$(mktemp); trap 'rm -f "$hk"' EXIT
{ printf 'header = "Authorization: Token '; tr -d '\n' < "$TOKF"; printf '"\n'; } > "$hk"

TS=$(date -u +%FT%TZ)
rows=""
if json=$(curl -fsS --max-time 25 -K "$hk" "$URL" 2>/dev/null) \
   && jq -e 'type == "array"' <<<"$json" >/dev/null 2>&1; then
    rows=$(jq -r '.[] | [(.Specs.gpus[0].model // "unknown"), (.Specs.gpus[0].count // 0),
                         (.OnDemandPrice // "-"), (.MinimumReservationMinutes // "-")] | @tsv' \
               <<<"$json" | sort)
    if [ -n "$rows" ]; then
        while IFS= read -r r; do printf '%s\t%s\n' "$TS" "$r" >&9; done <<<"$rows"
    else
        printf '%s\tnone\t-\t-\t-\n' "$TS" >&9
    fi
else
    printf '%s\terror\t-\t-\t-\n' "$TS" >&9
    exit 0
fi

now_row=$(awk -F'\t' '$1 == "MI300X" && $2 == 1 { print; exit }' <<<"$rows")
if [ -n "$now_row" ] && [ "$prev_up" != 1 ]; then
    price=$(cut -f3 <<<"$now_row")
    if [ -n "$last_up_ts" ]; then
        gap="was gone $(fmt_dur $(($(date -u -d "$TS" +%s) - $(date -u -d "$last_up_ts" +%s))))"
    else
        gap="first sighting in this log"
    fi
    notify "rocket" "high" "Hot Aisle 1×MI300X available" "${price}¢/GPU-h — ${gap}"
fi
