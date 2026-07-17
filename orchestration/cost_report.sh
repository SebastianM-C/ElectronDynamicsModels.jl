#!/usr/bin/env bash
# cost_report.sh — EDM cloud spend report from the persistent cost ledger + run manifests.
#
#   bash orchestration/cost_report.sh [--since ISO] [--until ISO] [--ledger FILE] \
#        [--rate-cents N] [results_dir ...]
#
# The ledger (~/.config/edm-cloud-ledger.tsv, appended by backends/hotaisle.sh and runpod.sh) is
# an append-only TSV:  ts_utc  provider  vm  event  detail  rate_cents_h  balance_usd
# events: provision (rate_cents_h = the VM's $/h in CENTS at provision — list prices drift, so the
# offering price is captured per-row), campaign_start / campaign_done / campaign_crash (detail
# carries campaign=<name> dir=<outdir>), teardown, topup (detail amount=<usd>), balance,
# rate_change (provider-wide re-rate of RUNNING VMs — Hot Aisle applies list-price rises to
# in-flight VMs, observed 2026-07-17 199→299), incident (free text, surfaced as a note).
#
# What it prints:
#   • per-campaign compute = Σ [timing].total over run_<uuid>.toml in each results dir given,
#     priced at the owning VM's time-averaged rate (attribution via campaign_* rows' dir=…)
#   • per-VM wall-clock cost (provision→teardown clipped to the window; running VMs bill through
#     --until, integrated piecewise across rate_change rows) and overhead = wall − compute.
#     Cells still IN FLIGHT have no manifest yet, so they appear as overhead until their
#     run_<uuid>.toml lands — rerun after the campaign downloads/publishes.
#   • topups, first/last balance and the ledger-implied spend for reconciliation.
# Rate for a VM missing a provision rate: live Hot Aisle API lookup (token file), else
# --rate-cents, else 0 + warning. Plain bash + GNU awk/date; jq/curl only for the API fallback.
set -euo pipefail

LEDGER="${EDM_CLOUD_LEDGER:-$HOME/.config/edm-cloud-ledger.tsv}"
SINCE="1970-01-01T00:00:00Z"
UNTIL="$(date -u +%FT%TZ)"
RATE_FALLBACK=""
DIRS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --since)      SINCE=${2:?}; shift 2 ;;
        --until)      UNTIL=${2:?}; shift 2 ;;
        --ledger)     LEDGER=${2:?}; shift 2 ;;
        --rate-cents) RATE_FALLBACK=${2:?}; shift 2 ;;
        -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
        *)            DIRS+=("$1"); shift ;;
    esac
done
[ -f "$LEDGER" ] || { echo "no ledger at $LEDGER (nothing recorded yet?)" >&2; exit 1; }
SINCE_E=$(date -u -d "$SINCE" +%s)
UNTIL_E=$(date -u -d "$UNTIL" +%s)

# Current Hot Aisle list price (cents/GPU-h) — fetched ONLY if a provision row lacks a rate.
hotaisle_list_rate() {
    local tokf="${HOTAISLE_TOKEN_FILE:-$HOME/.config/hotaisle/token}"
    [ -f "$tokf" ] || return 1
    curl -fsS --max-time 10 -H "Authorization: Token $(cat "$tokf")" \
        "https://admin.hotaisle.app/api/teams/${HOTAISLE_TEAM:-REDACTED}/virtual_machines/available/" \
      | jq -r '[.[]|select(.Specs.gpus[0].model=="MI300X" and .Specs.gpus[0].count==1)|.OnDemandPrice][0] // empty'
}
API_RATE=""
if tail -n +2 "$LEDGER" | awk -F'\t' '$4=="provision" && $2=="hotaisle" && $6==""{ex=1} END{exit !ex}'; then
    API_RATE=$(hotaisle_list_rate 2>/dev/null || true)
fi

digest_manifests() {   # M \t realdir \t name \t cells \t gpu_seconds
    local d rd
    for d in ${DIRS[@]+"${DIRS[@]}"}; do
        rd=$(realpath "$d" 2>/dev/null || echo "$d")
        if ! ls "$d"/run_*.toml >/dev/null 2>&1; then
            printf 'M\t%s\t%s\t0\t0\n' "$rd" "$(basename "$rd")"; continue
        fi
        awk -v dir="$rd" -v name="$(basename "$rd")" -F' *= *' '
            FNR==1 { insec=0 }
            /^\[/  { insec = ($0=="[timing]") }
            insec && $1=="total" { s += $2; n++; nextfile }
            END    { printf "M\t%s\t%s\t%d\t%.1f\n", dir, name, n, s+0 }' "$d"/run_*.toml
    done
}

tag_ledger() {   # L \t ts \t provider \t vm \t event \t detail \t rate \t balance
    # NOT a bash read loop: with IFS=$'\t' consecutive tabs collapse and empty fields shift.
    tail -n +2 "$LEDGER" | awk -F'\t' -v OFS='\t' 'NF { print "L", $0 }'
}

{ digest_manifests; tag_ledger; } | awk -F'\t' \
    -v since="$SINCE_E" -v until="$UNTIL_E" -v fallback="$RATE_FALLBACK" -v api_rate="$API_RATE" '
function iso(e) { return strftime("%Y-%m-%dT%H:%M:%SZ", e, 1) }
function epoch(ts,   t) { t = ts; gsub(/[-:TZ]/, " ", t); return mktime(t, 1) }   # ISO-8601 Z → UTC epoch (gawk)
function vmcost(vm, a, b,   p, i, j, n, t, r, c, seg_t, seg_r) {
    # integrate the VM hourly rate over [a,b]: provision rate, stepped by provider rate_change rows
    if (b <= a) return 0
    p = vprovider[vm]; n = 0
    seg_t[++n] = a; seg_r[n] = (vrate[vm] > 0) ? vrate[vm] : def_rate
    for (i = 1; i <= nrc[p]; i++)
        if (rct[p, i] <= a) { if (vrate[vm] == 0 || rct[p, i] >= vprov[vm]) seg_r[1] = rcr[p, i] }
        else if (rct[p, i] < b) { seg_t[++n] = rct[p, i]; seg_r[n] = rcr[p, i] }
    for (i = 2; i <= n; i++) {   # insertion sort (rate_change rows may be unordered)
        t = seg_t[i]; r = seg_r[i]; j = i - 1
        while (j > 0 && seg_t[j] > t) { seg_t[j+1] = seg_t[j]; seg_r[j+1] = seg_r[j]; j-- }
        seg_t[j+1] = t; seg_r[j+1] = r
    }
    c = 0
    for (i = 1; i <= n; i++) { t = (i < n) ? seg_t[i+1] : b; c += (t - seg_t[i]) / 3600.0 * seg_r[i] / 100.0 }
    return c
}
$1 == "M" { nm++; mdir[nm] = $2; mname[nm] = $3; mcells[nm] = $4; msec[nm] = $5; next }
$1 == "L" {
    e = epoch($2); prov = $3; vm = $4; ev = $5; det = $6; rate = $7; bal = $8
    if (e < 0) { printf "[warn] unparsable ts skipped: %s\n", $2 > "/dev/stderr"; next }
    if (vm != "" && vm != "-" && !(vm in known)) { known[vm] = 1; vlist[++nv] = vm; vprovider[vm] = prov }
    if      (ev == "provision")   { vprov[vm] = e; vrate[vm] = rate + 0 }
    else if (ev == "teardown")    { vdown[vm] = e }
    else if (ev == "rate_change") { nrc[prov]++; rct[prov, nrc[prov]] = e; rcr[prov, nrc[prov]] = rate + 0
                                    if (e >= since && e <= until) notes[++nn] = iso(e) "  " prov " rate_change → " rate "¢/h  (" det ")" }
    else if (ev == "campaign_start" || ev == "campaign_done") {
        if (match(det, /dir=[^\t ]+/)) dirvm[substr(det, RSTART + 4, RLENGTH - 4)] = vm
    }
    else if (ev == "campaign_crash" && e >= since && e <= until) { notes[++nn] = iso(e) "  " vm "  " det }
    else if (ev == "incident"       && e >= since && e <= until) { notes[++nn] = iso(e) "  " det }
    else if (ev == "topup"          && e >= since && e <= until) {
        tn++; if (match(det, /amount=[0-9.]+/)) tsum += substr(det, RSTART + 7, RLENGTH - 7) + 0
    }
    if (bal != "" && e >= since && e <= until) {
        if (!bf_set || e < bf_e)  { bf_e = e; bf = bal + 0; bf_set = 1 }
        if (!bl_set || e >= bl_e) { bl_e = e; bl = bal + 0; bl_set = 1 }
    }
    next
}
END {
    def_rate = (fallback != "") ? fallback + 0 : ((api_rate != "") ? api_rate + 0 : 0)
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]
        if (!(vm in vprov) && !(vm in vdown)) { vskip[vm] = 1; continue }   # attribution-only vm
        a = (vm in vprov)  ? vprov[vm] : since; if (a < since) a = since
        b = (vm in vdown)  ? vdown[vm] : until; if (b > until) b = until
        vwall[vm] = (b > a) ? (b - a) / 3600.0 : 0
        vusd[vm]  = vmcost(vm, a, b)
        vavg[vm]  = (vwall[vm] > 0) ? vusd[vm] / vwall[vm] : 0
    }
    printf "== EDM cloud spend  %s → %s ==\n", iso(since), iso(until)
    if (nm > 0) {
        printf "\n%-28s %-9s %-16s %5s %8s %9s\n", "campaign", "provider", "vm", "cells", "GPU-h", "USD"
        for (i = 1; i <= nm; i++) {
            vm = (mdir[i] in dirvm) ? dirvm[mdir[i]] : "?"
            if (vm == "?" || vavg[vm] == 0) used_def = 1
            r  = (vm != "?" && vavg[vm] > 0) ? vavg[vm] : def_rate / 100.0
            usd = msec[i] / 3600.0 * r
            csum[vm] += usd; tot_compute += usd
            printf "%-28s %-9s %-16s %5d %8.2f %9.2f\n", mname[i], (vm == "?") ? "?" : vprovider[vm], vm, mcells[i], msec[i] / 3600.0, usd
        }
    }
    printf "\n%-16s %-9s %8s %9s %12s %13s\n", "vm", "provider", "wall-h", "wall-USD", "compute-USD", "overhead-USD"
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]; if (vm in vskip) continue
        printf "%-16s %-9s %8.2f %9.2f %12.2f %13.2f%s\n", vm, vprovider[vm], vwall[vm], vusd[vm], csum[vm] + 0, vusd[vm] - csum[vm], (vm in vdown) ? "" : "  (still running)"
        tot_wall += vusd[vm]
    }
    printf "\nTOTAL wall-clock spend in window: $%.2f  (manifest-backed compute $%.2f; overhead + in-flight cells $%.2f)\n", tot_wall, tot_compute, tot_wall - tot_compute
    if (tn > 0) printf "topups in window: $%.2f (%d)\n", tsum, tn
    if (bf_set && bl_set && bl_e > bf_e)
        printf "ledger-implied spend ($%.2f @%s + topups − $%.2f @%s): $%.2f\n", bf, iso(bf_e), bl, iso(bl_e), bf + tsum - bl
    if (def_rate == 0 && used_def) print "[warn] some rows lacked a rate and no fallback was available (priced at $0)"
    for (i = 1; i <= nn; i++) { if (i == 1) print "\nnotes:"; print "  " notes[i] }
}'
