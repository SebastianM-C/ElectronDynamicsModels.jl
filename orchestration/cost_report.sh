#!/usr/bin/env bash
# cost_report.sh — EDM cloud spend report from the persistent cost ledger + run manifests.
#
#   bash orchestration/cost_report.sh [--since ISO] [--until ISO] [--ledger FILE] \
#        [--rate-cents N] [--html FILE] [results_dir ...]
#
# The ledger (~/.config/edm-cloud-ledger.tsv, appended by backends/hotaisle.sh and runpod.sh) is
# an append-only TSV:  ts_utc  provider  vm  event  detail  rate_cents_h  balance_usd
# events: provision (rate_cents_h = the VM's $/h in CENTS at provision — list prices drift, so the
# offering price is captured per-row; reconciliation vs authoritative top-ups confirmed 2026-07-17
# that Hot Aisle bills a VM at its provision-time list price for its whole life), campaign_start /
# campaign_done / campaign_crash (detail carries campaign=<name> dir=<outdir>), teardown, topup
# (detail amount=<usd>), balance, rate_change (generic provider-wide re-rate hook, integrated
# piecewise — unused so far), incident (free text, surfaced as a note).
#
# What it prints (text to stdout, or a self-contained HTML page with --html FILE):
#   • per-campaign compute = Σ [timing].total over run_<uuid>.toml in each results dir given,
#     priced at the owning VM's time-averaged rate (attribution via campaign_* rows' dir=…);
#     dirs with no ledger attribution are non-cloud (local/SLURM) or pre-ledger → shown at $0
#   • per-VM wall-clock cost (provision→teardown clipped to the window; running VMs bill through
#     --until) and overhead = wall − compute. Cells still IN FLIGHT have no manifest yet, so they
#     appear as overhead until their run_<uuid>.toml lands — rerun after download/publish.
#   • per-provider reconciliation: window top-ups + balance-implied spend, and LIFETIME
#     top-ups vs last known balance ⇒ lifetime spend. Costs stay DERIVED (ledger + manifests),
#     never stored in run metadata — a rate correction heals everything by regeneration.
# Rate for a VM missing a provision rate: live Hot Aisle API lookup (token file), else
# --rate-cents, else 0 + warning. Plain bash + GNU awk/date; jq/curl only for the API fallback.
#
# MULTI-MACHINE: each campaign-driving machine appends to its own local ledger. The reporter
# merges the main ledger with every *.tsv under ~/.config/edm-cloud-ledgers/ (drop other
# machines' ledgers there — the dashboard publish hook rsyncs them in), or takes explicit
# --ledger FILE (repeatable; replaces the defaults). Byte-identical rows are deduped, so a
# ledger accidentally present twice is harmless. Convention: account-scoped rows (topup,
# balance) live ONLY in the publish hub's own ledger or provider lifetime sums double-count;
# machine-scoped rows (provision/campaign/teardown) live where the campaign was driven.
# Campaign→VM attribution matches dir=… first (machine-local paths), then falls back to the
# campaign=<name> token when exactly one VM claims that name (ambiguous names never match).
set -euo pipefail

LEDGER="${EDM_CLOUD_LEDGER:-$HOME/.config/edm-cloud-ledger.tsv}"
LEDGER_DIR="${EDM_CLOUD_LEDGER_DIR:-$HOME/.config/edm-cloud-ledgers}"
LEDGERS=()
SINCE="1970-01-01T00:00:00Z"
UNTIL="$(date -u +%FT%TZ)"
RATE_FALLBACK=""
HTML_FILE=""
DIRS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --since)      SINCE=${2:?}; shift 2 ;;
        --until)      UNTIL=${2:?}; shift 2 ;;
        --ledger)     LEDGERS+=("${2:?}"); shift 2 ;;
        --rate-cents) RATE_FALLBACK=${2:?}; shift 2 ;;
        --html)       HTML_FILE=${2:?}; shift 2 ;;
        -h|--help)    sed -n '2,28p' "$0"; exit 0 ;;
        *)            DIRS+=("$1"); shift ;;
    esac
done
if [ ${#LEDGERS[@]} -eq 0 ]; then   # defaults: the machine ledger + any synced-in remote ledgers
    [ -f "$LEDGER" ] && LEDGERS+=("$LEDGER")
    if [ -d "$LEDGER_DIR" ]; then
        for f in "$LEDGER_DIR"/*.tsv; do [ -f "$f" ] && LEDGERS+=("$f"); done
    fi
fi
[ ${#LEDGERS[@]} -gt 0 ] || { echo "no ledger at $LEDGER or $LEDGER_DIR/*.tsv (nothing recorded yet?)" >&2; exit 1; }
SINCE_E=$(date -u -d "$SINCE" +%s)
UNTIL_E=$(date -u -d "$UNTIL" +%s)

# Current Hot Aisle list price (cents/GPU-h) — fetched ONLY if a provision row lacks a rate.
# Team handle stays out of git: HOTAISLE_TEAM (config.env) or ~/.config/hotaisle/team.
hotaisle_list_rate() {
    local tokf="${HOTAISLE_TOKEN_FILE:-$HOME/.config/hotaisle/token}"
    local team="${HOTAISLE_TEAM:-$(cat "$HOME/.config/hotaisle/team" 2>/dev/null)}"
    [ -f "$tokf" ] && [ -n "$team" ] || return 1
    curl -fsS --max-time 10 -H "Authorization: Token $(cat "$tokf")" \
        "https://admin.hotaisle.app/api/teams/${team}/virtual_machines/available/" \
      | jq -r '[.[]|select(.Specs.gpus[0].model=="MI300X" and .Specs.gpus[0].count==1)|.OnDemandPrice][0] // empty'
}
API_RATE=""
if awk -F'\t' 'FNR>1 && $4=="provision" && $2=="hotaisle" && $6==""{ex=1} END{exit !ex}' "${LEDGERS[@]}"; then
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
    # Merge every ledger (header-stripped); sort -u dedupes byte-identical rows (a ledger
    # synced twice is harmless) and ISO timestamps sort lexicographically = chronologically.
    awk 'FNR > 1' "${LEDGERS[@]}" | sort -u | awk -F'\t' -v OFS='\t' 'NF { print "L", $0 }'
}

run_report() {
{ digest_manifests; tag_ledger; } | awk -F'\t' \
    -v since="$SINCE_E" -v until="$UNTIL_E" -v fallback="$RATE_FALLBACK" -v api_rate="$API_RATE" \
    -v html="${HTML_FILE:+1}" '
function iso(e) { return strftime("%Y-%m-%dT%H:%M:%SZ", e, 1) }
function epoch(ts,   t) { t = ts; gsub(/[-:TZ]/, " ", t); return mktime(t, 1) }   # ISO-8601 Z → UTC epoch (gawk)
function esc(s) { gsub(/&/, "\\&amp;", s); gsub(/</, "\\&lt;", s); gsub(/>/, "\\&gt;", s); return s }
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
    if (prov != "" && !(prov in pseen)) { pseen[prov] = 1; plist[++np] = prov }
    if (vm != "" && vm != "-" && !(vm in known)) { known[vm] = 1; vlist[++nv] = vm; vprovider[vm] = prov }
    if      (ev == "provision")   { vprov[vm] = e; vrate[vm] = rate + 0 }
    else if (ev == "teardown")    { vdown[vm] = e }
    else if (ev == "rate_change") { nrc[prov]++; rct[prov, nrc[prov]] = e; rcr[prov, nrc[prov]] = rate + 0
                                    if (e >= since && e <= until) notes[++nn] = iso(e) "  " prov " rate_change → " rate "¢/h  (" det ")" }
    else if (ev == "campaign_start" || ev == "campaign_done") {
        if (match(det, /dir=[^\t ]+/)) dirvm[substr(det, RSTART + 4, RLENGTH - 4)] = vm
        if (match(det, /campaign=[^\t ]+/)) {   # name fallback for cross-machine dirs
            cn = substr(det, RSTART + 9, RLENGTH - 9)
            if ((cn in namevm) && namevm[cn] != vm) nameamb[cn] = 1; else namevm[cn] = vm
        }
    }
    else if (ev == "campaign_crash" && e >= since && e <= until) { notes[++nn] = iso(e) "  " vm "  " det }
    else if (ev == "incident"       && e >= since && e <= until) { notes[++nn] = iso(e) "  " det }
    else if (ev == "topup") {
        amt = 0; if (match(det, /amount=[0-9.]+/)) amt = substr(det, RSTART + 7, RLENGTH - 7) + 0
        lt_sum[prov] += amt; lt_n[prov]++
        if (e >= since && e <= until) { wt_sum[prov] += amt; wt_n[prov]++ }
    }
    if (bal != "") {
        if (!(prov in lb_e) || e >= lb_e[prov]) { lb_e[prov] = e; lb[prov] = bal + 0 }
        if (e >= since && e <= until) {
            if (!(prov in wbf_e) || e < wbf_e[prov])  { wbf_e[prov] = e; wbf[prov] = bal + 0 }
            if (!(prov in wbl_e) || e >= wbl_e[prov]) { wbl_e[prov] = e; wbl[prov] = bal + 0 }
        }
    }
    next
}
END {
    def_rate = (fallback != "") ? fallback + 0 : ((api_rate != "") ? api_rate + 0 : 0)
    # ── per-VM wall / cost ──
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]
        if (!(vm in vprov) && !(vm in vdown)) { vskip[vm] = 1; continue }   # attribution-only vm
        if ((vm in vprov) && vrate[vm] == 0 && def_rate == 0) used_def = 1
        a = (vm in vprov) ? vprov[vm] : since; if (a < since) a = since
        b = (vm in vdown) ? vdown[vm] : until; if (b > until) b = until
        vwall[vm] = (b > a) ? (b - a) / 3600.0 : 0
        vusd[vm]  = vmcost(vm, a, b)
        vavg[vm]  = (vwall[vm] > 0) ? vusd[vm] / vwall[vm] : 0
        tot_wall += vusd[vm]
    }
    # ── per-campaign compute (unattributed dirs = non-cloud/pre-ledger → $0) ──
    for (i = 1; i <= nm; i++) {
        cvm[i] = (mdir[i] in dirvm) ? dirvm[mdir[i]] : \
                 ((mname[i] in namevm) && !(mname[i] in nameamb) ? namevm[mname[i]] : "-")
        cusd[i] = (cvm[i] != "-") ? msec[i] / 3600.0 * vavg[cvm[i]] : 0
        if (cvm[i] != "-") { csum[cvm[i]] += cusd[i]; tot_compute += cusd[i] }
    }
    if (html) { html_out(); exit }
    # ── text output ──
    printf "== EDM cloud spend  %s → %s ==\n", iso(since), iso(until)
    if (nm > 0) {
        printf "\n%-28s %-9s %-16s %5s %8s %9s\n", "campaign", "provider", "vm", "cells", "GPU-h", "USD"
        for (i = 1; i <= nm; i++)
            printf "%-28s %-9s %-16s %5d %8.2f %9.2f\n", mname[i], (cvm[i] == "-") ? "-" : vprovider[cvm[i]], cvm[i], mcells[i], msec[i] / 3600.0, cusd[i]
    }
    printf "\n%-16s %-9s %8s %9s %12s %13s\n", "vm", "provider", "wall-h", "wall-USD", "compute-USD", "overhead-USD"
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]; if (vm in vskip) continue
        printf "%-16s %-9s %8.2f %9.2f %12.2f %13.2f%s\n", vm, vprovider[vm], vwall[vm], vusd[vm], csum[vm] + 0, vusd[vm] - csum[vm], (vm in vdown) ? "" : "  (still running)"
    }
    printf "\nTOTAL wall-clock spend in window: $%.2f  (manifest-backed compute $%.2f; overhead + in-flight cells $%.2f)\n", tot_wall, tot_compute, tot_wall - tot_compute
    for (i = 1; i <= np; i++) {
        p = plist[i]
        if (!(p in lt_sum) && !(p in lb)) continue
        printf "\n%s:", p
        if (p in wt_sum) printf " window top-ups $%.2f (%d)", wt_sum[p], wt_n[p]
        if ((p in wbf_e) && (p in wbl_e) && wbl_e[p] > wbf_e[p])
            printf "; implied window spend $%.2f ($%.2f @%s + top-ups − $%.2f @%s)", wbf[p] + wt_sum[p] - wbl[p], wbf[p], iso(wbf_e[p]), wbl[p], iso(wbl_e[p])
        if (p in lt_sum) {
            printf "\n%s lifetime: top-ups $%.2f (%d)", p, lt_sum[p], lt_n[p]
            if (p in lb) printf ", balance $%.2f @%s ⇒ spent $%.2f", lb[p], iso(lb_e[p]), lt_sum[p] - lb[p]
        }
        printf "\n"
    }
    if (used_def) print "[warn] some VMs lacked a recorded rate and no fallback was available (priced at $0)"
    for (i = 1; i <= nn; i++) { if (i == 1) print "\nnotes:"; print "  " notes[i] }
}
function html_out(   i, p, vm) {
    print "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    print "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    print "<title>EDM cloud costs</title><style>"
    print ":root{--ink:#1a1d24;--muted:#6a7284;--border:#dee2ea;--border-2:#c5cad6;--accent:#3b6fb3;"
    print "--ui:system-ui,-apple-system,\"Segoe UI\",Roboto,\"Helvetica Neue\",Arial,sans-serif;"
    print "--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,\"Liberation Mono\",monospace}"
    print "html,body{margin:0;padding:0;background:#fff;color:var(--ink);font-family:var(--ui);font-size:14px;line-height:1.5}"
    print "main{max-width:960px;margin:0 auto;padding:24px 16px}"
    print "h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px}"
    print ".meta{color:var(--muted);font-family:var(--mono);font-size:11px;margin:0 0 16px}"
    print "table{border-collapse:collapse;width:100%;font-size:13px}"
    print "th,td{padding:4px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}"
    print "th{color:var(--muted);font-weight:600;border-bottom:1px solid var(--border-2)}"
    print "td.n,th.n{text-align:right;font-family:var(--mono)}"
    print "tr.total td{border-top:1px solid var(--border-2);font-weight:600}"
    print ".note{color:var(--muted);font-size:12px;margin-top:6px}"
    print ".run{color:var(--accent);font-weight:600}"
    print "</style></head><body><main>"
    print "<h1>EDM cloud campaign costs</h1>"
    printf "<p class=\"meta\">updated %s &middot; window %s → %s &middot; derived from the cost ledger + run-manifest [timing] (never stored in run metadata)</p>\n", iso(systime()), iso(since), iso(until)
    if (nm > 0) {
        print "<h2>Per campaign</h2><table><tr><th>campaign</th><th>provider</th><th>vm</th><th class=\"n\">cells</th><th class=\"n\">GPU-h</th><th class=\"n\">USD</th></tr>"
        for (i = 1; i <= nm; i++)
            printf "<tr><td>%s</td><td>%s</td><td>%s</td><td class=\"n\">%d</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td></tr>\n", esc(mname[i]), (cvm[i] == "-") ? "&mdash;" : esc(vprovider[cvm[i]]), (cvm[i] == "-") ? "&mdash;" : esc(cvm[i]), mcells[i], msec[i] / 3600.0, cusd[i]
        print "</table>"
        print "<p class=\"note\">campaigns marked &mdash; have no cloud-ledger attribution (local / SLURM / pre-ledger) and are shown at $0.</p>"
    }
    print "<h2>Per VM</h2><table><tr><th>vm</th><th>provider</th><th class=\"n\">wall-h</th><th class=\"n\">wall USD</th><th class=\"n\">compute USD</th><th class=\"n\">overhead USD</th><th></th></tr>"
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]; if (vm in vskip) continue
        printf "<tr><td>%s</td><td>%s</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td>%s</td></tr>\n", esc(vm), esc(vprovider[vm]), vwall[vm], vusd[vm], csum[vm] + 0, vusd[vm] - csum[vm], (vm in vdown) ? "" : "<span class=\"run\">still running</span>"
    }
    printf "<tr class=\"total\"><td>total</td><td></td><td class=\"n\"></td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td></td></tr>\n", tot_wall, tot_compute, tot_wall - tot_compute
    print "</table>"
    print "<p class=\"note\">overhead = wall-clock − manifest-backed compute: provision/warm, idle gaps, drain tails, incidents — and cells still in flight (their manifests have not landed yet).</p>"
    print "<h2>Providers &mdash; lifetime</h2><table><tr><th>provider</th><th class=\"n\">top-ups (window)</th><th class=\"n\">top-ups (lifetime)</th><th class=\"n\">last balance</th><th></th><th class=\"n\">lifetime spend</th></tr>"
    for (i = 1; i <= np; i++) {
        p = plist[i]
        if (!(p in lt_sum) && !(p in lb)) continue
        printf "<tr><td>%s</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td class=\"n\">%s</td><td>%s</td><td class=\"n\">%s</td></tr>\n", esc(p), wt_sum[p] + 0, lt_sum[p] + 0, (p in lb) ? sprintf("%.2f", lb[p]) : "?", (p in lb) ? "<span class=\"meta\">@" iso(lb_e[p]) "</span>" : "", (p in lb) ? sprintf("%.2f", lt_sum[p] - lb[p]) : "?"
    }
    print "</table>"
    if (nn > 0) {
        print "<h2>Notes</h2>"
        for (i = 1; i <= nn; i++) printf "<p class=\"note\">%s</p>\n", esc(notes[i])
    }
    print "</main></body></html>"
}'
}

if [ -n "$HTML_FILE" ]; then
    tmp="${HTML_FILE}.tmp.$$"
    run_report > "$tmp" && mv "$tmp" "$HTML_FILE"
    echo "wrote $HTML_FILE" >&2
else
    run_report
fi
