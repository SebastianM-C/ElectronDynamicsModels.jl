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
#   • per-VM wall-clock cost = Σ over the VM's lifetime SPANS (each provision→teardown clipped to
#     the window). A VM NAME is reused across rentals, so one name can hold several spans; a span
#     with no teardown is still LIVE and bills through --until (= now), marked "live" on the page.
#     Overhead = wall − compute; cells still IN FLIGHT have no manifest yet, so they appear as
#     overhead until their run_<uuid>.toml lands — rerun after download/publish.
#   All displayed timestamps render in Europe/Bucharest (the ledger and all math stay UTC).
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
# TZ set for the reporter awk only: strftime() (display) honours it; ledger parsing (mktime …,1)
# and the shell-side SINCE/UNTIL (date -u) stay UTC, so this converts DISPLAY times to Bucharest.
{ digest_manifests; tag_ledger; } | TZ='Europe/Bucharest' awk -F'\t' \
    -v since="$SINCE_E" -v until="$UNTIL_E" -v fallback="$RATE_FALLBACK" -v api_rate="$API_RATE" \
    -v html="${HTML_FILE:+1}" '
# Times render in the process TZ (run_report sets TZ=Europe/Bucharest); the ledger stays UTC
# and all epoch math below is UTC (mktime/date -u), so this is a display-only conversion.
function iso(e) { return strftime("%Y-%m-%d %H:%M %Z", e) }
function epoch(ts,   t) { t = ts; gsub(/[-:TZ]/, " ", t); return mktime(t, 1) }   # ISO-8601 Z → UTC epoch (gawk)
function esc(s) { gsub(/&/, "\\&amp;", s); gsub(/</, "\\&lt;", s); gsub(/>/, "\\&gt;", s); return s }
function vmcost(a, b, base, prov_time, p,   i, j, n, t, r, c, seg_t, seg_r) {
    # integrate the hourly rate over [a,b] for one span: its provision rate `base`, stepped by
    # provider rate_change rows (prov_time gates which changes predate the span). rate_change is
    # unused so far, so in practice this is (b-a)/3600 * base/100.
    if (b <= a) return 0
    n = 0
    seg_t[++n] = a; seg_r[n] = (base > 0) ? base : def_rate
    for (i = 1; i <= nrc[p]; i++)
        if (rct[p, i] <= a) { if (base == 0 || rct[p, i] >= prov_time) seg_r[1] = rcr[p, i] }
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
    if      (ev == "provision") {   # open a new lifetime span (VM names get REUSED across rentals)
        if ((vm in vopen) && vopen[vm] > 0 && sp_end[vm, vopen[vm]] == 0) sp_end[vm, vopen[vm]] = e   # close a dangling prior span
        k = ++nspan[vm]; sp_start[vm, k] = e; sp_rate[vm, k] = rate + 0; sp_end[vm, k] = 0; vopen[vm] = k
    }
    else if (ev == "teardown") {    # close the open span; an orphan teardown (span starts pre-window) clamps to `since`
        if ((vm in vopen) && vopen[vm] > 0 && sp_end[vm, vopen[vm]] == 0) { sp_end[vm, vopen[vm]] = e; vopen[vm] = 0 }
        else { k = ++nspan[vm]; sp_start[vm, k] = 0; sp_rate[vm, k] = 0; sp_end[vm, k] = e }
    }
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
    # ── per-VM wall / cost: sum over every lifetime span (a reused VM name has >1). A span with
    #    no teardown is LIVE → billed through `until` (= now), so a running VM shows in-progress cost. ──
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]
        if (!(vm in nspan)) { vskip[vm] = 1; continue }   # attribution-only vm (campaign rows, never provisioned)
        vwall[vm] = 0; vusd[vm] = 0; vrun[vm] = 0
        for (k = 1; k <= nspan[vm]; k++) {
            a = (sp_start[vm, k] > 0) ? sp_start[vm, k] : since; if (a < since) a = since
            b = (sp_end[vm, k]   > 0) ? sp_end[vm, k]   : until; if (b > until) b = until
            if (sp_end[vm, k] == 0) vrun[vm] = 1   # open span ⇒ VM is live right now
            if (b <= a) continue
            base = sp_rate[vm, k]
            if (base == 0 && def_rate == 0) used_def = 1
            vwall[vm] += (b - a) / 3600.0
            vusd[vm]  += vmcost(a, b, base, (sp_start[vm, k] > 0 ? sp_start[vm, k] : a), vprovider[vm])
        }
        vavg[vm] = (vwall[vm] > 0) ? vusd[vm] / vwall[vm] : 0
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
    printf "== EDM cloud spend  %s → %s  (times Europe/Bucharest) ==\n", iso(since), iso(until)
    if (nm > 0) {
        printf "\n%-28s %-9s %-16s %5s %8s %9s\n", "campaign", "provider", "vm", "cells", "GPU-h", "USD"
        for (i = 1; i <= nm; i++)
            printf "%-28s %-9s %-16s %5d %8.2f %9.2f\n", mname[i], (cvm[i] == "-") ? "-" : vprovider[cvm[i]], cvm[i], mcells[i], msec[i] / 3600.0, cusd[i]
    }
    printf "\n%-16s %-9s %8s %9s %12s %13s\n", "vm", "provider", "wall-h", "wall-USD", "compute-USD", "overhead-USD"
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]; if (vm in vskip) continue
        printf "%-16s %-9s %8.2f %9.2f %12.2f %13.2f%s\n", vm, vprovider[vm], vwall[vm], vusd[vm], csum[vm] + 0, vusd[vm] - csum[vm], vrun[vm] ? "  (live — billed to now)" : ""
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
function rbar(x, y, w, h, r, fill, title,   p) {   # horizontal bar, 4px rounded data-end, square baseline end
    if (w < r + 1)
        p = sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.2f\" height=\"%d\" fill=\"%s\"/>", x, y, (w < 0.75 ? 0.75 : w), h, fill)
    else
        p = sprintf("<path d=\"M%.1f %.1f h%.2f a%d %d 0 0 1 %d %d v%d a%d %d 0 0 1 -%d %d h-%.2f z\" fill=\"%s\"/>", \
                    x, y, w - r, r, r, r, r, h - 2*r, r, r, r, r, w - r, fill)
    if (title != "") p = "<g><title>" title "</title>" p "</g>"
    return p
}
function html_out(   i, j, k, p, vm, m, ord, mx, GUT, PW, BH, PITCH, y, w, wc, wo, fh, lab) {
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
    print ":root{--s1:#2a78d6;--s2:#008300}"   # validated 2-slot categorical palette (light mode, white surface)
    print ".viz{margin:6px 0 10px} .viz svg{width:100%;height:auto;display:block;max-width:760px}"
    print ".viz text{font-family:var(--ui);font-size:12px}"
    print ".viz .lab{fill:#6a7284} .viz .val{fill:var(--ink)} .viz .inlab{fill:#fff}"
    print ".legend{font-size:12px;color:#6a7284;margin:2px 0 4px}"
    print ".legend span{margin-right:16px}"
    print ".sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}"
    print "</style></head><body><main>"
    print "<h1>EDM cloud campaign costs</h1>"
    printf "<p class=\"meta\">updated %s &middot; window %s → %s &middot; all times Europe/Bucharest (ledger stays UTC) &middot; derived from the cost ledger + run-manifest [timing] (never stored in run metadata)</p>\n", iso(systime()), iso(since), iso(until)
    # ── figure: cost per campaign (single series → no legend; magnitude, sorted desc) ──
    GUT = 210; PW = 440; BH = 16; PITCH = 26
    m = 0
    for (i = 1; i <= nm; i++) if (cusd[i] > 0) ord[++m] = i
    for (j = 2; j <= m; j++) {   # insertion sort desc by USD
        k = ord[j]; i = j - 1
        while (i > 0 && cusd[ord[i]] < cusd[k]) { ord[i+1] = ord[i]; i-- }
        ord[i+1] = k
    }
    if (m > 0) {
        mx = cusd[ord[1]]
        fh = m * PITCH + 12
        printf "<h2>Cost per campaign</h2><div class=\"viz\"><svg viewBox=\"0 0 720 %d\" role=\"img\" aria-label=\"Cost per campaign, USD\">\n", fh
        printf "<line x1=\"%d\" y1=\"4\" x2=\"%d\" y2=\"%d\" stroke=\"#c3c2b7\" stroke-width=\"1\"/>\n", GUT, GUT, fh - 4
        for (j = 1; j <= m; j++) {
            i = ord[j]; y = 6 + (j - 1) * PITCH; w = cusd[i] / mx * PW
            printf "<text x=\"%d\" y=\"%d\" text-anchor=\"end\" class=\"lab\">%s</text>\n", GUT - 8, y + 12, esc(mname[i])
            print rbar(GUT + 1, y, w, BH, 4, "var(--s1)", \
                       esc(mname[i]) " — " esc(cvm[i]) " — " sprintf("%.2f GPU-h — $%.2f", msec[i] / 3600.0, cusd[i]))
            printf "<text x=\"%.1f\" y=\"%d\" class=\"val\">$%.2f</text>\n", GUT + 1 + w + 6, y + 12, cusd[i]
        }
        print "</svg></div>"
    }
    if (nm > 0) {
        print "<h2>Per campaign</h2><table><tr><th>campaign</th><th>provider</th><th>vm</th><th class=\"n\">cells</th><th class=\"n\">GPU-h</th><th class=\"n\">USD</th></tr>"
        for (i = 1; i <= nm; i++)
            printf "<tr><td>%s</td><td>%s</td><td>%s</td><td class=\"n\">%d</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td></tr>\n", esc(mname[i]), (cvm[i] == "-") ? "&mdash;" : esc(vprovider[cvm[i]]), (cvm[i] == "-") ? "&mdash;" : esc(cvm[i]), mcells[i], msec[i] / 3600.0, cusd[i]
        print "</table>"
        print "<p class=\"note\">campaigns marked &mdash; have no cloud-ledger attribution (local / SLURM / pre-ledger) and are shown at $0.</p>"
    }
    # ── figure: per-VM compute vs overhead (2 series → legend; stacked, 2px surface gaps) ──
    m = 0; mx = 0
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]; if (vm in vskip) continue
        ord[++m] = i; if (vusd[vm] > mx) mx = vusd[vm]
    }
    if (m > 0 && mx > 0) {
        fh = m * PITCH + 12
        print "<h2>Per VM: compute vs overhead</h2>"
        print "<p class=\"legend\"><span><span class=\"sw\" style=\"background:var(--s1)\"></span>compute (manifest-backed)</span><span><span class=\"sw\" style=\"background:var(--s2)\"></span>overhead + in-flight</span></p>"
        printf "<div class=\"viz\"><svg viewBox=\"0 0 720 %d\" role=\"img\" aria-label=\"Per-VM wall-clock cost split into compute and overhead, USD\">\n", fh
        printf "<line x1=\"%d\" y1=\"4\" x2=\"%d\" y2=\"%d\" stroke=\"#c3c2b7\" stroke-width=\"1\"/>\n", GUT, GUT, fh - 4
        for (j = 1; j <= m; j++) {
            vm = vlist[ord[j]]; y = 6 + (j - 1) * PITCH
            wc = (csum[vm] > 0 ? csum[vm] : 0) / mx * PW
            wo = (vusd[vm] - csum[vm] > 0 ? vusd[vm] - csum[vm] : 0) / mx * PW
            printf "<text x=\"%d\" y=\"%d\" text-anchor=\"end\" class=\"lab\">%s</text>\n", GUT - 8, y + 12, esc(vm)
            if (wc > 0.1) printf "<g><title>%s — compute $%.2f</title><rect x=\"%d\" y=\"%d\" width=\"%.2f\" height=\"%d\" fill=\"var(--s1)\"/></g>\n", esc(vm), csum[vm], GUT + 1, y, wc, BH
            if (wo > 0.1) print rbar(GUT + 1 + wc + (wc > 0.1 ? 2 : 0), y, wo, BH, 4, "var(--s2)", \
                                     esc(vm) " — overhead $" sprintf("%.2f", vusd[vm] - csum[vm]) (vrun[vm] ? " — live, billed to now" : ""))
            lab = sprintf("$%.2f", csum[vm] + 0)
            if (wc > length(lab) * 6.8 + 14) printf "<text x=\"%d\" y=\"%d\" class=\"inlab\">%s</text>\n", GUT + 8, y + 12, lab
            printf "<text x=\"%.1f\" y=\"%d\" class=\"val\">$%.2f</text>\n", GUT + 1 + wc + 2 + wo + 6, y + 12, vusd[vm]
        }
        print "</svg></div>"
    }
    print "<h2>Per VM</h2><table><tr><th>vm</th><th>provider</th><th class=\"n\">wall-h</th><th class=\"n\">wall USD</th><th class=\"n\">compute USD</th><th class=\"n\">overhead USD</th><th></th></tr>"
    for (i = 1; i <= nv; i++) {
        vm = vlist[i]; if (vm in vskip) continue
        printf "<tr><td>%s</td><td>%s</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td class=\"n\">%.2f</td><td>%s</td></tr>\n", esc(vm), esc(vprovider[vm]), vwall[vm], vusd[vm], csum[vm] + 0, vusd[vm] - csum[vm], vrun[vm] ? "<span class=\"run\">live</span>" : ""
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
