#!/usr/bin/env bash
# availability_report.sh — Hot Aisle GPU availability report from the watcher log.
#
#   bash orchestration/availability_report.sh [--log FILE] [--days N] [--html FILE]
#
# Reads the TSV appended by hotaisle_availability.sh (ts_utc model gpu_count
# price_cents_h min_reserve_min; model none = polled-and-empty, error = failed poll).
# Text output: per-spec current state, last seen, availability fraction over 24h/7d,
# median closed-gap length. --html FILE writes a self-contained page (mirrors
# cost_report.sh — inline SVG, no CDNs): a timeline strip per spec over the last
# --days days (default 7; blank = watcher down, i.e. poll gap > 15 min), the summary
# table, and a list-price chart when the price varied. Plain bash + GNU awk; no network.
set -euo pipefail

LOG="${AVAIL_LOG:-$HOME/.local/share/edm-availability/availability.tsv}"
DAYS=7
HTML_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --log)     LOG=${2:?}; shift 2 ;;
        --days)    DAYS=${2:?}; shift 2 ;;
        --html)    HTML_FILE=${2:?}; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *)         echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[ -f "$LOG" ] || { echo "no availability log at $LOG (watcher not run yet?)" >&2; exit 1; }

run_report() {
awk -F'\t' -v days="$DAYS" -v html="${HTML_FILE:+1}" '
function iso(e)  { return strftime("%Y-%m-%dT%H:%MZ", e, 1) }
function epoch(ts,   t) { t = ts; gsub(/[-:TZ]/, " ", t); return mktime(t, 1) }   # ISO-8601 Z → UTC epoch (gawk)
function esc(s)  { gsub(/&/, "\\&amp;", s); gsub(/</, "\\&lt;", s); gsub(/>/, "\\&gt;", s); return s }
function dur(s)  { return (s < 3600) ? int(s/60) "m" : (s < 86400) ? int(s/3600) "h" int(s%3600/60) "m" : int(s/86400) "d" int(s%86400/3600) "h" }
function usd(c)  { return sprintf("$%.2f", c / 100.0) }
function median(j,   i, m, t) {   # median of the closed gaps of spec j (g[] filled in order)
    m = gn[j]; if (m == 0) return "-"
    for (i = 1; i <= m; i++) t[i] = g[j, i]
    asort(t)
    return dur((m % 2) ? t[(m+1)/2] : (t[m/2] + t[m/2+1]) / 2)
}
FNR > 1 && NF >= 2 {
    e = epoch($1)
    if (e != cur) { cur = e; np++; pt[np] = e }
    if ($2 == "error") { perr[np] = 1; nerr++; next }
    if ($2 == "none") next
    k = $2 " ×" $3
    if (!(k in ks)) { ks[k] = ++nk; kn[nk] = k }
    j = ks[k]
    pres[np, j] = 1; pprice[np, j] = $4; pmin[np, j] = $5
}
END {
    if (np == 0) { print "log has a header but no polls yet" > "/dev/stderr"; exit 1 }
    now = systime()
    CAP = 900   # poll gap beyond this = watcher down (3 × the 5-min cadence)
    for (i = 1; i <= nk; i++) ord[i] = i   # display order: by name (insertion sort, nk is tiny)
    for (i = 2; i <= nk; i++) {
        j = ord[i]; m = i - 1
        while (m > 0 && kn[ord[m]] > kn[j]) { ord[m+1] = ord[m]; m-- }
        ord[m+1] = j
    }
    # ── per-spec stats: window fractions, last-seen, closed gaps ──
    for (p = 1; p <= np; p++) {
        if (perr[p]) continue   # unknown — neither an up- nor a down-observation
        for (j = 1; j <= nk; j++) {
            up = ((p, j) in pres) ? 1 : 0
            if (pt[p] >= now -   86400) { t24[j]++; u24[j] += up }
            if (pt[p] >= now - 7*86400) { t7[j]++;  u7[j]  += up }
            if (up) {
                if (sawdown[j]) g[j, ++gn[j]] = pt[p] - lastup[j]
                lastup[j] = pt[p]; sawdown[j] = 0
                lastprice[j] = pprice[p, j]; lastmin[j] = pmin[p, j]
            } else if (j in lastup) sawdown[j] = 1
        }
    }
    for (p = 1; p < np; p++)   # watcher downtime = gaps between polls beyond CAP
        if (pt[p+1] - pt[p] > CAP) { wd_n++; wd_s += pt[p+1] - pt[p] }
    for (j = 1; j <= nk; j++) {   # current state string per spec
        if      (perr[np])       snow[j] = "error (last poll)"
        else if ((np, j) in pres) snow[j] = "UP @" pprice[np, j] "¢/GPU-h"
        else if (j in lastup)    snow[j] = "out " dur(now - lastup[j])
        else                     snow[j] = "never seen"
    }
    if (html) { html_out(); exit }
    # ── text output ──
    printf "== Hot Aisle availability  %s → %s  (%d polls, %d errors) ==\n", iso(pt[1]), iso(pt[np]), np, nerr
    if (now - pt[np] > CAP) printf "[warn] last poll was %s ago — watcher down?\n", dur(now - pt[np])
    if (wd_n) printf "watcher downtime: %d poll gaps >15m totalling %s\n", wd_n, dur(wd_s)
    printf "\n%-13s %-20s %-18s %8s %6s %6s %11s %5s\n", "spec", "now", "last seen", "¢/GPU-h", "24h", "7d", "median gap", "gaps"
    for (i = 1; i <= nk; i++) {
        j = ord[i]
        printf "%-13s %-20s %-18s %8s %6s %6s %11s %5d\n", kn[j], snow[j], \
               (j in lastup) ? iso(lastup[j]) : "never", (j in lastprice) ? lastprice[j] : "-", \
               t24[j] ? sprintf("%.1f%%", 100.0*u24[j]/t24[j]) : "-", \
               t7[j]  ? sprintf("%.1f%%", 100.0*u7[j]/t7[j])  : "-", median(j), gn[j] + 0
    }
    if (nk == 0) print "\nno offerings seen yet (every poll so far was empty or an error)"
}
function seg(j, y, a, b, st,   x, w, fill, lab) {   # one timeline segment, clipped to [T0,T1]
    if (a < T0) a = T0
    if (b > T1) b = T1
    if (b <= a) return
    x = GUT + (a - T0) / (T1 - T0) * PW; w = (b - a) / (T1 - T0) * PW
    if (w < 0.5) w = 0.5
    fill = (st == 1) ? "var(--up)" : (st == 2) ? "var(--err)" : "var(--down)"
    lab = (st == 1) ? "available" : (st == 2) ? "API error" : "unavailable"
    printf "<g><title>%s — %s %s → %s (%s)</title><rect x=\"%.2f\" y=\"%d\" width=\"%.2f\" height=\"%d\" fill=\"%s\"/></g>\n", \
           esc(kn[j]), lab, iso(a), iso(b), dur(b - a), x, y, w, BH, fill
}
function html_out(   i, j, p, y, fh, t0f, st, runst, runa, prevend, sege, tick, step, x, \
                     havep, pmn, pmx, npts, tv, pv, lastt, py, px, d, nvary, ci, cols) {
    T1 = now; T0 = now - days * 86400
    if (T0 < pt[1]) T0 = pt[1]
    GUT = 110; PW = 590; BH = 16; PITCH = 26
    print "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    print "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    print "<title>Hot Aisle GPU availability</title><style>"
    print ":root{--ink:#1a1d24;--muted:#6a7284;--border:#dee2ea;--border-2:#c5cad6;--accent:#3b6fb3;"
    print "--up:#0ca30c;--down:#e6e9ef;--err:#fab219;--upink:#006300;"
    print "--ui:system-ui,-apple-system,\"Segoe UI\",Roboto,\"Helvetica Neue\",Arial,sans-serif;"
    print "--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,\"Liberation Mono\",monospace}"
    print "html,body{margin:0;padding:0;background:#fff;color:var(--ink);font-family:var(--ui);font-size:14px;line-height:1.5}"
    print "main{max-width:960px;margin:0 auto;padding:24px 16px}"
    print "h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px}"
    print ".meta{color:var(--muted);font-family:var(--mono);font-size:11px;margin:0 0 16px}"
    print ".hero{font-size:16px;margin:14px 0} .hero .up{color:var(--upink);font-weight:600} .hero .out{font-weight:600}"
    print "table{border-collapse:collapse;width:100%;font-size:13px}"
    print "th,td{padding:4px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}"
    print "th{color:var(--muted);font-weight:600;border-bottom:1px solid var(--border-2)}"
    print "td.n,th.n{text-align:right;font-family:var(--mono)}"
    print ".note{color:var(--muted);font-size:12px;margin-top:6px}"
    print ".viz{margin:6px 0 10px} .viz svg{width:100%;height:auto;display:block;max-width:760px}"
    print ".viz text{font-family:var(--ui);font-size:12px}"
    print ".viz .lab{fill:#6a7284} .viz .tick{fill:#6a7284;font-size:11px}"
    print ".legend{font-size:12px;color:#6a7284;margin:2px 0 4px}"
    print ".legend span{margin-right:16px}"
    print ".sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}"
    print "</style></head><body><main>"
    print "<h1>Hot Aisle GPU availability</h1>"
    printf "<p class=\"meta\">updated %s &middot; log %s → %s &middot; %d polls (%d errors) every ~5 min &middot; read-only offerings endpoint — absent spec = out of capacity</p>\n", \
           iso(now), iso(pt[1]), iso(pt[np]), np, nerr
    # ── headline: the spec campaigns actually provision ──
    j = ("MI300X ×1" in ks) ? ks["MI300X ×1"] : 0
    if (j) {
        if ((np, j) in pres) printf "<p class=\"hero\">1&times;MI300X: <span class=\"up\">AVAILABLE @ %s/GPU-h</span></p>\n", usd(pprice[np, j])
        else if (j in lastup) printf "<p class=\"hero\">1&times;MI300X: <span class=\"out\">out of capacity for %s</span> (last seen %s)</p>\n", dur(now - lastup[j]), iso(lastup[j])
    } else print "<p class=\"hero\">1&times;MI300X: <span class=\"out\">never seen in this log</span></p>"
    # ── timeline strips ──
    if (nk > 0) {
        fh = nk * PITCH + 28
        printf "<h2>Timeline %s → %s</h2>\n", iso(T0), iso(T1)
        print "<p class=\"legend\"><span><span class=\"sw\" style=\"background:var(--up)\"></span>available</span>" \
              "<span><span class=\"sw\" style=\"background:var(--down)\"></span>unavailable</span>" \
              "<span><span class=\"sw\" style=\"background:var(--err)\"></span>API error</span>" \
              "<span><span class=\"sw\" style=\"background:#fff;border:1px solid var(--border-2)\"></span>blank = watcher down</span></p>"
        printf "<div class=\"viz\"><svg viewBox=\"0 0 720 %d\" role=\"img\" aria-label=\"Availability timeline per GPU spec\">\n", fh
        step = (days >= 3) ? 86400 : 21600   # day ticks; 6-hourly for short windows
        for (tick = int(T0 / step + 1) * step; tick < T1; tick += step) {
            x = GUT + (tick - T0) / (T1 - T0) * PW
            printf "<line x1=\"%.1f\" y1=\"4\" x2=\"%.1f\" y2=\"%d\" stroke=\"#eceef3\" stroke-width=\"1\"/>\n", x, x, nk * PITCH + 4
            printf "<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\" class=\"tick\">%s</text>\n", x, nk * PITCH + 18, \
                   (tick % 86400 == 0) ? strftime("%m-%d", tick, 1) : strftime("%Hh", tick, 1)
        }
        for (i = 1; i <= nk; i++) {
            j = ord[i]; y = 4 + (i - 1) * PITCH
            printf "<text x=\"%d\" y=\"%d\" text-anchor=\"end\" class=\"lab\">%s</text>\n", GUT - 8, y + 12, esc(kn[j])
            runst = -1; runa = 0; prevend = -1
            for (p = 1; p <= np; p++) {   # merge contiguous same-state polls into segments
                if (pt[p] > T1) break
                sege = (p < np) ? ((pt[p+1] - pt[p] <= CAP) ? pt[p+1] : pt[p] + CAP) \
                                : ((now - pt[np] <= CAP) ? now : pt[np] + CAP)
                if (sege < T0) continue
                st = perr[p] ? 2 : (((p, j) in pres) ? 1 : 0)
                if (st != runst || pt[p] > prevend) {
                    if (runst >= 0) seg(j, y, runa, prevend, runst)
                    runst = st; runa = pt[p]
                }
                prevend = sege
            }
            if (runst >= 0) seg(j, y, runa, prevend, runst)
        }
        print "</svg></div>"
    } else print "<p class=\"note\">no offerings seen yet — every poll so far returned an empty list or an error.</p>"
    # ── summary table ──
    if (nk > 0) {
        print "<h2>Summary</h2><table><tr><th>spec</th><th>now</th><th>last seen</th><th class=\"n\">$/GPU-h</th><th class=\"n\">min reserve</th><th class=\"n\">avail 24h</th><th class=\"n\">avail 7d</th><th class=\"n\">median gap</th><th class=\"n\">gaps</th></tr>"
        for (i = 1; i <= nk; i++) {
            j = ord[i]
            printf "<tr><td>%s</td><td>%s</td><td class=\"n\">%s</td><td class=\"n\">%s</td><td class=\"n\">%s</td><td class=\"n\">%s</td><td class=\"n\">%s</td><td class=\"n\">%s</td><td class=\"n\">%d</td></tr>\n", \
                   esc(kn[j]), \
                   ((np, j) in pres) ? "<span style=\"color:var(--upink);font-weight:600\">available</span>" : (perr[np] ? "unknown (error)" : "out"), \
                   (j in lastup) ? iso(lastup[j]) : "never", \
                   ((j in lastprice) && lastprice[j] + 0 > 0) ? usd(lastprice[j]) : "&mdash;", \
                   ((j in lastmin) && lastmin[j] != "-") ? lastmin[j] " min" : "&mdash;", \
                   t24[j] ? sprintf("%.1f%%", 100.0*u24[j]/t24[j]) : "&mdash;", \
                   t7[j]  ? sprintf("%.1f%%", 100.0*u7[j]/t7[j])  : "&mdash;", median(j), gn[j] + 0
        }
        print "</table>"
        print "<p class=\"note\">availability fraction = polls listing the spec / non-error polls in the window; gap = closed unavailable stretch between two sightings; price = OnDemandPrice per GPU-hour.</p>"
        if (wd_n) printf "<p class=\"note\">watcher downtime: %d poll gaps &gt;15&thinsp;min totalling %s (blank in the timeline).</p>\n", wd_n, dur(wd_s)
    }
    # ── list price over time (only if it ever varied) ──
    pmn = 1e12; pmx = -1
    for (p = 1; p <= np; p++) for (j = 1; j <= nk; j++)
        if (((p, j) in pres) && pprice[p, j] + 0 > 0) { d = pprice[p, j] + 0; if (d < pmn) pmn = d; if (d > pmx) pmx = d }
    if (pmx > pmn) {
        split("#2a78d6 #008300 #e87ba4 #eda100 #1baf7a #eb6834", cols, " ")   # fixed categorical order (validated palette)
        PH = 110
        print "<h2>List price</h2>"
        printf "<p class=\"legend\">"
        ci = 0
        for (i = 1; i <= nk; i++) { j = ord[i]; if (j in lastprice) printf "<span><span class=\"sw\" style=\"background:%s\"></span>%s</span>", cols[(ci++ % 6) + 1], esc(kn[j]) }
        print "</p>"
        printf "<div class=\"viz\"><svg viewBox=\"0 0 720 %d\" role=\"img\" aria-label=\"List price per GPU-hour over time\">\n", PH + 30
        printf "<text x=\"%d\" y=\"14\" text-anchor=\"end\" class=\"tick\">%s</text>\n", GUT - 8, usd(pmx)
        printf "<text x=\"%d\" y=\"%d\" text-anchor=\"end\" class=\"tick\">%s</text>\n", GUT - 8, PH + 4, usd(pmn)
        printf "<line x1=\"%d\" y1=\"10\" x2=\"%d\" y2=\"10\" stroke=\"#eceef3\"/>\n", GUT, GUT + PW
        printf "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#eceef3\"/>\n", GUT, PH, GUT + PW, PH
        ci = 0
        for (i = 1; i <= nk; i++) {   # step line per spec, vertices only where the price changes
            j = ord[i]; if (!(j in lastprice)) continue
            npts = 0; pv = ""; lastt = 0
            printf "<path fill=\"none\" stroke=\"%s\" stroke-width=\"2\" d=\"", cols[(ci++ % 6) + 1]
            for (p = 1; p <= np; p++) {
                if (!((p, j) in pres) || pprice[p, j] + 0 <= 0) continue
                d = pprice[p, j] + 0
                px = GUT + (pt[p] - pt[1]) / (now - pt[1] + 1) * PW
                py = PH - (d - pmn) / (pmx - pmn) * (PH - 10)
                if (npts == 0) printf "M%.1f %.1f", px, py
                else if (d != pv) printf " H%.1f V%.1f", px, py
                pv = d; lastt = px; npts++
            }
            if (npts > 0) printf " H%.1f", lastt
            print "\"/>"
        }
        printf "<text x=\"%d\" y=\"%d\" class=\"tick\">%s</text><text x=\"%d\" y=\"%d\" text-anchor=\"end\" class=\"tick\">%s</text>\n", \
               GUT, PH + 16, iso(pt[1]), GUT + PW, PH + 16, iso(now)
        print "</svg></div>"
        print "<p class=\"note\">sampled while the spec was offered; flat stretches between sightings are interpolated.</p>"
    }
    print "</main></body></html>"
}' "$LOG"
}

if [ -n "$HTML_FILE" ]; then
    tmp="${HTML_FILE}.tmp.$$"
    run_report > "$tmp" && mv "$tmp" "$HTML_FILE"
    echo "wrote $HTML_FILE" >&2
else
    run_report
fi
