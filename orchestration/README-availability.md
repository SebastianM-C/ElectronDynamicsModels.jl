# Hot Aisle availability watcher

Standalone section until `orchestration/README.md` lands on main (PR #42); fold it in there afterwards.

Polls the Hot Aisle offerings list every 5 min and logs which GPU specs are on offer, so
capacity gaps become visible over time and a returning 1×MI300X fires an ntfy ping.
`GET /virtual_machines/available/` is read-only and free — polling never provisions anything;
an absent spec (e.g. no 1×MI300X row) means out of capacity right now.

- `hotaisle_availability.sh` — one poll: appends one TSV row per offering to
  `${AVAIL_LOG:-~/.local/share/edm-availability/availability.tsv}`
  (`ts_utc  model  gpu_count  price_cents_h  min_reserve_min`; price = cents per GPU-hour).
  `model=none` marks a polled-and-empty list (distinct from watcher downtime), `model=error`
  a failed poll. On a 1×MI300X absent→available transition it notifies via the existing
  campaign ntfy creds (`$NTFY_ENV`, default `~/.config/ntfy/edm-campaigns.env`) with the
  price and how long the spec was gone.
- `availability_report.sh [--log FILE] [--days N] [--html FILE]` — summary table
  (current state, last seen, availability fraction 24h/7d, median gap) to stdout, or with
  `--html` a self-contained page: a timeline strip per spec (blank = watcher down), the
  summary table, and a list-price chart when the price varied.

## Setup (systemd user timer)

```sh
cp orchestration/hotaisle_availability.sh ~/.local/bin/edm-hotaisle-availability
chmod +x ~/.local/bin/edm-hotaisle-availability
cp orchestration/edm-hotaisle-availability.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now edm-hotaisle-availability.timer
loginctl enable-linger   # or the timer dies on logout
```

Config (all outside git): the Hot Aisle API token at `~/.config/hotaisle/token` (0600,
per-machine — mint one in the TUI at `ssh admin.hotaisle.app`); the team handle from
`$HOTAISLE_TEAM`, `config.env` beside the script, `~/.config/hotaisle/env`
(`HOTAISLE_TEAM=...`) or `~/.config/hotaisle/team` — the installed copy in `~/.local/bin`
has no `config.env` beside it, so create one of the latter two. ntfy is optional; without
creds the watcher only logs.

## Publishing the dashboard

On the machine holding `~/.config/research-publish.env` (do not copy those creds around):

```sh
bash orchestration/availability_report.sh --html /tmp/availability.html \
  && (source ~/.config/research-publish.env \
      && rsync -e /usr/bin/ssh --chmod=F644 --no-owner --no-group \
           /tmp/availability.html "${VPS_HOST}:${VPS_DATA_DIR}/availability.html")
```

Same destination convention as `costs.html` in the dashboard repo's `research-publish.sh`.
Like costs.html, the container Caddy needs an explicit handle route for `/availability.html`
or the SPA answers instead. If the log lives on another machine, rsync the TSV over and pass
`--log`.
