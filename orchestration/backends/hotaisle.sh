#!/usr/bin/env bash
# HOTAISLE backend — run a campaign on a Hot Aisle cloud MI300X.
# Cloud lifecycle only: provision → warm (clone repo + instantiate + write a VM-local config.env) →
# run the campaign ON THE VM via the shared local backend (rocm) → download reduced products → KEEP
# the VM warm. Then `teardown` (cost-safe). Cell execution, tagging, logging, cube policy all come
# from the SAME run_cell.sh the VM clones — so this file holds only the cloud wrapper, no run logic.
#
#   bash orchestration/backends/hotaisle.sh run orchestration/campaigns/<campaign>.sh
#   bash orchestration/backends/hotaisle.sh teardown
#
# `run` REUSES a kept-warm, reachable VM if one is recorded (else provisions + warms a new one), so
# the proven smoke→real flow shares ONE paid VM:
#   hotaisle.sh run campaigns/smoke.sh   # provision+warm+smoke, KEEP warm — eyeball products
#   hotaisle.sh run campaigns/lowa0_maps.sh   # reuses the SAME warm VM (no re-provision/warm)
#   hotaisle.sh teardown                 # cost-safe delete (waits the reservation minimum)
#
# config.env: HOTAISLE_TEAM, HOTAISLE_GPUS, HOTAISLE_REPO_URL, HOTAISLE_BRANCH (, HOTAISLE_API).
# Secrets stay external: API token at ~/.config/hotaisle/token; ntfy creds (driving-side) via NTFY_ENV.
# BILLING: 1 GPU = per-minute (1-min minimum); 2/4 carry 60/120-min minimums. Teardown waits out the
# minimum, then a non-force DELETE — ALWAYS verify the VM is gone in the TUI (ssh admin.hotaisle.app);
# the API DELETE is best-effort and billing only truly stops on TUI destroy.
set -Eeuo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."; ORCH="$(cd "$ORCH" && pwd)"
. "$ORCH/run_cell.sh"        # config.env + notify() (notifications fire from THIS driving machine)

MODE="${1:?usage: hotaisle.sh run <campaign.sh> | teardown}"
TOK=$(cat "${HOTAISLE_TOKEN_FILE:-$HOME/.config/hotaisle/token}")
TEAM="${HOTAISLE_TEAM:?set HOTAISLE_TEAM in config.env}"
API="${HOTAISLE_API:-https://admin.hotaisle.app/api/teams}/$TEAM"
GPUS="${HOTAISLE_GPUS:-1}"
REPO_URL="${HOTAISLE_REPO_URL:?set HOTAISLE_REPO_URL in config.env}"
BRANCH="${HOTAISLE_BRANCH:-main}"
STATE="${HOTAISLE_STATE:-$HOME/.config/hotaisle/campaign_vm}"
OUT="${HOTAISLE_OUT:-$HOME/campaign_out}"
DEPOT_CACHE="${DEPOT_CACHE:-}"; DEPOT_CACHE_KEY="${DEPOT_CACHE_KEY:-$HOME/.config/runpod/depot_key}"   # shared cache (see depot_cache.sh)
CM="$HOME/.ssh/cm-hotaisle.sock"
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"

api()       { curl -fsS -X "$1" -H "Authorization: Token $TOK" "${@:3}" "$API$2"; }
ssh_vm()    { /usr/bin/ssh $SSHOPTS hotaisle@"$IP" "$@"; }
log()       { echo "[$(date -u +%FT%TZ)] $*"; }
balance()   { api GET /balance/ | jq -r '.available_balance/100'; }
min_resv()  { api GET /virtual_machines/available/ | jq -r --argjson g "$GPUS" \
              '[.[]|select(.Specs.gpus[0].model=="MI300X" and .Specs.gpus[0].count==$g)|.MinimumReservationMinutes][0]//1'; }

provision() {
    log "provisioning ${GPUS}×MI300X (balance \$$(balance))…"
    local vm; vm=$(api POST /virtual_machines/ -H "Content-Type: application/json" \
        --data-binary "{\"gpus\":[{\"model\":\"MI300X\",\"count\":$GPUS}]}")
    NAME=$(echo "$vm"|jq -r .name); IP=$(echo "$vm"|jq -r .ssh_access.ip_address); PROV_TS=$(date +%s)
    log "provisioned $NAME ($IP)"
}
wait_ssh() { log "waiting for ssh…"; local i; for i in $(seq 1 40); do ssh_vm true 2>/dev/null && return 0; sleep 15; done; return 1; }
warm() {
    log "warm: julia + clone $BRANCH + instantiate (depot cache: ${DEPOT_CACHE:-none})"
    if [ -n "$DEPOT_CACHE" ]; then
        if [ -f "$DEPOT_CACHE_KEY" ]; then   # jailed key — the VM can ONLY rsync inside the archive store
            ssh_vm 'mkdir -p ~/.ssh && cat > ~/.ssh/depot_key && chmod 600 ~/.ssh/depot_key' < "$DEPOT_CACHE_KEY"
        else
            log "[warn] DEPOT_CACHE set but $DEPOT_CACHE_KEY missing — building the depot fresh"
            DEPOT_CACHE=""
        fi
    fi
    ssh_vm "REPO_URL='$REPO_URL' BRANCH='$BRANCH' DEPOT_CACHE='$DEPOT_CACHE' bash -s" <<'WARM'
set -e
[ -x "$HOME/.juliaup/bin/julia" ] || curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"   # no JULIA_PKG_SERVER: the VM uses Julia's public default (registries also ride the depot cache)
rm -rf EDM && git clone --quiet --branch "$BRANCH" "$REPO_URL" EDM
# VM-local config.env (gitignored ⇒ not cloned): rocm local backend, no ntfy on the VM, and
# REDUCE_OVERLAP=1 so each cell's reduction overlaps the next cell's GPU compute (paid-time win).
printf 'LOCAL_BACKEND=rocm\nLOCAL_JL_THREADS=auto\nLOCAL_PREENV=\nREDUCE_OVERLAP=1\n' > EDM/orchestration/config.env
BK=rocm REPO_DIR="$HOME/EDM"; . EDM/orchestration/depot_cache.sh   # julia-actions/cache semantics (default ~/.julia depot)
depot_cache_restore   # → DC_RESTORED = exact | prefix (instantiate tops it up) | miss (fresh build)
ok=0; for i in 1 2 3; do
  if julia --startup=no --project=EDM/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then ok=1; break; fi
  echo "instantiate retry $i"; sleep 20
done; [ "$ok" = 1 ] || { echo "instantiate failed"; exit 1; }
[ "$DC_RESTORED" = exact ] || depot_cache_push   # exact hit ⇒ the store already has this depot
WARM
}

# Is there a kept-warm VM we can reuse? (lets smoke→real, or several campaigns, share ONE paid VM
# instead of re-provisioning + re-warming — ~8 min of paid clone/instantiate — every time.)
vm_reachable() { [ -f "$STATE" ] && read -r NAME IP PROV_TS MIN_MIN < "$STATE" && [ -n "${IP:-}" ] && ssh_vm true 2>/dev/null; }

# Overlay the driver's current orchestration/ onto the VM clone (keeping the VM's own config.env),
# so the VM runs THIS machine's framework + campaign files — even ones not yet pushed to BRANCH.
push_orchestration() {
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS" --exclude='config.env' "$ORCH/" hotaisle@"$IP":EDM/orchestration/
}

# Integrity-check the downloaded reduced products: md5 each non-cube file VM-vs-local; alert on any
# mismatch/missing. Cheap insurance against a truncated rsync silently corrupting the physics.
download_verify() {
    local dst=$1 bad=0 vsum fn lsum
    while read -r vsum fn; do
        [ -n "$vsum" ] || continue
        fn=$(basename "$fn")
        if [ ! -f "$dst/$fn" ]; then log "[verify] MISSING locally: $fn"; bad=1; continue; fi
        lsum=$(md5sum "$dst/$fn" | awk '{print $1}')
        [ "$lsum" = "$vsum" ] || { log "[verify] MD5 MISMATCH: $fn (vm=$vsum local=$lsum)"; bad=1; }
    done < <(ssh_vm "cd EDM/runs/$CAMPAIGN && md5sum * 2>/dev/null | grep -vE 'field_.*\\.jls'")
    [ "$bad" -eq 0 ] && { log "[verify] $CAMPAIGN OK (md5 match; cubes excluded)"; return 0; }
    notify rotating_light high "EDM download CHECK FAILED" "$CAMPAIGN: md5 mismatch/missing on download to $dst"
    return 1
}

run_campaign() {
    local cf="${1:?usage: hotaisle.sh run <campaign.sh>}" cname; cname=$(basename "$cf")
    . "$cf"   # CAMPAIGN (for product paths); the same file is re-read on the VM
    if vm_reachable; then
        log "reusing kept-warm VM $NAME ($IP) from $STATE (no provision/warm)"
    else
        trap 'rc=$?; log "FAILED (rc=$rc) before VM was handed off"; notify rotating_light urgent "EDM hotaisle FAILED" "$CAMPAIGN setup errored (rc=$rc); tearing down"; teardown; exit $rc' ERR
        provision; wait_ssh; warm
        echo "$NAME $IP $PROV_TS $(min_resv)" > "$STATE"
        trap - ERR    # VM is up + warm; a campaign/download hiccup below must NOT auto-destroy it
    fi
    push_orchestration   # VM runs the driver's framework + this campaign file (works pre-merge too)
    notify hourglass_flowing_sand default "EDM hotaisle started" "$CAMPAIGN on $NAME (${GPUS}×MI300X)"
    log "launching $CAMPAIGN on the VM via the local backend (rocm), detached…"
    # nohup + a pidfile (not setsid): keeps the driver's PID so we can tell "still running" from "crashed".
    ssh_vm "cd EDM && mkdir -p runs && nohup bash orchestration/backends/local.sh orchestration/campaigns/$cname > runs/${CAMPAIGN}.out 2>&1 < /dev/null & echo \$! > runs/${CAMPAIGN}.pid"
    log "polling for completion (DONE marker, or driver-death = crash so we don't poll forever while billing)…"
    until ssh_vm "grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null"; do
        if ! ssh_vm "kill -0 \$(cat runs/${CAMPAIGN}.pid 2>/dev/null) 2>/dev/null"; then
            notify rotating_light urgent "EDM hotaisle CRASH" "$CAMPAIGN driver died with no DONE on $NAME — VM KEPT for inspection; teardown when done (verify in TUI)."
            log "[ERROR] driver process gone, no DONE marker — likely crashed. VM $NAME KEPT. tail:"; ssh_vm "tail -n 20 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[vm] /'
            return 1
        fi
        sleep 60; ssh_vm "tail -n1 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[vm] /'
    done
    log "campaign done; downloading reduced products (cubes excluded)…"
    mkdir -p "$OUT/$CAMPAIGN"
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS" --exclude='field_*.jls' \
        hotaisle@"$IP":"EDM/runs/$CAMPAIGN/" "$OUT/$CAMPAIGN/"
    download_verify "$OUT/$CAMPAIGN" || log "[verify] integrity problems — products also remain on VM:EDM/runs/$CAMPAIGN"
    notify white_check_mark default "EDM hotaisle done" "$CAMPAIGN → $OUT/$CAMPAIGN ; VM $NAME KEPT WARM — run teardown."
    log "products → $OUT/$CAMPAIGN ; VM $NAME KEPT WARM (state $STATE). Add more: $0 run <campaign>. Finish: $0 teardown"
}

teardown() {
    [ -f "$STATE" ] || { echo "no kept VM recorded in $STATE"; exit 0; }
    read -r NAME IP PROV_TS MIN_MIN < "$STATE"
    local min_s=$(( ${MIN_MIN:-1} * 60 )) el=$(( $(date +%s) - ${PROV_TS:-$(date +%s)} ))
    if [ "$el" -lt "$min_s" ]; then
        log "waiting $(( (min_s - el + 59) / 60 )) min to reach the ${MIN_MIN}-min reservation minimum (billed either way; Ctrl-C to delete now)"
        sleep $(( min_s - el ))
    fi
    /usr/bin/ssh -O exit -o ControlPath="$CM" hotaisle@"$IP" 2>/dev/null || true
    if api DELETE "/virtual_machines/$NAME/" >/dev/null 2>&1; then
        rm -f "$STATE"
        log "API delete of $NAME accepted; balance \$$(balance). VERIFY it's gone in the TUI (ssh admin.hotaisle.app) — billing stops on TUI destroy."
        notify checkered_flag default "EDM hotaisle torn down" "$NAME deleted; balance \$$(balance). Verify in TUI."
    else
        notify rotating_light urgent "EDM teardown FAILED" "API DELETE of $NAME errored — VM likely STILL UP + billing. Destroy it in the TUI now (ssh admin.hotaisle.app)."
        log "[ERROR] API DELETE of $NAME FAILED — VM likely STILL UP + billing. DESTROY IN THE TUI. state kept at $STATE."
        exit 1
    fi
}

case "$MODE" in
    run)      shift; run_campaign "${1:-}" ;;
    teardown) teardown ;;
    *) echo "usage: $0 run <campaign.sh> | teardown" >&2; exit 64 ;;
esac
