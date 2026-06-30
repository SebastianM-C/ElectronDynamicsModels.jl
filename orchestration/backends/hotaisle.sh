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
    log "warm: julia + clone $BRANCH + instantiate (registry from GitHub)"
    ssh_vm "REPO_URL='$REPO_URL' BRANCH='$BRANCH' bash -s" <<'WARM'
set -e
[ -x "$HOME/.juliaup/bin/julia" ] || curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH" JULIA_PKG_SERVER=""
rm -rf EDM && git clone --quiet --branch "$BRANCH" "$REPO_URL" EDM
# VM-local config.env (gitignored ⇒ not cloned): rocm local backend, no ntfy on the VM.
printf 'LOCAL_BACKEND=rocm\nLOCAL_JL_THREADS=auto\nLOCAL_PREENV=\n' > EDM/orchestration/config.env
ok=0; for i in 1 2 3; do
  if julia --startup=no --project=EDM/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then ok=1; break; fi
  echo "instantiate retry $i"; sleep 20
done; [ "$ok" = 1 ] || { echo "instantiate failed"; exit 1; }
WARM
}

run_campaign() {
    local cf="${1:?usage: hotaisle.sh run <campaign.sh>}" cname; cname=$(basename "$cf")
    . "$cf"   # CAMPAIGN (for product paths); the same file is read again on the VM
    trap 'rc=$?; log "FAILED (rc=$rc) before VM was handed off"; notify rotating_light urgent "EDM hotaisle FAILED" "$CAMPAIGN setup errored (rc=$rc); tearing down"; teardown; exit $rc' ERR
    provision; wait_ssh; warm
    echo "$NAME $IP $PROV_TS $(min_resv)" > "$STATE"
    trap - ERR    # VM is up + warm; a campaign/download hiccup below must NOT auto-destroy it
    notify hourglass_flowing_sand default "EDM hotaisle started" "$CAMPAIGN on $NAME (${GPUS}×MI300X)"
    log "launching $CAMPAIGN on the VM via the local backend (rocm), detached…"
    ssh_vm "cd EDM && setsid nohup bash orchestration/backends/local.sh orchestration/campaigns/$cname > runs/${CAMPAIGN}.out 2>&1 < /dev/null & echo launched"
    log "polling for completion…"
    until ssh_vm "grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null"; do
        sleep 60; ssh_vm "tail -n1 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[vm] /'
    done
    log "campaign done; downloading reduced products (cubes excluded)…"
    mkdir -p "$OUT/$CAMPAIGN"
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS" --exclude='field_*.jls' \
        hotaisle@"$IP":"EDM/runs/$CAMPAIGN/" "$OUT/$CAMPAIGN/"
    notify white_check_mark default "EDM hotaisle done" "$CAMPAIGN → $OUT/$CAMPAIGN ; VM $NAME KEPT WARM — run teardown."
    log "products → $OUT/$CAMPAIGN ; VM $NAME KEPT WARM (state $STATE). Finish: $0 teardown"
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
