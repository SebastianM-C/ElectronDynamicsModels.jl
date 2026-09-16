#!/usr/bin/env bash
# VERDA backend — run a campaign on a Verda Cloud (ex-DataCrunch) GPU VM. Mirrors runpod.sh's
# lifecycle (lanes, attach, teardown, R2 drainer, cube gate, ledger); cell execution, tagging,
# logging and cube policy all come from the SAME run_cell.sh the VM clones.
#
#   bash orchestration/backends/verda.sh run <campaign.sh>...     provision → warm → run → download
#   bash orchestration/backends/verda.sh attach <campaign.sh>...  re-attach to a kept VM's campaign
#                                                                 (monitor + download, NO relaunch)
#   bash orchestration/backends/verda.sh teardown                 delete the kept VM (stops billing)
#
# Lanes (multi-GPU types, e.g. 8B300.240V): several campaign files in one `run` are launched
# CONCURRENTLY on the VM, one sequential local.sh lane each — same contract as runpod.sh
# (a lane owns runs/<file-stem>.out/.pid; cells pin devices with CUDA_VISIBLE_DEVICES=…).
#
# Verda specifics (API mapped 2026-09-16 from https://api.verda.com/v1/openapi.json):
#   • full KVM VMs with root ssh on an Ubuntu+NVIDIA-driver image — no container/sshd bootstrap
#   • OAuth2 client-credentials (1 h tokens) read from the SAME ~/.verda/credentials the `verda`
#     CLI and its MCP server write (`verda auth login`); VERDA_CLIENT_ID/SECRET env override
#   • capacity: GET /instance-availability/<type>?location_code= is a free boolean and POST
#     /instances answers 503 when the type has no capacity ⇒ provision() polls a type×location ladder
#   • the OS volume IS the disk ("dynamic" storage): size it for the campaign (VERDA_OS_VOLUME_GB,
#     $0.20/GiB-month ⇒ 200 GiB ≈ 5.5 ¢/h); `delete` removes the OS volume by default (no leak)
#   • billing in prepaid 10-min increments (unused remainder refunded); spot = 50 % list, evicted
#     without warning (VERDA_SPOT=1 — OS volume → trash on eviction, salvageable for 96 h)
#   • NVIDIA-only catalog (B300/B200/H200/H100/…) ⇒ BACKEND=cuda; depot cache keyed cuda+host CPU
#
# config.env: VERDA_TYPES (ladder override), VERDA_LOCATIONS, VERDA_IMAGE, VERDA_OS_VOLUME_GB,
# VERDA_SPOT, VERDA_REPO_URL/BRANCH, VERDA_SSH_PUBKEY (+ optional VERDA_SSH_KEY for headless
# drivers), DEPOT_CACHE/DEPOT_CACHE_KEY. Secrets external: ~/.verda/credentials; ntfy via NTFY_ENV.
#
# VM hooks (env or config.env; paths relative to orchestration/, which push_orchestration ships):
#   VERDA_VM_PRELUDE=<script>   run as root on the VM after warm, BEFORE any lane launches, with the
#                               campaign dir as $1 (e.g. vm_prelude_nvidia_ncu.sh: an ncu that knows
#                               the chip). Non-zero exit: VERDA_PRELUDE_FAIL=keep (default) launches
#                               anyway, =teardown deletes the VM (stops billing) and exits 6.
#   VERDA_VM_POSTLUDE=<script>  run after every lane is DONE, before the download, same $1 (e.g.
#                               vm_postlude_gpudiag_suite.sh). Its status is reported, never fatal.
# Both log to <campaign dir>/<hook>.log on the VM, so the logs ride home with the products.
set -Eeuo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."; ORCH="$(cd "$ORCH" && pwd)"
# Caller-env overrides must survive config.env (sourced via run_cell.sh) — capture before, prefer after.
_CALLER_BRANCH="${VERDA_BRANCH-}"; _CALLER_TYPES="${VERDA_TYPES-}"; _CALLER_LOCS="${VERDA_LOCATIONS-}"
_CALLER_OSGB="${VERDA_OS_VOLUME_GB-}"; _CALLER_SPOT="${VERDA_SPOT-}"; _CALLER_IMAGE="${VERDA_IMAGE-}"
. "$ORCH/run_cell.sh"        # config.env + notify() (notifications fire from THIS driving machine)
[ -n "$_CALLER_BRANCH" ] && VERDA_BRANCH="$_CALLER_BRANCH"
[ -n "$_CALLER_TYPES" ] && VERDA_TYPES="$_CALLER_TYPES"
[ -n "$_CALLER_LOCS" ] && VERDA_LOCATIONS="$_CALLER_LOCS"
[ -n "$_CALLER_OSGB" ] && VERDA_OS_VOLUME_GB="$_CALLER_OSGB"
[ -n "$_CALLER_SPOT" ] && VERDA_SPOT="$_CALLER_SPOT"
[ -n "$_CALLER_IMAGE" ] && VERDA_IMAGE="$_CALLER_IMAGE"

MODE="${1:?usage: verda.sh run <campaign.sh>... | attach <campaign.sh>... | teardown}"
API="${VERDA_API:-https://api.verda.com/v1}"
# Blackwell needs the open kernel modules; CUDA.jl ships its own runtime, only the driver matters.
IMAGE="${VERDA_IMAGE:-ubuntu-24.04-cuda-13.0-open}"
OSGB="${VERDA_OS_VOLUME_GB:-200}"
SPOT="${VERDA_SPOT:-0}"; [[ "$SPOT" =~ ^[01]$ ]] || { echo "VERDA_SPOT must be 0 or 1, got '$SPOT'" >&2; exit 64; }
REPO_URL="${VERDA_REPO_URL:?set VERDA_REPO_URL in config.env}"; BRANCH="${VERDA_BRANCH:-main}"
DEPOT_CACHE="${DEPOT_CACHE:-}"; DEPOT_CACHE_KEY="${DEPOT_CACHE_KEY:-$HOME/.config/runpod/depot_key}"   # shared cache (see depot_cache.sh)
STATE="${VERDA_STATE:-$HOME/.config/verda/campaign_vm}"; OUT="${VERDA_OUT:-$HOME/campaign_out}"
POLL="${VERDA_POLL_SEC:-120}"; MAXTRIES="${VERDA_MAX_TRIES:-240}"
PUBKEY="$(cat "${VERDA_SSH_PUBKEY:-$HOME/.config/verda/ssh_pubkey}" 2>/dev/null || ssh-add -L 2>/dev/null | head -1)"
CM="$HOME/.ssh/cm-verda-$(basename "$STATE").sock"   # per-STATE: concurrent drivers must not remux onto one socket
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600${VERDA_SSH_KEY:+ -i $VERDA_SSH_KEY -o IdentitiesOnly=yes}"
SSH_WAIT_TRIES="${VERDA_SSH_WAIT_TRIES:-40}"

log()     { echo "[$(date -u +%FT%TZ)] $*"; }
ssh_vm()  { /usr/bin/ssh $SSHOPTS root@"$IP" "$@"; }

# ── auth ─────────────────────────────────────────────────────────────────────
# Client id/secret: env, else the CLI's AWS-style INI (~/.verda/credentials, profile VERDA_PROFILE).
# Exported so jq reads them from env (never argv). Tokens last 3600 s; refreshed after 50 min.
# Errors go to STDERR: these run inside $(vapi … | jq) pipelines, where stdout is jq's input.
creds() {
    [ -n "${VERDA_CLIENT_ID:-}" ] && [ -n "${VERDA_CLIENT_SECRET:-}" ] && { export VERDA_CLIENT_ID VERDA_CLIENT_SECRET; return 0; }
    local f="${VERDA_CREDENTIALS_FILE:-${VERDA_SHARED_CREDENTIALS_FILE:-$HOME/.verda/credentials}}" p="${VERDA_PROFILE:-default}"
    [ -f "$f" ] || { log "[ERROR] no Verda credentials — run 'verda auth login' (writes $f) or export VERDA_CLIENT_ID/VERDA_CLIENT_SECRET" >&2; return 1; }
    VERDA_CLIENT_ID=$(awk -v p="[$p]" '$0==p{s=1;next} /^\[/{s=0} s && $1=="verda_client_id"{print $3}' "$f")
    VERDA_CLIENT_SECRET=$(awk -v p="[$p]" '$0==p{s=1;next} /^\[/{s=0} s && $1=="verda_client_secret"{print $3}' "$f")
    [ -n "$VERDA_CLIENT_ID" ] && [ -n "$VERDA_CLIENT_SECRET" ] || { log "[ERROR] profile [$p] in $f lacks verda_client_id/secret" >&2; return 1; }
    export VERDA_CLIENT_ID VERDA_CLIENT_SECRET
}
TOK=""; TOK_TS=0
vtoken() {
    local now; now=$(date +%s)
    [ -n "$TOK" ] && [ $((now - TOK_TS)) -lt 3000 ] && return 0
    creds || return 1
    TOK=$(curl -fsS -X POST "$API/oauth2/token" -H "Content-Type: application/json" \
        -d "$(jq -n '{grant_type:"client_credentials",client_id:env.VERDA_CLIENT_ID,client_secret:env.VERDA_CLIENT_SECRET}')" \
        | jq -r '.access_token // empty') || TOK=""
    [ -n "$TOK" ] || { log "[ERROR] Verda token request failed (bad client id/secret?)" >&2; return 1; }
    TOK_TS=$now
}
vapi()    { vtoken && curl -fsS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" "$@"; }

# Persistent cost ledger (shared with hotaisle.sh/runpod.sh; reported by the private dashboard
# repo's cost_report.sh). rate_cents_h = the VM's price_per_hour in cents at provision (list
# prices drift); balance_usd = GET /balance .amount. A ledger write must NEVER break a campaign.
LEDGER="${EDM_CLOUD_LEDGER:-$HOME/.config/edm-cloud-ledger.tsv}"
ledger()  {   # ledger <vm> <event> <detail> [rate_cents_h] [balance_usd]
    { mkdir -p "$(dirname "$LEDGER")"
      [ -f "$LEDGER" ] || printf 'ts_utc\tprovider\tvm\tevent\tdetail\trate_cents_h\tbalance_usd\n' > "$LEDGER"
      printf '%s\tverda\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-}" "${5:-}" >> "$LEDGER"
    } 2>/dev/null || true
}
vm_balance() { vapi --max-time 5 "$API/balance" 2>/dev/null | jq -r '.amount // empty' 2>/dev/null || true; }

# ── GPU acquisition policy ───────────────────────────────────────────────────
# Verda's catalog is NVIDIA-only; every type maps to the cuda backend (a future AMD listing
# slots in here — the depot cache is keyed by $BACKEND).
gpu_profile() { case "$1" in *MI3*) echo rocm ;; *) echo cuda ;; esac; }

# Echo the instance types to attempt, ONE PER LINE, in priority order. provision() tries each
# type × location every poll round; the first with capacity wins. Catalog (GET /instance-types,
# public, 2026-09-16; on-demand / spot $/h per VM): 1B300.30V 7.73/3.86 (268 GB), 1B200.30V
# 6.36/3.18 (180 GB), 1H200.141S.44V 4.28/2.14 (141 GB), 1H100.80S.30V 3.28/1.64 (80 GB);
# multi-GPU = <n>B300.<cpus>V etc. `verda instance-types` prints the live list.
gpu_candidates() {
    # VERDA_TYPES (comma-separated instance_type ids) overrides the ladder —
    # e.g. pin 1B300.30V-only when the campaign is a per-card benchmark.
    if [ -n "${VERDA_TYPES:-}" ]; then
        echo "$VERDA_TYPES" | tr ',' '\n' | sed 's/^ *//; s/ *$//' | grep -v '^$'
        return
    fi
    # Default ladder: Blackwell only, so an unpinned run never silently becomes a Hopper run
    # (H200/H100 are pinned explicitly via VERDA_TYPES when a campaign wants them).
    echo 1B300.30V    # the card RunPod could not deliver — why this backend exists
    echo 1B200.30V    # same Blackwell FP64 class at 82 % of the price
}

locations() {   # VERDA_LOCATIONS (comma) else every location the API lists, in API order
    if [ -n "${VERDA_LOCATIONS:-}" ]; then echo "$VERDA_LOCATIONS" | tr ',' '\n' | sed 's/ //g' | grep -v '^$'; return; fi
    vapi "$API/locations" | jq -r '.[].code'
}
available() {   # free boolean; a miss here saves a doomed create
    vapi "$API/instance-availability/$1?location_code=$2&is_spot=$([ "$SPOT" = 1 ] && echo true || echo false)" 2>/dev/null | grep -q true
}

ensure_ssh_key() {   # the driver's pubkey must be an ACCOUNT key (ssh_key_ids at create) — match by type+blob, else add
    [ -n "$PUBKEY" ] || { log "[ERROR] no SSH pubkey — set VERDA_SSH_PUBKEY (or load one in ssh-agent)"; return 1; }
    local blob; blob=$(echo "$PUBKEY" | awk '{print $1" "$2}')
    KEYID=$(vapi "$API/ssh-keys" | jq -r --arg k "$blob" '.[] | select(((.key|split(" "))[0:2]|join(" "))==$k) | .id' | head -1)
    [ -n "$KEYID" ] && { log "ssh key on account: $KEYID"; return 0; }
    KEYID=$(vapi -X POST "$API/ssh-keys" -d "$(jq -n --arg n "edm-driver-$(hostname -s)" --arg k "$PUBKEY" '{name:$n,key:$k}')" \
        | jq -r 'if type=="object" then (.id // empty) else . end')
    [ -n "$KEYID" ] || { log "[ERROR] adding the ssh key to the Verda account failed"; return 1; }
    log "added ssh key edm-driver-$(hostname -s) → $KEYID"
}

# Catalog lookup (GET /instance-types is PUBLIC): "<model> <n>" for an instance_type, e.g.
# "B300 1" — the ledger's gpu=<model> gpus=<n> tokens, in the shape the dashboard's
# cost/availability reporters already parse (model = the watcher's lane label).
type_model() { curl -fsS --max-time 10 "$API/instance-types" 2>/dev/null \
    | jq -r --arg t "$1" '.[] | select(.instance_type==$t) | "\(.model) \(.gpu.number_of_gpus)"' 2>/dev/null | head -1; }
vm_json()   { vapi "$API/instances/$VM" 2>/dev/null; }
vm_status() { vm_json | jq -r '.status // "unknown"' 2>/dev/null || echo unknown; }

provision() {
    local -a cands locs; mapfile -t cands < <(gpu_candidates)
    [ "${#cands[@]}" -gt 0 ] || { log "[ERROR] gpu_candidates returned nothing (set VERDA_TYPES or fill the ladder)"; return 1; }
    ensure_ssh_key || return 1   # explicit: provision runs under `|| {…}`, so errexit is off in here
    mapfile -t locs < <(locations)
    [ "${#locs[@]}" -gt 0 ] || { log "[ERROR] no locations (API/creds?)"; return 1; }
    local host="edm-verda-$(date +%s)" spotj; spotj=$([ "$SPOT" = 1 ] && echo true || echo false)
    log "provisioning (poll ${POLL}s ≤$MAXTRIES tries, spot=$SPOT, os volume ${OSGB} GiB, image $IMAGE): ${cands[*]} @ ${locs[*]}"
    local try t l resp code body rate
    for ((try=1; try<=MAXTRIES; try++)); do
        for t in "${cands[@]}"; do
            for l in "${locs[@]}"; do
                available "$t" "$l" || continue
                vtoken || return 1
                # Not vapi (curl -f): keep the body AND status — 503 = raced out of capacity (poll on),
                # anything else is a real, explained rejection.
                resp=$(curl -sS -o - -w '\n%{http_code}' -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" "$API/instances" \
                    -d "$(jq -n --arg t "$t" --arg img "$IMAGE" --arg h "$host" --arg l "$l" --arg k "$KEYID" --argjson gb "$OSGB" --argjson spot "$spotj" \
                        '{instance_type:$t,image:$img,hostname:$h,description:"EDM campaign VM (orchestration/backends/verda.sh)",
                          location_code:$l,ssh_key_ids:[$k],is_spot:$spot,
                          os_volume:({name:($h+"-os"),size:$gb} + (if $spot then {on_spot_discontinue:"move_to_trash"} else {} end))}')") || resp=$'\n000'
                code=${resp##*$'\n'}; body=${resp%$'\n'*}
                case "$code" in
                    2*) VM=$(echo "$body" | jq -r 'if type=="string" then . else (.id // .instance_id // empty) end' 2>/dev/null) || VM=""
                        [ -n "$VM" ] || VM=$(echo "$body" | tr -d '"[:space:]')
                        # instance ids are UUIDs — anything else is a body we misparsed, never a ledger "vm"
                        [[ "$VM" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
                            || { log "create $t@$l: 2xx but no instance id in '$(echo "$body" | head -c 120)'"; VM=""; continue; } ;;
                    503) continue ;;
                    *)  log "create $t@$l rejected: HTTP $code $(echo "$body" | head -c 200)"; continue ;;
                esac
                BACKEND=$(gpu_profile "$t"); TYPE=$t; LOC=$l
                # Billing starts HERE — ledger + ping before anything else can fail.
                rate=$(vm_json | jq -r 'if .price_per_hour then (.price_per_hour*100|round) else empty end' 2>/dev/null) || rate=""
                local mn; mn=$(type_model "$t"); mn=${mn:-"$t 1"}
                ledger "$VM" provision "gpu=${mn% *} gpus=${mn##* } dc=$l type=$t spot=$SPOT os_gb=$OSGB image=$IMAGE" "$rate" "$(vm_balance)"
                log "provisioned $t @ $l → $VM ($BACKEND, ${rate:-?}¢/h, spot=$SPOT)"
                notify satellite default "EDM verda provisioned" "$t @ $l → $VM (try $try/$MAXTRIES, billing started)"
                return 0
            done
        done
        [ $(( (try-1) % 10 )) -eq 0 ] && log "  no capacity yet (try $try/$MAXTRIES)…"
        sleep "$POLL"
    done
    log "[ERROR] gave up after $MAXTRIES tries"; return 1
}

vm_shape() {   # what we actually got — log only, never fail
    vm_json | jq -r '"vm shape: \(.gpu.description // "?"), \(.cpu.description // "?"), \(.memory.description // "?"), \(.storage.description // "?"), \(.location // "?"), \(.os_name // .image // "?")"' 2>/dev/null \
        | sed 's/^/[vm] /' || true
}

wait_ready() {   # status → running + public ip, then sshd
    log "waiting for the VM to come up…"; local i st
    for i in $(seq 1 60); do
        st=$(vm_status)
        IP=$(vm_json | jq -r '.ip // empty' 2>/dev/null) || IP=""
        [ "$st" = running ] && [ -n "$IP" ] && { log "endpoint root@$IP ($st)"; vm_shape; break; }
        case "$st" in error|installation_failed|no_capacity|discontinued)
            log "[ERROR] instance $VM entered '$st' — deleting it"; vm_delete; return 1 ;; esac
        sleep 15
    done
    [ -n "${IP:-}" ] && [ "$st" = running ] || { log "[ERROR] VM not running after wait (status $st)"; return 1; }
    for i in $(seq 1 "$SSH_WAIT_TRIES"); do ssh_vm true 2>/dev/null && return 0; sleep 10; done
    log "[ERROR] sshd never answered after $SSH_WAIT_TRIES tries"; return 1
}

warm() {
    log "warm: clone $BRANCH + instantiate ($BACKEND depot on the OS volume; cache: ${DEPOT_CACHE:-none})"
    # Ubuntu cloud images run unattended-upgrades at first boot — wait for the dpkg lock
    # instead of failing the apt call; rsync + zstd serve the depot cache + product download.
    ssh_vm 'while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done
            command -v rsync >/dev/null && command -v zstd >/dev/null && command -v git >/dev/null \
              || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync zstd git; }'
    if [ -n "$DEPOT_CACHE" ]; then
        if [ -f "$DEPOT_CACHE_KEY" ]; then   # jailed key — the VM can ONLY rsync inside the archive store
            ssh_vm 'mkdir -p /root/.ssh && cat > /root/.ssh/depot_key && chmod 600 /root/.ssh/depot_key' < "$DEPOT_CACHE_KEY"
        else
            log "[warn] DEPOT_CACHE set but $DEPOT_CACHE_KEY missing — building the depot fresh"
            DEPOT_CACHE=""
        fi
    fi
    ssh_vm "REPO_URL='$REPO_URL' BRANCH='$BRANCH' BK='$BACKEND' DEPOT_CACHE='$DEPOT_CACHE' bash -s" <<'WARM'
set -e
export JULIA_DEPOT_PATH="/root/julia-depot-$BK"
[ -x "$HOME/.juliaup/bin/julia" ] || curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"   # no JULIA_PKG_SERVER: fresh VM uses Julia's public default
rm -rf ~/EDM && git clone --quiet --branch "$BRANCH" "$REPO_URL" ~/EDM   # fresh clone = always current
# Pin Julia to the version the tracked scripts manifest was resolved on (Manifest-v<ver>.toml) —
# same rule as runpod/hotaisle (PR #128): juliaup's `release` moved to 1.13 on 2026-09-16, and a
# fresh VM then resolved a NEW manifest, missed every depot archive and precompiled under v1.13.
JULIA_CHANNEL=$(ls EDM/scripts/Manifest-v*.toml 2>/dev/null | sed -E 's/.*Manifest-v([0-9]+\.[0-9]+)\.toml/\1/' | head -1)
JULIA_CHANNEL="${JULIA_CHANNEL:-release}"
juliaup add "$JULIA_CHANNEL" >/dev/null 2>&1 || true; juliaup default "$JULIA_CHANNEL"
mkdir -p ~/EDM/runs
# Orchestration lives OUTSIDE the repo (~/edm-orch, driver-pushed) so the clone stays
# pristine; EDM_REPO points run_cell back at the clone (config.env's documented override).
mkdir -p ~/edm-orch
printf 'LOCAL_BACKEND=%s\nLOCAL_JL_THREADS=auto\nLOCAL_PREENV=JULIA_DEPOT_PATH=%s\nREDUCE_OVERLAP=1\nLOCAL_CLOUD_PROVIDER=verda\nEDM_REPO=%s\nJULIA_CHANNEL=%s\n' \
    "$BK" "$JULIA_DEPOT_PATH" "$HOME/EDM" "$JULIA_CHANNEL" > ~/edm-orch/config.env
REPO_DIR=~/EDM; . ~/EDM/orchestration/depot_cache.sh   # julia-actions/cache semantics over the rsync store
depot_cache_restore   # → DC_RESTORED = exact | prefix (instantiate tops it up) | miss (fresh build)
ok=0; for i in 1 2 3; do
  if julia --startup=no --project=EDM/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then ok=1; break; fi
  echo "instantiate retry $i"; sleep 20
done; [ "$ok" = 1 ] || { echo "instantiate failed"; exit 1; }
[ "$DC_RESTORED" = exact ] || depot_cache_push   # exact hit ⇒ the store already has this depot
WARM
}

# Is a kept VM still reachable? (lets smoke→real / several campaigns share ONE paid VM.)
vm_reachable() {
    [ -f "$STATE" ] || return 1
    read -r VM IP BACKEND TYPE LOC < "$STATE" || return 1
    [ -n "${VM:-}" ] || return 1
    [ "$(vm_status)" = running ] || return 1
    ssh_vm true 2>/dev/null
}

push_orchestration() {   # the driver's orchestration/ → ~/edm-orch (outside the repo; VM's config.env kept)
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS" --exclude='config.env' "$ORCH/" root@"$IP":edm-orch/
}

download_verify() {   # md5 each non-cube product VM-vs-local (subdirs included); alert on mismatch/missing
    local dst=$1 bad=0 vsum fn lsum
    while read -r vsum fn; do
        [ -n "$vsum" ] || continue; fn=${fn#./}
        if [ ! -f "$dst/$fn" ]; then log "[verify] MISSING locally: $fn"; bad=1; continue; fi
        lsum=$(md5sum "$dst/$fn" | awk '{print $1}')
        [ "$lsum" = "$vsum" ] || { log "[verify] MD5 MISMATCH: $fn"; bad=1; }
    done < <(ssh_vm "cd EDM/runs/$CAMPAIGN && find . -type f ! -name 'field_*.jls' -exec md5sum {} +")
    [ "$bad" -eq 0 ] && { log "[verify] $CAMPAIGN OK (cubes excluded)"; return 0; }
    notify rotating_light high "EDM download CHECK FAILED" "$CAMPAIGN: md5 mismatch/missing → $dst"; return 1
}

# Cube drainer (R2): same contract as hotaisle/runpod — script + creds copied OUT of the clone,
# pgrep-guarded nohup start, only when the campaign sets KEEP_CUBE=1.
start_drainer() {
    [ "${KEEP_CUBE:-0}" = 1 ] || return 0
    local envf="${CUBE_R2_ENV:-$HOME/.config/edm-r2.env}"
    [ -f "$envf" ] || { log "[drain] $envf missing — no R2 creds to ship"; return 1; }
    ssh_vm 'mkdir -p ~/.config && cat > ~/.config/edm-r2.env && chmod 600 ~/.config/edm-r2.env' < "$envf" || return 1
    ssh_vm 'cat > ~/cube_drain_r2.sh' < "$ORCH/cube_drain_r2.sh" || return 1
    # VERDA_DRAIN_DELETE_LOCAL=1 frees each cube once its upload + sha sidecar are in the bucket
    # (the OS volume is the only disk — size VERDA_OS_VOLUME_GB for the campaign or turn this on).
    if ! drainer_active; then
        ssh_vm "DRAIN_DELETE_LOCAL='${VERDA_DRAIN_DELETE_LOCAL:-0}' nohup bash ~/cube_drain_r2.sh >> ~/drain_r2.log 2>&1 < /dev/null &" || return 1
    fi
    log "[drain] cube_drain_r2.sh running on the VM (log: ~/drain_r2.log)"
}
drainer_active() { ssh_vm "pgrep -f '[c]ube_drain_r2.sh' >/dev/null" 2>/dev/null; }

# Cube-safety gate: delete destroys the OS volume — the ONLY copy of an undrained cube.
cube_gate() {
    local pending rc=0
    pending=$(ssh_vm 'bash -s' 2>/dev/null <<'GATE'
for cube in "$HOME"/EDM/runs/*/field_*.jls; do
    [ -e "$cube" ] || continue
    dir=$(dirname "$cube"); camp=$(basename "$dir"); base=$(basename "$cube")
    uuid=${base%.jls}; uuid=${uuid##*_}
    [ "$camp" = smoke ] && continue
    [ -e "$dir/$uuid.reduced" ] || continue
    [ -e "$dir/.drained_$base" ] || echo "$camp/$base"
done
GATE
    ) || rc=$?
    if [ "$rc" -ne 0 ]; then
        log "[gate] cube check failed (ssh rc=$rc) — cannot verify drains; proceeding (nothing to save over dead ssh)"
        return 0
    fi
    [ -z "$pending" ] && return 0
    if [ "${FORCE_TEARDOWN:-0}" = 1 ]; then
        log "[gate] FORCE_TEARDOWN=1 — deleting despite undrained cubes:"; echo "$pending" | sed 's/^/  /'
        return 0
    fi
    log "[gate] REFUSING teardown — undrained cubes on $VM (the VM holds the only copy):"
    echo "$pending" | sed 's/^/  /'
    log "[gate] wait for the drainer (tail ~/drain_r2.log on the VM), or override: FORCE_TEARDOWN=1 $0 teardown"
    notify rotating_light urgent "EDM teardown BLOCKED" "vm $VM: undrained cubes — $(echo "$pending" | tr '\n' ' '); drainer still working? FORCE_TEARDOWN=1 to override."
    exit 1
}

# ── lanes (same contract as runpod.sh) ───────────────────────────────────────
declare -a LANE_FILE LANE_STEM LANE_CAMP LANE_KEEP
load_lanes() {
    [ "$#" -ge 1 ] || { echo "usage: $0 $MODE <campaign.sh>..." >&2; exit 64; }
    local cf c k
    LANE_FILE=(); LANE_STEM=(); LANE_CAMP=(); LANE_KEEP=()
    for cf in "$@"; do
        [ -f "$cf" ] || { log "[ERROR] campaign file not found: $cf"; exit 64; }
        read -r c k < <( . "$cf" && echo "${CAMPAIGN:?$cf sets no CAMPAIGN} ${KEEP_CUBE:-0}" ) \
            || { log "[ERROR] cannot source $cf (or it sets no CAMPAIGN)"; exit 64; }
        LANE_FILE+=("$cf"); LANE_STEM+=("$(basename "$cf" .sh)"); LANE_CAMP+=("$c"); LANE_KEEP+=("$k")
    done
    CAMPAIGN=$(printf '%s\n' "${LANE_CAMP[@]}" | sort -u | paste -sd+)
    KEEP_CUBE=0; for k in "${LANE_KEEP[@]}"; do [ "$k" = 1 ] && KEEP_CUBE=1; done
    LANES=$(printf '%s\n' "${LANE_STEM[@]}" | paste -sd,)
}
lane_done()  { ssh_vm "cd EDM && grep -q '\\] ${LANE_CAMP[$1]} DONE' runs/${LANE_STEM[$1]}.out 2>/dev/null"; }
lane_alive() { ssh_vm "cd EDM && kill -0 \$(cat runs/${LANE_STEM[$1]}.pid 2>/dev/null) 2>/dev/null"; }
lane_tail()  { ssh_vm "cd EDM && tail -n1 runs/${LANE_STEM[$1]}.out 2>/dev/null" | sed "s/^/[vm ${LANE_STEM[$1]}] /" || true; }

launch_lane() {
    local i=$1 stem=${LANE_STEM[$1]} cname; cname=$(basename "${LANE_FILE[$i]}")
    if lane_done "$i"; then
        log "$stem already DONE on the VM — skipping launch (rm runs/$stem.out on the VM to force a rerun)"
    elif lane_alive "$i"; then
        log "$stem already RUNNING on the VM — monitoring it (no relaunch)"
    else
        log "launching lane $stem (→ runs/${LANE_CAMP[$i]}) via the local backend ($BACKEND), detached…"
        ssh_vm "export PATH=\"\$HOME/.juliaup/bin:\$PATH\"; cd EDM && mkdir -p runs/${LANE_CAMP[$i]} && { [ -f runs/${LANE_CAMP[$i]}/cells.tsv ] || printf 'label\\tuuid\\tscript\\tbackend\\toverrides\\n' > runs/${LANE_CAMP[$i]}/cells.tsv; } && rm -f runs/$stem.out runs/$stem.pid && { DIAG_CAMPAIGN='${DIAG_CAMPAIGN:-}' nohup bash \$HOME/edm-orch/backends/local.sh \$HOME/edm-orch/campaigns/$cname > runs/$stem.out 2>&1 < /dev/null & echo \$! > runs/$stem.pid; }"
    fi
}

download_campaign() {   # rsync one campaign dir off the VM (+ md5 verify, + driver-side publish)
    local camp=$1 keep=$2
    log "downloading $camp via rsync…"
    mkdir -p "$OUT/$camp"
    local -a excl=(--include='*_obscache.jls' --exclude='field_*.jls')
    if [ "$keep" = 1 ] && ! drainer_active; then
        excl=(); log "  KEEP_CUBE=1 and no drainer ⇒ cubes included in the download (bulky!)"
    fi
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS" ${excl[@]+"${excl[@]}"} root@"$IP":"EDM/runs/$camp/" "$OUT/$camp/"
    ( CAMPAIGN="$camp"; download_verify "$OUT/$camp" ) || log "[verify] issues — products still on the VM"
    if [ -n "${PUBLISH_HOOK:-}" ] && ls "$OUT/$camp"/run_*.toml >/dev/null 2>&1; then
        log "publish: $camp → PUBLISH_HOOK"
        ( CAMP="$OUT/$camp"; CAMPAIGN="$camp"; _run_publish_hook )
    fi
    ledger "$VM" campaign_done "campaign=$camp dir=$OUT/$camp"
}

monitor_and_download() {   # poll every lane for DONE (crash = 3 consecutive dead liveness checks), then download
    log "polling ${#LANE_STEM[@]} lane(s) for completion: $LANES (DONE marker; crash = 3 consecutive failed liveness checks)…"
    local -a misses fin; local i alldone anybad=0 st
    for i in "${!LANE_STEM[@]}"; do misses[$i]=0; fin[$i]=0; done
    while :; do
        alldone=1
        for i in "${!LANE_STEM[@]}"; do
            [ "${fin[$i]}" = 1 ] && continue
            if lane_done "$i"; then fin[$i]=1; log "lane ${LANE_STEM[$i]} DONE"; continue; fi
            if lane_alive "$i"; then
                misses[$i]=0
            else
                misses[$i]=$((misses[$i]+1))
                if [ "${misses[$i]}" -ge 3 ]; then
                    lane_done "$i" && { fin[$i]=1; continue; }   # finished between checks
                    st=$(vm_status)   # a spot eviction shows up here as discontinued/offline
                    notify rotating_light urgent "EDM verda CRASH" "lane ${LANE_STEM[$i]} (${LANE_CAMP[$i]}) driver died with no DONE on $VM (status $st${SPOT:+, spot=$SPOT}) — VM KEPT; '$0 attach' to retry or teardown."
                    ledger "$VM" campaign_crash "campaign=${LANE_CAMP[$i]} lane=${LANE_STEM[$i]} driver died, no DONE, vm status $st"
                    log "[ERROR] lane ${LANE_STEM[$i]}: driver gone, no DONE — VM $VM KEPT (status $st). tail:"
                    ssh_vm "cd EDM && tail -n 20 runs/${LANE_STEM[$i]}.out 2>/dev/null" | sed 's/^/[vm] /' || true
                    fin[$i]=1; anybad=1; continue
                fi
                log "  liveness check failed for ${LANE_STEM[$i]} (${misses[$i]}/3) — transient ssh blip or a real crash, retrying…"
            fi
            alldone=0
        done
        [ "$alldone" = 1 ] && break
        sleep 60
        for i in "${!LANE_STEM[@]}"; do [ "${fin[$i]}" = 1 ] || lane_tail "$i"; done
    done
    vm_hook POSTLUDE || notify warning high "EDM verda postlude FAILED" "$LANES on $VM: $VERDA_VM_POSTLUDE exited non-zero — products still downloaded; see runs/<camp>/$(basename "${VERDA_VM_POSTLUDE:-x}" .sh).log"
    local camp keep j
    for camp in $(printf '%s\n' "${LANE_CAMP[@]}" | sort -u); do
        keep=0; for j in "${!LANE_CAMP[@]}"; do [ "${LANE_CAMP[$j]}" = "$camp" ] && [ "${LANE_KEEP[$j]}" = 1 ] && keep=1; done
        download_campaign "$camp" "$keep"
    done
    if [ "$anybad" = 1 ]; then
        notify warning high "EDM verda finished WITH CRASHES" "$LANES on $VM: some lane(s) died — products downloaded; VM KEPT."
    else
        notify white_check_mark default "EDM verda done" "$LANES → $OUT/{$CAMPAIGN} ; VM $VM KEPT — run teardown."
    fi
    log "products → $OUT/{$CAMPAIGN} ; VM $VM KEPT (state $STATE). More: $0 run <campaign>... Finish: $0 teardown"
    return "$anybad"
}

# vm_hook <PRELUDE|POSTLUDE> — run the configured hook script on the VM for every campaign dir of
# this run (lanes usually share one). Output → EDM/runs/<camp>/<hook stem>.log (downloaded with the
# campaign). Returns the hook's status; the callers decide what a failure means.
vm_hook() {
    local which=$1 var="VERDA_VM_$1" script rc=0 camp stem
    script="${!var:-}"; [ -n "$script" ] || return 0
    [ -f "$ORCH/$script" ] || { log "[ERROR] $var=$script not found under orchestration/"; return 64; }
    stem=$(basename "$script" .sh)
    for camp in $(printf '%s\n' "${LANE_CAMP[@]}" | sort -u); do
        log "$which $script → runs/$camp/$stem.log"
        ssh_vm "export PATH=\"\$HOME/.juliaup/bin:\$PATH\"; cd EDM && mkdir -p runs/$camp && bash \$HOME/edm-orch/$script runs/$camp > runs/$camp/$stem.log 2>&1; rc=\$?; tail -n 5 runs/$camp/$stem.log; exit \$rc" \
            | sed "s/^/[vm $stem] /" ; rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] && log "$which $stem OK" || log "$which $stem FAILED (rc=$rc)"
    done
    return "$rc"
}

run_campaign() {   # run <campaign.sh>... — several files = concurrent lanes on one VM
    load_lanes "$@"
    if vm_reachable; then
        if ssh_vm '[ -d "$HOME/EDM/.git" ] && [ -x "$HOME/.juliaup/bin/julia" ]' 2>/dev/null; then
            log "reusing kept VM $VM ($IP, $TYPE @ $LOC) from $STATE (no provision/warm) — syncing repo to $BRANCH"
            ssh_vm "cd EDM && git fetch --quiet origin '$BRANCH' && git checkout --quiet -f '$BRANCH' && git reset --quiet --hard 'origin/$BRANCH' && echo '[sync] now at' \$(git rev-parse --short HEAD) 'on' \$(git branch --show-current)"
        else
            log "kept VM $VM ($IP) was never warmed — warming now"
            warm
        fi
    else
        if [ -f "$STATE" ]; then   # unreachable but maybe still billing — never silently orphan it
            read -r VM _ < "$STATE"
            st=$(vm_status)
            case "$st" in notfound|deleting|unknown|discontinued) log "stale state: VM $VM is $st — clearing $STATE"; rm -f "$STATE" ;;
                *) log "[ERROR] kept VM $VM ($STATE) is '$st' but unreachable — investigate or '$0 teardown' first."; exit 1 ;; esac
        fi
        provision || { log "FAILED: no VM provisioned"; notify rotating_light high "EDM verda FAILED" "$LANES: no VM provisioned (capacity / API)"; exit 5; }
        echo "$VM PENDING $BACKEND $TYPE $LOC" > "$STATE"   # bills from NOW — record before anything can fail
        trap 'rc=$?; log "FAILED (rc=$rc) before campaign launch"; notify rotating_light urgent "EDM verda FAILED" "$LANES setup errored (rc=$rc); tearing down"; teardown; exit $rc' ERR
        wait_ready
        echo "$VM $IP $BACKEND $TYPE $LOC" > "$STATE"
        warm
        trap - ERR    # VM up + warm; a campaign hiccup below must NOT auto-destroy it
    fi
    push_orchestration
    if ! vm_hook PRELUDE; then
        if [ "${VERDA_PRELUDE_FAIL:-keep}" = teardown ]; then
            notify rotating_light urgent "EDM verda PRELUDE FAILED" "$LANES on $VM: $VERDA_VM_PRELUDE failed — tearing down (VERDA_PRELUDE_FAIL=teardown), nothing launched."
            ledger "$VM" campaign_crash "campaign=$CAMPAIGN prelude $VERDA_VM_PRELUDE failed; teardown"
            teardown; exit 6
        fi
        notify warning high "EDM verda prelude FAILED" "$LANES on $VM: $VERDA_VM_PRELUDE failed — launching anyway (VERDA_PRELUDE_FAIL=keep); see runs/<camp>/$(basename "$VERDA_VM_PRELUDE" .sh).log"
    fi
    notify hourglass_flowing_sand default "EDM verda started" "$LANES on $VM ($TYPE @ $LOC, $BACKEND, spot=$SPOT)"
    # One row per campaign dir with dir=$OUT/<camp> (same as runpod.sh): dir= attribution
    # needs the campaign dir, and a crashed campaign only ever gets this row.
    local camp lanes
    for camp in $(printf '%s\n' "${LANE_CAMP[@]}" | sort -u); do
        lanes=""; for i in "${!LANE_CAMP[@]}"; do [ "${LANE_CAMP[$i]}" = "$camp" ] && lanes="$lanes,${LANE_STEM[$i]}"; done
        ledger "$VM" campaign_start "campaign=$camp lanes=${lanes#,} dir=$OUT/$camp"
    done
    local i; for i in "${!LANE_STEM[@]}"; do launch_lane "$i"; done
    start_drainer || notify warning high "EDM drainer NOT started" "$LANES on $VM: cubes stay on the VM only; teardown gate will hold them"
    monitor_and_download
}

attach_campaign() {   # resume monitoring+download after a driver-side interruption — never relaunches
    load_lanes "$@"
    vm_reachable || { log "[ERROR] no reachable kept VM ($STATE)"; exit 1; }
    log "attached to kept VM $VM ($IP) for $LANES"
    start_drainer || notify warning high "EDM drainer NOT started" "$LANES on $VM: cubes stay on the VM only; teardown gate will hold them"
    monitor_and_download
}

vm_delete() {   # PUT /instances {action:delete} — deletes the OS volume too (API default), i.e. no storage leak
    vapi -X PUT "$API/instances" -d "$(jq -n --arg id "$VM" '{action:"delete",id:$id}')" >/dev/null
}

teardown() {
    [ -f "$STATE" ] || { echo "no kept VM recorded in $STATE"; exit 0; }
    read -r VM IP BACKEND TYPE LOC < "$STATE"
    [ "$IP" = PENDING ] || cube_gate
    /usr/bin/ssh -O exit -o ControlPath="$CM" root@"$IP" 2>/dev/null || true
    if vm_delete 2>/dev/null; then
        rm -f "$STATE"; ledger "$VM" teardown "type=$TYPE loc=$LOC" "" "$(vm_balance)"
        log "deleted VM $VM (OS volume removed with it; billing stops at the current 10-min increment)"
        notify checkered_flag default "EDM verda torn down" "VM $VM ($TYPE) deleted."
    else
        notify rotating_light urgent "EDM teardown FAILED" "delete of VM $VM errored — likely STILL billing. Delete in the Verda console."
        log "[ERROR] delete of $VM FAILED — likely STILL billing. Delete in the console (or 'verda vm delete $VM'). state kept at $STATE."; exit 1
    fi
}

case "$MODE" in
    run)      shift; run_campaign "$@" ;;
    attach)   shift; attach_campaign "$@" ;;
    teardown) teardown ;;
    *) echo "usage: $0 run <campaign.sh>... | attach <campaign.sh>... | teardown" >&2; exit 64 ;;
esac
