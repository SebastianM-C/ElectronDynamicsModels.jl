#!/usr/bin/env bash
# RUNPOD backend — run a campaign on a RunPod cloud GPU (MI300X in EU-RO-1 by default). Mirrors
# hotaisle.sh's lifecycle; cell execution, tagging, logging and cube policy all come from the SAME
# run_cell.sh the pod clones.
#
#   bash orchestration/backends/runpod.sh run <campaign.sh>      grab pod → warm → run → download
#   bash orchestration/backends/runpod.sh attach <campaign.sh>   re-attach to a kept pod's campaign
#                                                                (monitor + download, NO relaunch)
#   bash orchestration/backends/runpod.sh teardown               delete the kept pod (stops billing)
#
# Storage layout (reworked 2026-07-02 — the FUSE network volume is pathological for depots):
#   • Julia depot → pod-LOCAL disk (/root/julia-depot-$BACKEND), restored from / pushed to a
#     zstd archive on the VPS (RUNPOD_DEPOT_CACHE, rrsync-jailed) ⇒ warm in minutes, not ~30
#   • outputs → pod-local EDM/runs by default, rsync'd down when the campaign finishes
#   • the network volume is OPT-IN (RUNPOD_VOLUME_GB>0): only for campaigns that must persist
#     cubes past the pod (S3 drain via orchestration/drain.sh); never put a depot on it
#
# RunPod specifics (validated 2026-07-01 — see memory runpod-mi300x-backend):
#   • MI300X is Secure-Cloud/EU-RO-1/intermittent → grab_pod POLLS create-pod until it schedules
#   • raw rocm/pytorch crash-loops without a start CMD → we inject a dockerStartCmd sshd bootstrap
#   • Secure Cloud assigns the public IP only once the container is stable → wait_ready polls for it
#
# config.env: RUNPOD_DC, RUNPOD_ROCM_IMAGE/RUNPOD_CUDA_IMAGE, RUNPOD_VOLUME_NAME/GB,
# RUNPOD_REPO_URL/BRANCH, RUNPOD_DEPOT_CACHE/RUNPOD_DEPOT_KEY. Secrets external: API token at
# ~/.config/runpod/token; ntfy via NTFY_ENV. The pod authorizes $RUNPOD_SSH_PUBKEY (a key you can
# auth as — e.g. your YubiKey pubkey; ControlMaster ⇒ one touch/run).
set -Eeuo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."; ORCH="$(cd "$ORCH" && pwd)"
# Caller-env overrides must survive config.env (sourced via run_cell.sh below, where a plain
# RUNPOD_BRANCH=... assignment would clobber them) — capture before, prefer after. Same for
# every per-campaign knob a caller passes inline: VOLUME_GB bit us 2026-07-19 (caller's 1400
# silently reverted to config.env's 0 ⇒ a volumeless pod for a volume-dependent campaign).
_CALLER_BRANCH="${RUNPOD_BRANCH-}"
_CALLER_VOLGB="${RUNPOD_VOLUME_GB-}"
_CALLER_DISK="${RUNPOD_DISK_GB-}"
_CALLER_DC="${RUNPOD_DC-__unset__}"
. "$ORCH/run_cell.sh"        # config.env + notify() (notifications fire from THIS driving machine)
[ -n "$_CALLER_BRANCH" ] && RUNPOD_BRANCH="$_CALLER_BRANCH"
[ -n "$_CALLER_VOLGB" ] && RUNPOD_VOLUME_GB="$_CALLER_VOLGB"
[ -n "$_CALLER_DISK" ] && RUNPOD_DISK_GB="$_CALLER_DISK"
[ "$_CALLER_DC" != "__unset__" ] && RUNPOD_DC="$_CALLER_DC"   # explicit empty = unpinned, must survive too

MODE="${1:?usage: runpod.sh run <campaign.sh> | attach <campaign.sh> | teardown}"
TOK=$(cat "${RUNPOD_TOKEN_FILE:-$HOME/.config/runpod/token}")
API="https://rest.runpod.io/v1"
DC="${RUNPOD_DC-EU-RO-1}"   # unset ⇒ EU-RO-1; EXPLICIT empty ⇒ unpinned (scheduler picks the DC)
ROCM_IMAGE="${RUNPOD_ROCM_IMAGE:-rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e}"
CUDA_IMAGE="${RUNPOD_CUDA_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
DISK="${RUNPOD_DISK_GB:-120}"
VOLNAME="${RUNPOD_VOLUME_NAME:-edm-vol}"; VOLGB="${RUNPOD_VOLUME_GB:-0}"
REPO_URL="${RUNPOD_REPO_URL:?set RUNPOD_REPO_URL in config.env}"; BRANCH="${RUNPOD_BRANCH:-main}"
DEPOT_CACHE="${DEPOT_CACHE:-}"; DEPOT_CACHE_KEY="${DEPOT_CACHE_KEY:-$HOME/.config/runpod/depot_key}"   # shared cache (see depot_cache.sh)
STATE="${RUNPOD_STATE:-$HOME/.config/runpod/campaign_pod}"; OUT="${RUNPOD_OUT:-$HOME/campaign_out}"
POLL="${RUNPOD_POLL_SEC:-120}"; MAXTRIES="${RUNPOD_MAX_TRIES:-240}"
PUBKEY="$(cat "${RUNPOD_SSH_PUBKEY:-$HOME/.config/runpod/ssh_pubkey}" 2>/dev/null || ssh-add -L 2>/dev/null | head -1)"
CM="$HOME/.ssh/cm-runpod-$(basename "$STATE").sock"   # per-STATE: concurrent drivers must not remux onto one socket
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"

rp()      { curl -fsS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" "$@"; }
ssh_vm()  { /usr/bin/ssh $SSHOPTS -p "$PORT" root@"$IP" "$@"; }
log()     { echo "[$(date -u +%FT%TZ)] $*"; }

# Persistent cost ledger (shared with hotaisle.sh; reported by the private results-dashboard
# repo's scripts/cost_report.sh — only the appends live here, 2026-07-22).
# rate_cents_h = the pod's costPerHr (converted to cents) captured at grab time — rates vary per
# GPU type and drift. balance_usd = live account balance via pod_balance() (GraphQL myself{clientBalance};
# REST v1 + MCP get-billing expose spend, NOT balance), recorded on provision + teardown so
# cost_report always has a fresh anchor (spent = Σ topups − latest balance). A ledger write must
# NEVER break a campaign — pod_balance() is fail-safe (empty on any error).
LEDGER="${EDM_CLOUD_LEDGER:-$HOME/.config/edm-cloud-ledger.tsv}"
ledger()  {   # ledger <vm> <event> <detail> [rate_cents_h] [balance_usd]
    { mkdir -p "$(dirname "$LEDGER")"
      [ -f "$LEDGER" ] || printf 'ts_utc\tprovider\tvm\tevent\tdetail\trate_cents_h\tbalance_usd\n' > "$LEDGER"
      printf '%s\trunpod\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-}" "${5:-}" >> "$LEDGER"
    } 2>/dev/null || true
}

# Live account balance in USD, for the ledger's balance_usd anchor. RunPod exposes balance ONLY via
# GraphQL myself{clientBalance} (REST v1 + MCP get-billing return spend, not balance; verified
# 2026-07-20 — matched the console to the cent). MUST be fail-safe: it feeds $(...) inside a ledger
# write that must never break a campaign (esp. at teardown), so echo the number or nothing and
# swallow every error (bad/absent token, network, GraphQL errors block, jq miss).
pod_balance() {
    rp --max-time 5 -X POST https://api.runpod.io/graphql \
       --data '{"query":"query{myself{clientBalance}}"}' 2>/dev/null \
       | jq -r '.data.myself.clientBalance // empty' 2>/dev/null || true
}

# sshd bootstrap: sshd in the foreground IS the keep-alive; also authorize the injected key.
# Needed on any image or the container crash-loops (rocm/pytorch's default CMD just exits).
# rsync + zstd serve the depot cache restore/push and the product download, on ANY base image.
read -r -d '' START_CMD <<'EOF' || true
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync zstd
mkdir -p /root/.ssh /run/sshd
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
exec /usr/sbin/sshd -D -e
EOF

# gpuTypeId → "backend image": keeps the acquisition policy backend-agnostic. A CUDA fallback needs
# a CUDA image + cuda backend + its own depot archive (warm() keys the cache by $BACKEND).
gpu_profile() {
    case "$1" in
        "AMD Instinct MI300X OAM")   echo "rocm $ROCM_IMAGE" ;;
        NVIDIA*)                     echo "cuda $CUDA_IMAGE" ;;
        *)                           echo "" ;;
    esac
}

# ── GPU acquisition policy ───────────────────────────────────────────────────
# Echo the gpuTypeIds to attempt, ONE PER LINE, in priority order. grab_pod tries each in turn every
# poll round; the first that schedules in EU-RO-1 wins (and its gpu_profile sets image+backend+depot).
# This encodes your "use whatever's available in EU-RO-1" strategy.
gpu_candidates() {
    # RUNPOD_GPU_CANDIDATES (comma-separated gpuTypeIds) overrides the ladder —
    # e.g. pin MI300X-only when the budget can't absorb a pricey fallback.
    if [ -n "${RUNPOD_GPU_CANDIDATES:-}" ]; then
        echo "$RUNPOD_GPU_CANDIDATES" | tr ',' '\n' | sed 's/^ *//; s/ *$//' | grep -v '^$'
        return
    fi
    echo "AMD Instinct MI300X OAM"   # primary (best FP64/$ for the LW kernel)
    echo "NVIDIA H200 NVL"           # CUDA fallbacks, cheapest first ($3.79 vs $4.39/hr secure)
    echo "NVIDIA H200"
    echo "NVIDIA B200"               # last resort — Blackwell FP64 is weak (measured 67 s/slice, ~2× MI300X cost-time)
}

ensure_volume() {   # opt-in: only called when RUNPOD_VOLUME_GB > 0; must match OUR datacenter
    VOLID="$(rp "$API/networkvolumes" | jq -r --arg n "$VOLNAME" --arg dc "$DC" \
        '.[]|select(.name==$n and .dataCenterId==$dc)|.id' | head -1)"
    if [ -n "$VOLID" ] && [ "$VOLID" != null ]; then log "reusing volume $VOLNAME ($VOLID) in $DC"; return 0; fi
    log "creating network volume $VOLNAME ${VOLGB}GB in $DC"
    VOLID="$(rp -X POST "$API/networkvolumes" \
        -d "$(jq -n --arg n "$VOLNAME" --argjson s "$VOLGB" --arg dc "$DC" '{name:$n,size:$s,dataCenterId:$dc}')" | jq -r .id)"
    [ -n "$VOLID" ] && [ "$VOLID" != null ] || { log "[ERROR] volume create failed"; return 1; }
    # Storage ledger row (volumes outlive pods, billed per GB-month — invisible to the
    # provision/teardown spans). rate col = the volume's whole cost in CENTS PER HOUR at its
    # creation size (gb × RUNPOD_STORAGE_CENTS_GB_MO / 730; default 7 ¢/GB-month = RunPod's
    # $0.07 list price) — the ledger's universal rate unit. Volume deletion is manual
    # (console/API), so the closing `volume_delete` row is hand-seeded on the hub ledger
    # when that happens; the reporter bills an open volume to now.
    local rate_ch
    rate_ch=$(awk -v g="$VOLGB" -v r="${RUNPOD_STORAGE_CENTS_GB_MO:-7}" 'BEGIN { printf "%.2f", g * r / 730 }')
    ledger "$VOLNAME" volume_create "id=$VOLID size_gb=$VOLGB dc=$DC" "$rate_ch"
}

grab_pod() {
    [ -n "$PUBKEY" ] || { log "[ERROR] no SSH pubkey — set RUNPOD_SSH_PUBKEY"; return 1; }
    if [ "$VOLGB" -gt 0 ] 2>/dev/null; then
        ensure_volume
    else
        VOLID=""; log "network volume: none (RUNPOD_VOLUME_GB=0) — outputs pod-local, downloaded before teardown"
    fi
    local -a cands; mapfile -t cands < <(gpu_candidates)
    [ "${#cands[@]}" -gt 0 ] || { log "[ERROR] gpu_candidates returned nothing"; return 1; }
    log "grabbing (poll ${POLL}s ≤$MAXTRIES tries): ${cands[*]}"
    local try gpu prof resp pid rate
    for ((try=1; try<=MAXTRIES; try++)); do
        for gpu in "${cands[@]}"; do
            prof="$(gpu_profile "$gpu")"; [ -n "$prof" ] || { log "no profile for '$gpu'"; continue; }
            BACKEND="${prof%% *}"; IMAGE="${prof#* }"
            resp="$(curl --fail-with-body -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -X POST "$API/pods" \
                -d "$(jq -n --arg gpu "$gpu" --arg img "$IMAGE" --arg pub "$PUBKEY" --arg vol "${VOLID:-}" \
                        --arg dc "$DC" --arg start "$START_CMD" --argjson disk "$DISK" \
                    '{name:"edm-runpod",imageName:$img,gpuTypeIds:[$gpu],cloudType:"SECURE",gpuCount:1,
                      containerDiskInGb:$disk,
                      ports:["22/tcp"],supportPublicIp:true,env:{PUBLIC_KEY:$pub},
                      dockerEntrypoint:["/bin/bash","-c"],dockerStartCmd:[$start]}
                     + (if $dc != "" then {dataCenterIds:[$dc]} else {} end)
                     + (if $vol != "" then {networkVolumeId:$vol,volumeMountPath:"/workspace"} else {} end)')" 2>&1)" || {
                # --fail-with-body keeps the API's reason; stay quiet on the two EXPECTED capacity-poll
                # messages (the throttled "no capacity yet" line below reports those), log everything else.
                case "$resp" in
                    *"no instances currently available"*|*"could not find any pods"*) : ;;
                    *) log "grab $gpu ($BACKEND) create rejected: $resp" ;;
                esac
                resp=""
            }
            pid="$(echo "$resp" | jq -r '.id // empty' 2>/dev/null)"
            # Camping can run unattended for hours and billing starts HERE, before warm —
            # ping as soon as a pod is secured, not only at campaign launch.
            [ -n "$pid" ] && { POD="$pid"
                rate="$(echo "$resp" | jq -r 'if .costPerHr then (.costPerHr*100|round) else empty end' 2>/dev/null)" || rate=""
                ledger "$POD" provision "gpu=$gpu dc=$DC" "$rate" "$(pod_balance)"
                log "grabbed $gpu → pod $POD ($BACKEND / $IMAGE, ${rate:-?}¢/h)"
                notify satellite default "EDM runpod grabbed" "$gpu → pod $POD (try $try/$MAXTRIES, billing started)"; return 0; }
        done
        [ $(( (try-1) % 10 )) -eq 0 ] && log "  no capacity yet (try $try/$MAXTRIES)…"
        sleep "$POLL"
    done
    log "[ERROR] gave up after $MAXTRIES tries"; return 1
}

resolve_endpoint() {   # set IP/PORT from the API for $POD — RunPod remaps container :22 to a new host port on restart
    local j; j="$(rp "$API/pods/$POD?includeMachine=true" 2>/dev/null || true)"
    IP="$(echo "$j" | jq -r '.publicIp // empty' 2>/dev/null)"
    PORT="$(echo "$j" | jq -r '.portMappings["22"] // empty' 2>/dev/null)"
    [ -n "$IP" ] && [ -n "$PORT" ]
}

wait_ready() {   # public IP + 22 mapping appear only once the container is stable; then sshd is up
    log "waiting for public IP + sshd…"; local i
    for i in $(seq 1 60); do
        resolve_endpoint && { log "endpoint root@$IP:$PORT"; break; }
        sleep 15
    done
    [ -n "${IP:-}" ] && [ -n "${PORT:-}" ] || { log "[ERROR] no endpoint after wait"; return 1; }
    for i in $(seq 1 40); do ssh_vm true 2>/dev/null && return 0; sleep 10; done
    log "[ERROR] sshd never came up"; return 1
}

warm() {
    log "warm: clone $BRANCH + instantiate ($BACKEND depot on pod-local disk; cache: ${DEPOT_CACHE:-none})"
    if [ -n "$DEPOT_CACHE" ]; then
        if [ -f "$DEPOT_CACHE_KEY" ]; then   # jailed key — the pod can ONLY rsync inside the archive store
            ssh_vm 'mkdir -p /root/.ssh && cat > /root/.ssh/depot_key && chmod 600 /root/.ssh/depot_key' < "$DEPOT_CACHE_KEY"
        else
            log "[warn] DEPOT_CACHE set but $DEPOT_CACHE_KEY missing — building the depot fresh"
            DEPOT_CACHE=""
        fi
    fi
    ssh_vm "REPO_URL='$REPO_URL' BRANCH='$BRANCH' BK='$BACKEND' DEPOT_CACHE='$DEPOT_CACHE' HAS_VOL='${VOLID:+1}' bash -s" <<'WARM'
set -e
export JULIA_DEPOT_PATH="/root/julia-depot-$BK"   # pod-local NVMe — NEVER the FUSE volume (100k tiny files)
[ -x "$HOME/.juliaup/bin/julia" ] || curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"   # no JULIA_PKG_SERVER: fresh pod uses Julia's public default (don't leak the driver's internal one)
rm -rf ~/EDM && git clone --quiet --branch "$BRANCH" "$REPO_URL" ~/EDM   # fresh clone = always current
if [ -n "$HAS_VOL" ]; then mkdir -p /workspace/runs && ln -sfn /workspace/runs ~/EDM/runs   # cubes persist on the volume
else mkdir -p ~/EDM/runs; fi
printf 'LOCAL_BACKEND=%s\nLOCAL_JL_THREADS=auto\nLOCAL_PREENV=JULIA_DEPOT_PATH=%s\nREDUCE_OVERLAP=1\nLOCAL_CLOUD_PROVIDER=runpod\n' \
    "$BK" "$JULIA_DEPOT_PATH" > ~/EDM/orchestration/config.env
REPO_DIR=~/EDM; . ~/EDM/orchestration/depot_cache.sh   # julia-actions/cache semantics over the rsync store
depot_cache_restore   # → DC_RESTORED = exact | prefix (instantiate tops it up) | miss (fresh build)
ok=0; for i in 1 2 3; do
  if julia --startup=no --project=EDM/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then ok=1; break; fi
  echo "instantiate retry $i"; sleep 20
done; [ "$ok" = 1 ] || { echo "instantiate failed"; exit 1; }
[ "$DC_RESTORED" = exact ] || depot_cache_push   # exact hit ⇒ the store already has this depot
WARM
}

# Is a kept pod still reachable? (lets smoke→real / several campaigns share ONE paid pod.)
pod_reachable() {
    [ -f "$STATE" ] || return 1
    read -r POD IP PORT VOLID BACKEND < "$STATE" || return 1
    [ "$VOLID" = "-" ] && VOLID=""
    [ -n "${POD:-}" ] || return 1
    resolve_endpoint || return 1   # a kept pod may have restarted onto a new host port; refresh before trusting STATE
    echo "$POD $IP $PORT ${VOLID:--} $BACKEND" > "$STATE"
    ssh_vm true 2>/dev/null
}

push_orchestration() {   # overlay the driver's orchestration/ (keeping the pod's config.env)
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS -p $PORT" --exclude='config.env' "$ORCH/" root@"$IP":EDM/orchestration/
}

download_verify() {   # md5 each non-cube product pod-vs-local (subdirs included); alert on mismatch/missing
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

# Cube drainer (R2): mirrors hotaisle.sh — copy script + creds OUT of the repo clone (a branch
# sync must not yank a running drainer), pgrep-guarded nohup start, only when the campaign sets
# KEEP_CUBE=1. rc>0 ⇒ caller alerts; a missing drainer never blocks (the teardown gate holds cubes).
start_drainer() {
    [ "${KEEP_CUBE:-0}" = 1 ] || return 0
    local envf="${CUBE_R2_ENV:-$HOME/.config/edm-r2.env}"
    [ -f "$envf" ] || { log "[drain] $envf missing — no R2 creds to ship"; return 1; }
    ssh_vm 'mkdir -p ~/.config && cat > ~/.config/edm-r2.env && chmod 600 ~/.config/edm-r2.env' < "$envf" || return 1
    ssh_vm 'cat > ~/cube_drain_r2.sh' < "$ORCH/cube_drain_r2.sh" || return 1
    # guard is its OWN ssh call: bundled with the nohup start, the remote shell's cmdline would
    # contain the plain script name and pgrep would always self-match (drainer never starts)
    if ! drainer_active; then
        ssh_vm 'nohup bash ~/cube_drain_r2.sh >> ~/drain_r2.log 2>&1 < /dev/null &' || return 1
    fi
    log "[drain] cube_drain_r2.sh running on the pod (log: ~/drain_r2.log)"
}
# check-only remote cmdline carries just the [c]-bracketed pattern — no self-match
drainer_active() { ssh_vm "pgrep -f '[c]ube_drain_r2.sh' >/dev/null" 2>/dev/null; }

# Cube-safety gate: pod deletion destroys pod-local disk — the ONLY copy of an undrained cube
# (volume-backed runs are exempt: teardown keeps the volume). Same eligibility contract as the
# drainer (<uuid>.reduced present, smoke excluded). FORCE_TEARDOWN=1 overrides.
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
    log "[gate] REFUSING teardown — undrained cubes on $POD (the pod holds the only copy):"
    echo "$pending" | sed 's/^/  /'
    log "[gate] wait for the drainer (tail ~/drain_r2.log on the pod), or override: FORCE_TEARDOWN=1 $0 teardown"
    notify rotating_light urgent "EDM teardown BLOCKED" "pod $POD: undrained cubes — $(echo "$pending" | tr '\n' ' '); drainer still working? FORCE_TEARDOWN=1 to override."
    exit 1
}

monitor_and_download() {   # poll for DONE (crash = 3 consecutive dead liveness checks), then download+verify
    log "polling for completion (DONE marker; crash = 3 consecutive failed liveness checks)…"
    local misses=0
    until ssh_vm "cd EDM && grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null"; do
        if ssh_vm "cd EDM && kill -0 \$(cat runs/${CAMPAIGN}.pid 2>/dev/null) 2>/dev/null"; then
            misses=0
        else
            misses=$((misses+1))
            if [ "$misses" -ge 3 ]; then
                ssh_vm "cd EDM && grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null" && break   # finished between checks
                notify rotating_light urgent "EDM runpod CRASH" "$CAMPAIGN driver died with no DONE on $POD — pod KEPT; '$0 attach' to retry or teardown."
                ledger "$POD" campaign_crash "campaign=$CAMPAIGN driver died, no DONE"
                log "[ERROR] driver gone, no DONE — pod $POD KEPT. tail:"; ssh_vm "cd EDM && tail -n 20 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[pod] /'; return 1
            fi
            log "  liveness check failed ($misses/3) — transient ssh blip or a real crash, retrying…"
        fi
        sleep 60; ssh_vm "cd EDM && tail -n1 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[pod] /' || true
    done
    local dl="${RUNPOD_RSYNC_DOWNLOAD:-1}"
    if [ "$dl" != 1 ] && [ -z "${VOLID:-}" ]; then
        log "[warn] RUNPOD_RSYNC_DOWNLOAD=0 but no volume attached — downloading anyway (products would die with the pod)"; dl=1
    fi
    if [ "$dl" = 1 ]; then
        log "campaign done; downloading products via rsync…"
        mkdir -p "$OUT/$CAMPAIGN"
        local -a excl=(--exclude='field_*.jls')
        if [ "${KEEP_CUBE:-0}" = 1 ] && [ -z "${VOLID:-}" ] && ! drainer_active; then
            excl=(); log "  KEEP_CUBE=1, no volume, no drainer ⇒ cubes included in the download (bulky!)"
        fi
        /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS -p $PORT" ${excl[@]+"${excl[@]}"} root@"$IP":"EDM/runs/$CAMPAIGN/" "$OUT/$CAMPAIGN/"
        download_verify "$OUT/$CAMPAIGN" || log "[verify] issues — products still on the pod${VOLID:+ and volume /workspace/runs/$CAMPAIGN}"
    else
        log "campaign done; rsync skipped (RUNPOD_RSYNC_DOWNLOAD=0) — drain from the volume later: orchestration/drain.sh $CAMPAIGN"
    fi
    notify white_check_mark default "EDM runpod done" "$CAMPAIGN → $OUT/$CAMPAIGN ; pod $POD KEPT — run teardown."
    ledger "$POD" campaign_done "campaign=$CAMPAIGN dir=$OUT/$CAMPAIGN"
    log "products → $OUT/$CAMPAIGN ; pod $POD KEPT (state $STATE). More: $0 run <campaign>. Finish: $0 teardown"
}

run_campaign() {
    local cf="${1:?usage: runpod.sh run <campaign.sh>}" cname; cname=$(basename "$cf")
    . "$cf"   # CAMPAIGN + KEEP_CUBE (names product paths + sets cube download policy); re-read on the pod
    if pod_reachable; then
        log "reusing kept pod $POD ($IP:$PORT) from $STATE (no grab/warm) — syncing repo to $BRANCH"
        # A kept pod carries whatever branch it was warmed with; sync so a re-run
        # with a different RUNPOD_BRANCH doesn't silently execute stale code.
        ssh_vm "cd EDM && git fetch --quiet origin '$BRANCH' && git checkout --quiet -f '$BRANCH' && git reset --quiet --hard 'origin/$BRANCH' && echo '[sync] now at' \$(git rev-parse --short HEAD) 'on' \$(git branch --show-current)"
    else
        if [ -f "$STATE" ]; then   # unreachable but maybe still billing — never silently orphan it
            read -r POD _ < "$STATE"
            if rp "$API/pods/$POD" >/dev/null 2>&1; then
                log "[ERROR] kept pod $POD ($STATE) still EXISTS but is unreachable — investigate or '$0 teardown' first."; exit 1
            fi
            log "stale state: pod $POD is gone — clearing $STATE"; rm -f "$STATE"
        fi
        trap 'rc=$?; log "FAILED (rc=$rc) before campaign launch"; notify rotating_light urgent "EDM runpod FAILED" "$CAMPAIGN setup errored (rc=$rc); tearing down"; teardown; exit $rc' ERR
        grab_pod
        echo "$POD PENDING 0 ${VOLID:--} $BACKEND" > "$STATE"   # pod bills from NOW — record it before anything can fail
        wait_ready
        echo "$POD $IP $PORT ${VOLID:--} $BACKEND" > "$STATE"
        warm
        trap - ERR    # pod up + warm; a campaign hiccup below must NOT auto-destroy it
    fi
    push_orchestration
    if ssh_vm "cd EDM && grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null"; then
        log "$CAMPAIGN already DONE on the pod — skipping launch (rm runs/${CAMPAIGN}.out on the pod to force a rerun)"
    elif ssh_vm "cd EDM && kill -0 \$(cat runs/${CAMPAIGN}.pid 2>/dev/null) 2>/dev/null"; then
        log "$CAMPAIGN already RUNNING on the pod — monitoring it (no relaunch)"
    else
        notify hourglass_flowing_sand default "EDM runpod started" "$CAMPAIGN on $POD ($BACKEND @$DC)"
        ledger "$POD" campaign_start "campaign=$CAMPAIGN dir=$OUT/$CAMPAIGN"
        log "launching $CAMPAIGN on the pod via the local backend ($BACKEND), detached…"
        ssh_vm "export PATH=\"\$HOME/.juliaup/bin:\$PATH\"; cd EDM && mkdir -p runs && rm -f runs/${CAMPAIGN}.out runs/${CAMPAIGN}.pid && { nohup bash orchestration/backends/local.sh orchestration/campaigns/$cname > runs/${CAMPAIGN}.out 2>&1 < /dev/null & echo \$! > runs/${CAMPAIGN}.pid; }"
    fi
    start_drainer || notify warning high "EDM drainer NOT started" "$CAMPAIGN on $POD: cubes stay on the pod only; teardown gate will hold them"
    monitor_and_download
}

attach_campaign() {   # resume monitoring+download after a driver-side interruption — never relaunches
    local cf="${1:?usage: runpod.sh attach <campaign.sh>}"; . "$cf"
    pod_reachable || { log "[ERROR] no reachable kept pod ($STATE)"; exit 1; }
    log "attached to kept pod $POD ($IP:$PORT) for $CAMPAIGN"
    start_drainer || notify warning high "EDM drainer NOT started" "$CAMPAIGN on $POD: cubes stay on the pod only; teardown gate will hold them"
    monitor_and_download
}

teardown() {   # delete the pod (stops GPU billing); a volume, if attached, is KEPT (bills ~\$0.07/GB-mo)
    [ -f "$STATE" ] || { echo "no kept pod recorded in $STATE"; exit 0; }
    read -r POD IP PORT VOLID BACKEND < "$STATE"; [ "$VOLID" = "-" ] && VOLID=""
    resolve_endpoint 2>/dev/null || true   # refresh the port in case the pod restarted, so cube_gate/ssh hit the right endpoint
    if [ -n "${VOLID:-}" ]; then
        log "[gate] volume $VOLID attached — cubes persist past the pod, gate skipped"
    else
        cube_gate
    fi
    /usr/bin/ssh -O exit -o ControlPath="$CM" root@"$IP" 2>/dev/null || true
    if rp -X DELETE "$API/pods/$POD" >/dev/null 2>&1; then
        rm -f "$STATE"; ledger "$POD" teardown "${VOLID:+volume=$VOLID kept}" "" "$(pod_balance)"
        log "deleted pod $POD${VOLID:+; volume $VOLID KEPT (drain via S3, then delete to stop storage \$)}"
        notify checkered_flag default "EDM runpod torn down" "pod $POD deleted${VOLID:+; volume $VOLID kept}."
    else
        notify rotating_light urgent "EDM teardown FAILED" "DELETE of pod $POD errored — likely STILL billing. Delete in console."
        log "[ERROR] DELETE of $POD FAILED — likely STILL billing. Delete in console. state kept at $STATE."; exit 1
    fi
}

case "$MODE" in
    run)      shift; run_campaign "${1:-}" ;;
    attach)   shift; attach_campaign "${1:-}" ;;
    teardown) teardown ;;
    *) echo "usage: $0 run <campaign.sh> | attach <campaign.sh> | teardown" >&2; exit 64 ;;
esac
