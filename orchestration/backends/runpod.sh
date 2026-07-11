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
. "$ORCH/run_cell.sh"        # config.env + notify() (notifications fire from THIS driving machine)

MODE="${1:?usage: runpod.sh run <campaign.sh> | attach <campaign.sh> | teardown}"
TOK=$(cat "${RUNPOD_TOKEN_FILE:-$HOME/.config/runpod/token}")
API="https://rest.runpod.io/v1"
DC="${RUNPOD_DC:-EU-RO-1}"
ROCM_IMAGE="${RUNPOD_ROCM_IMAGE:-rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e}"
CUDA_IMAGE="${RUNPOD_CUDA_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
DISK="${RUNPOD_DISK_GB:-120}"
VOLNAME="${RUNPOD_VOLUME_NAME:-edm-vol}"; VOLGB="${RUNPOD_VOLUME_GB:-0}"
REPO_URL="${RUNPOD_REPO_URL:?set RUNPOD_REPO_URL in config.env}"; BRANCH="${RUNPOD_BRANCH:-main}"
DEPOT_CACHE="${DEPOT_CACHE:-}"; DEPOT_CACHE_KEY="${DEPOT_CACHE_KEY:-$HOME/.config/runpod/depot_key}"   # shared cache (see depot_cache.sh)
STATE="${RUNPOD_STATE:-$HOME/.config/runpod/campaign_pod}"; OUT="${RUNPOD_OUT:-$HOME/campaign_out}"
POLL="${RUNPOD_POLL_SEC:-120}"; MAXTRIES="${RUNPOD_MAX_TRIES:-240}"
PUBKEY="$(cat "${RUNPOD_SSH_PUBKEY:-$HOME/.config/runpod/ssh_pubkey}" 2>/dev/null || ssh-add -L 2>/dev/null | head -1)"
CM="$HOME/.ssh/cm-runpod.sock"
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"

rp()      { curl -fsS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" "$@"; }
ssh_vm()  { /usr/bin/ssh $SSHOPTS -p "$PORT" root@"$IP" "$@"; }
log()     { echo "[$(date -u +%FT%TZ)] $*"; }

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
    echo "AMD Instinct MI300X OAM"   # primary (best FP64/$ for the LW kernel)
    echo "NVIDIA H200 NVL"           # CUDA fallbacks, cheapest first ($3.79 vs $4.39/hr secure)
    echo "NVIDIA H200"
    echo "NVIDIA B200"               # last resort
}

ensure_volume() {   # opt-in: only called when RUNPOD_VOLUME_GB > 0; must match OUR datacenter
    VOLID="$(rp "$API/networkvolumes" | jq -r --arg n "$VOLNAME" --arg dc "$DC" \
        '.[]|select(.name==$n and .dataCenterId==$dc)|.id' | head -1)"
    if [ -n "$VOLID" ] && [ "$VOLID" != null ]; then log "reusing volume $VOLNAME ($VOLID) in $DC"; return 0; fi
    log "creating network volume $VOLNAME ${VOLGB}GB in $DC"
    VOLID="$(rp -X POST "$API/networkvolumes" \
        -d "$(jq -n --arg n "$VOLNAME" --argjson s "$VOLGB" --arg dc "$DC" '{name:$n,size:$s,dataCenterId:$dc}')" | jq -r .id)"
    [ -n "$VOLID" ] && [ "$VOLID" != null ] || { log "[ERROR] volume create failed"; return 1; }
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
    local try gpu prof resp pid
    for ((try=1; try<=MAXTRIES; try++)); do
        for gpu in "${cands[@]}"; do
            prof="$(gpu_profile "$gpu")"; [ -n "$prof" ] || { log "no profile for '$gpu'"; continue; }
            BACKEND="${prof%% *}"; IMAGE="${prof#* }"
            resp="$(curl -fsS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -X POST "$API/pods" \
                -d "$(jq -n --arg gpu "$gpu" --arg img "$IMAGE" --arg pub "$PUBKEY" --arg vol "${VOLID:-}" \
                        --arg dc "$DC" --arg start "$START_CMD" --argjson disk "$DISK" \
                    '{name:"edm-runpod",imageName:$img,gpuTypeIds:[$gpu],cloudType:"SECURE",gpuCount:1,
                      containerDiskInGb:$disk,dataCenterIds:[$dc],
                      ports:["22/tcp"],supportPublicIp:true,env:{PUBLIC_KEY:$pub},
                      dockerEntrypoint:["/bin/bash","-c"],dockerStartCmd:[$start]}
                     + (if $vol != "" then {networkVolumeId:$vol,volumeMountPath:"/workspace"} else {} end)')" 2>/dev/null)" || resp=""
            pid="$(echo "$resp" | jq -r '.id // empty' 2>/dev/null)"
            [ -n "$pid" ] && { POD="$pid"; log "grabbed $gpu → pod $POD ($BACKEND / $IMAGE)"; return 0; }
        done
        [ $(( (try-1) % 10 )) -eq 0 ] && log "  no capacity yet (try $try/$MAXTRIES)…"
        sleep "$POLL"
    done
    log "[ERROR] gave up after $MAXTRIES tries"; return 1
}

wait_ready() {   # public IP + 22 mapping appear only once the container is stable; then sshd is up
    log "waiting for public IP + sshd…"; local i j
    for i in $(seq 1 60); do
        j="$(rp "$API/pods/$POD?includeMachine=true" 2>/dev/null || true)"
        IP="$(echo "$j" | jq -r '.publicIp // empty' 2>/dev/null)"
        PORT="$(echo "$j" | jq -r '.portMappings["22"] // empty' 2>/dev/null)"
        [ -n "$IP" ] && [ -n "$PORT" ] && { log "endpoint root@$IP:$PORT"; break; }
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
printf 'LOCAL_BACKEND=%s\nLOCAL_JL_THREADS=auto\nLOCAL_PREENV=JULIA_DEPOT_PATH=%s\nREDUCE_OVERLAP=1\n' \
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
    [ -n "${IP:-}" ] && [ "$IP" != PENDING ] && ssh_vm true 2>/dev/null
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
        if [ "${KEEP_CUBE:-0}" = 1 ] && [ -z "${VOLID:-}" ]; then
            excl=(); log "  KEEP_CUBE=1 with no volume ⇒ cubes included in the download (bulky!)"
        fi
        /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS -p $PORT" ${excl[@]+"${excl[@]}"} root@"$IP":"EDM/runs/$CAMPAIGN/" "$OUT/$CAMPAIGN/"
        download_verify "$OUT/$CAMPAIGN" || log "[verify] issues — products still on the pod${VOLID:+ and volume /workspace/runs/$CAMPAIGN}"
    else
        log "campaign done; rsync skipped (RUNPOD_RSYNC_DOWNLOAD=0) — drain from the volume later: orchestration/drain.sh $CAMPAIGN"
    fi
    notify white_check_mark default "EDM runpod done" "$CAMPAIGN → $OUT/$CAMPAIGN ; pod $POD KEPT — run teardown."
    log "products → $OUT/$CAMPAIGN ; pod $POD KEPT (state $STATE). More: $0 run <campaign>. Finish: $0 teardown"
}

run_campaign() {
    local cf="${1:?usage: runpod.sh run <campaign.sh>}" cname; cname=$(basename "$cf")
    . "$cf"   # CAMPAIGN + KEEP_CUBE (names product paths + sets cube download policy); re-read on the pod
    if pod_reachable; then
        log "reusing kept pod $POD ($IP:$PORT) from $STATE (no grab/warm)"
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
        log "launching $CAMPAIGN on the pod via the local backend ($BACKEND), detached…"
        ssh_vm "export PATH=\"\$HOME/.juliaup/bin:\$PATH\"; cd EDM && mkdir -p runs && rm -f runs/${CAMPAIGN}.out runs/${CAMPAIGN}.pid && { nohup bash orchestration/backends/local.sh orchestration/campaigns/$cname > runs/${CAMPAIGN}.out 2>&1 < /dev/null & echo \$! > runs/${CAMPAIGN}.pid; }"
    fi
    monitor_and_download
}

attach_campaign() {   # resume monitoring+download after a driver-side interruption — never relaunches
    local cf="${1:?usage: runpod.sh attach <campaign.sh>}"; . "$cf"
    pod_reachable || { log "[ERROR] no reachable kept pod ($STATE)"; exit 1; }
    log "attached to kept pod $POD ($IP:$PORT) for $CAMPAIGN"
    monitor_and_download
}

teardown() {   # delete the pod (stops GPU billing); a volume, if attached, is KEPT (bills ~\$0.07/GB-mo)
    [ -f "$STATE" ] || { echo "no kept pod recorded in $STATE"; exit 0; }
    read -r POD IP PORT VOLID BACKEND < "$STATE"; [ "$VOLID" = "-" ] && VOLID=""
    /usr/bin/ssh -O exit -o ControlPath="$CM" root@"$IP" 2>/dev/null || true
    if rp -X DELETE "$API/pods/$POD" >/dev/null 2>&1; then
        rm -f "$STATE"; log "deleted pod $POD${VOLID:+; volume $VOLID KEPT (drain via S3, then delete to stop storage \$)}"
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
