#!/usr/bin/env bash
# RUNPOD backend — run a campaign on a RunPod cloud GPU (MI300X in EU-RO-1 by default), using a
# PERSISTENT network volume that outlives the pod: it caches the Julia depot (precompile paid once)
# and stages cubes/products on S3-drainable storage. Mirrors hotaisle.sh's lifecycle; cell execution,
# tagging, logging and cube policy all come from the SAME run_cell.sh the pod clones.
#
#   bash orchestration/backends/runpod.sh run orchestration/campaigns/<campaign>.sh
#   bash orchestration/backends/runpod.sh teardown
#
# RunPod specifics (validated 2026-07-01 — see memory runpod-mi300x-backend):
#   • MI300X is Secure-Cloud/EU-RO-1/intermittent → grab_pod POLLS create-pod until it schedules
#   • raw rocm/pytorch crash-loops without a start CMD → we inject a dockerStartCmd sshd bootstrap
#   • Secure Cloud assigns the public IP only once the container is stable → wait_ready polls for it
#   • depot + outputs live on the network volume (/workspace) → precompile paid once, cubes persist
#
# config.env: RUNPOD_DC, RUNPOD_ROCM_IMAGE/RUNPOD_CUDA_IMAGE, RUNPOD_VOLUME_NAME/GB, RUNPOD_REPO_URL/BRANCH.
# Secrets external: API token ~/.config/runpod/token; ntfy via NTFY_ENV. The pod authorizes
# $RUNPOD_SSH_PUBKEY (a key you can auth as — e.g. your YubiKey pubkey; ControlMaster ⇒ one touch/run).
set -Eeuo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."; ORCH="$(cd "$ORCH" && pwd)"
. "$ORCH/run_cell.sh"        # config.env + notify() (notifications fire from THIS driving machine)

MODE="${1:?usage: runpod.sh run <campaign.sh> | teardown}"
TOK=$(cat "${RUNPOD_TOKEN_FILE:-$HOME/.config/runpod/token}")
API="https://rest.runpod.io/v1"
DC="${RUNPOD_DC:-EU-RO-1}"
ROCM_IMAGE="${RUNPOD_ROCM_IMAGE:-rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e}"
CUDA_IMAGE="${RUNPOD_CUDA_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
DISK="${RUNPOD_DISK_GB:-120}"
VOLNAME="${RUNPOD_VOLUME_NAME:-edm-vol}"; VOLGB="${RUNPOD_VOLUME_GB:-200}"
REPO_URL="${RUNPOD_REPO_URL:?set RUNPOD_REPO_URL in config.env}"; BRANCH="${RUNPOD_BRANCH:-main}"
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
read -r -d '' START_CMD <<'EOF' || true
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server
mkdir -p /root/.ssh /run/sshd
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
exec /usr/sbin/sshd -D -e
EOF

# gpuTypeId → "backend image": keeps the acquisition policy backend-agnostic. A CUDA fallback needs
# a CUDA image + cuda backend + its own depot (warm() namespaces JULIA_DEPOT_PATH by $BACKEND).
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
    echo "AMD Instinct MI300X OAM"   # primary
    echo "NVIDIA B200"               # in-region fallback
}

ensure_volume() {
    VOLID="$(rp "$API/networkvolumes" | jq -r --arg n "$VOLNAME" '.[]|select(.name==$n)|.id' | head -1)"
    if [ -n "$VOLID" ] && [ "$VOLID" != null ]; then log "reusing volume $VOLNAME ($VOLID) in $DC"; return 0; fi
    log "creating network volume $VOLNAME ${VOLGB}GB in $DC"
    VOLID="$(rp -X POST "$API/networkvolumes" \
        -d "$(jq -n --arg n "$VOLNAME" --argjson s "$VOLGB" --arg dc "$DC" '{name:$n,size:$s,dataCenterId:$dc}')" | jq -r .id)"
    [ -n "$VOLID" ] && [ "$VOLID" != null ] || { log "[ERROR] volume create failed"; return 1; }
}

grab_pod() {
    [ -n "$PUBKEY" ] || { log "[ERROR] no SSH pubkey — set RUNPOD_SSH_PUBKEY"; return 1; }
    ensure_volume
    local -a cands; mapfile -t cands < <(gpu_candidates)
    [ "${#cands[@]}" -gt 0 ] || { log "[ERROR] gpu_candidates returned nothing"; return 1; }
    log "grabbing (poll ${POLL}s ≤$MAXTRIES tries): ${cands[*]}"
    local try gpu prof resp pid
    for ((try=1; try<=MAXTRIES; try++)); do
        for gpu in "${cands[@]}"; do
            prof="$(gpu_profile "$gpu")"; [ -n "$prof" ] || { log "no profile for '$gpu'"; continue; }
            BACKEND="${prof%% *}"; IMAGE="${prof#* }"
            resp="$(curl -fsS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -X POST "$API/pods" \
                -d "$(jq -n --arg gpu "$gpu" --arg img "$IMAGE" --arg pub "$PUBKEY" --arg vol "$VOLID" \
                        --arg dc "$DC" --arg start "$START_CMD" --argjson disk "$DISK" \
                    '{name:"edm-runpod",imageName:$img,gpuTypeIds:[$gpu],cloudType:"SECURE",gpuCount:1,
                      containerDiskInGb:$disk,dataCenterIds:[$dc],networkVolumeId:$vol,volumeMountPath:"/workspace",
                      ports:["22/tcp"],supportPublicIp:true,env:{PUBLIC_KEY:$pub},
                      dockerEntrypoint:["/bin/bash","-c"],dockerStartCmd:[$start]}')" 2>/dev/null)" || resp=""
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
    log "warm: clone $BRANCH + instantiate ($BACKEND depot on the volume ⇒ precompile paid once)"
    ssh_vm "REPO_URL='$REPO_URL' BRANCH='$BRANCH' BK='$BACKEND' bash -s" <<'WARM'
set -e
export JULIA_DEPOT_PATH="/workspace/julia-depot-$BK"          # persists on the network volume
[ -x "$HOME/.juliaup/bin/julia" ] || curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="$HOME/.juliaup/bin:$PATH"   # no JULIA_PKG_SERVER: fresh pod uses Julia's public default (don't leak the driver's internal one)
rm -rf ~/EDM && git clone --quiet --branch "$BRANCH" "$REPO_URL" ~/EDM   # fresh clone = always current
mkdir -p /workspace/runs && ln -sfn /workspace/runs ~/EDM/runs           # cubes+products on the volume
printf 'LOCAL_BACKEND=%s\nLOCAL_JL_THREADS=auto\nLOCAL_PREENV=JULIA_DEPOT_PATH=/workspace/julia-depot-%s\nREDUCE_OVERLAP=1\n' \
    "$BK" "$BK" > ~/EDM/orchestration/config.env
ok=0; for i in 1 2 3; do
  if julia --startup=no --project=EDM/scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then ok=1; break; fi
  echo "instantiate retry $i"; sleep 20
done; [ "$ok" = 1 ] || { echo "instantiate failed"; exit 1; }
WARM
}

# Is a kept pod still reachable? (lets smoke→real / several campaigns share ONE paid pod.)
pod_reachable() { [ -f "$STATE" ] && read -r POD IP PORT VOLID BACKEND < "$STATE" && [ -n "${IP:-}" ] && ssh_vm true 2>/dev/null; }

push_orchestration() {   # overlay the driver's orchestration/ (keeping the pod's config.env)
    /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS -p $PORT" --exclude='config.env' "$ORCH/" root@"$IP":EDM/orchestration/
}

download_verify() {   # md5 each non-cube product pod-vs-local; alert on mismatch/missing
    local dst=$1 bad=0 vsum fn lsum
    while read -r vsum fn; do
        [ -n "$vsum" ] || continue; fn=$(basename "$fn")
        if [ ! -f "$dst/$fn" ]; then log "[verify] MISSING locally: $fn"; bad=1; continue; fi
        lsum=$(md5sum "$dst/$fn" | awk '{print $1}')
        [ "$lsum" = "$vsum" ] || { log "[verify] MD5 MISMATCH: $fn"; bad=1; }
    done < <(ssh_vm "cd EDM/runs/$CAMPAIGN && md5sum * 2>/dev/null | grep -vE 'field_.*\\.jls'")
    [ "$bad" -eq 0 ] && { log "[verify] $CAMPAIGN OK (cubes excluded)"; return 0; }
    notify rotating_light high "EDM download CHECK FAILED" "$CAMPAIGN: md5 mismatch/missing → $dst"; return 1
}

run_campaign() {
    local cf="${1:?usage: runpod.sh run <campaign.sh>}" cname; cname=$(basename "$cf")
    . "$cf"   # CAMPAIGN (product paths); re-read on the pod
    if pod_reachable; then
        log "reusing kept pod $POD ($IP:$PORT) from $STATE (no grab/warm)"
    else
        trap 'rc=$?; log "FAILED (rc=$rc) before pod handoff"; notify rotating_light urgent "EDM runpod FAILED" "$CAMPAIGN setup errored (rc=$rc); tearing down"; teardown; exit $rc' ERR
        grab_pod; wait_ready; warm
        echo "$POD $IP $PORT $VOLID $BACKEND" > "$STATE"
        trap - ERR    # pod up + warm; a campaign hiccup below must NOT auto-destroy it
    fi
    push_orchestration
    notify hourglass_flowing_sand default "EDM runpod started" "$CAMPAIGN on $POD ($BACKEND @$DC)"
    log "launching $CAMPAIGN on the pod via the local backend ($BACKEND), detached…"
    ssh_vm "export PATH=\"\$HOME/.juliaup/bin:\$PATH\"; cd EDM && mkdir -p runs && rm -f runs/${CAMPAIGN}.out runs/${CAMPAIGN}.pid && { nohup bash orchestration/backends/local.sh orchestration/campaigns/$cname > runs/${CAMPAIGN}.out 2>&1 < /dev/null & echo \$! > runs/${CAMPAIGN}.pid; }"
    log "polling for completion (DONE marker, or driver-death = crash)…"
    until ssh_vm "cd EDM && grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null"; do
        if ! ssh_vm "cd EDM && kill -0 \$(cat runs/${CAMPAIGN}.pid 2>/dev/null) 2>/dev/null"; then
            ssh_vm "cd EDM && grep -q '\\] ${CAMPAIGN} DONE' runs/${CAMPAIGN}.out 2>/dev/null" && break   # finished fast — DONE is present, not a crash
            notify rotating_light urgent "EDM runpod CRASH" "$CAMPAIGN driver died with no DONE on $POD — pod KEPT; teardown when done."
            log "[ERROR] driver gone, no DONE — pod $POD KEPT. tail:"; ssh_vm "cd EDM && tail -n 20 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[pod] /'; return 1
        fi
        sleep 60; ssh_vm "cd EDM && tail -n1 runs/${CAMPAIGN}.out 2>/dev/null" | sed 's/^/[pod] /'
    done
    if [ "${RUNPOD_RSYNC_DOWNLOAD:-1}" = 1 ]; then
        log "campaign done; downloading reduced products via rsync (cubes excluded)…"
        mkdir -p "$OUT/$CAMPAIGN"
        /usr/bin/rsync -az -e "/usr/bin/ssh $SSHOPTS -p $PORT" --exclude='field_*.jls' root@"$IP":"EDM/runs/$CAMPAIGN/" "$OUT/$CAMPAIGN/"
        download_verify "$OUT/$CAMPAIGN" || log "[verify] issues — products also on the volume at /workspace/runs/$CAMPAIGN"
    else
        log "campaign done; rsync skipped (RUNPOD_RSYNC_DOWNLOAD=0) — drain from the volume later: orchestration/drain.sh $CAMPAIGN"
    fi
    notify white_check_mark default "EDM runpod done" "$CAMPAIGN → $OUT/$CAMPAIGN ; pod $POD KEPT — run teardown."
    log "products → $OUT/$CAMPAIGN ; pod $POD KEPT (state $STATE). More: $0 run <campaign>. Finish: $0 teardown"
}

teardown() {   # delete the pod (stops GPU billing); KEEP the volume (depot+cubes persist, ~\$0.07/GB-mo)
    [ -f "$STATE" ] || { echo "no kept pod recorded in $STATE"; exit 0; }
    read -r POD IP PORT VOLID BACKEND < "$STATE"
    /usr/bin/ssh -O exit -o ControlPath="$CM" root@"$IP" 2>/dev/null || true
    if rp -X DELETE "$API/pods/$POD" >/dev/null 2>&1; then
        rm -f "$STATE"; log "deleted pod $POD; volume $VOLID KEPT (drain via S3, or delete to stop storage \$)."
        notify checkered_flag default "EDM runpod torn down" "pod $POD deleted; volume $VOLID kept."
    else
        notify rotating_light urgent "EDM teardown FAILED" "DELETE of pod $POD errored — likely STILL billing. Delete in console."
        log "[ERROR] DELETE of $POD FAILED — likely STILL billing. Delete in console. state kept at $STATE."; exit 1
    fi
}

case "$MODE" in
    run)      shift; run_campaign "${1:-}" ;;
    teardown) teardown ;;
    *) echo "usage: $0 run <campaign.sh> | teardown" >&2; exit 64 ;;
esac
