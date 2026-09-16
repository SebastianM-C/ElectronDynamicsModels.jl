#!/usr/bin/env bash
# vm_prelude_nvidia_ncu.sh — VM prelude for an NVIDIA counter campaign (run as root on the VM,
# after warm, before any lane launches; verda.sh VERDA_VM_PRELUDE). Makes sure an Nsight Compute
# that KNOWS THE CHIP is on PATH and records whether counter collection is permitted, so the
# EDM_PROFILE cells (profile_cell.sh → ncu) collect instead of failing one by one.
#
#   bash orchestration/vm_prelude_nvidia_ncu.sh <campaign dir>      # writes <campaign dir>/prelude_ncu.txt
#
# Why: the B300 (CC 10.3) is newer than the ncu shipped with CUDA 12.x images — ncu 2025.1 answers
# "unsupported chip Unknown" for it (RunPod, 2026-09-16), while 2025.3.1 (the 13.0 image) and
# 2026.3.0 profile it. Support is PROBED (a tiny launch under each candidate), not read off
# --list-chips (no B300 name appears there even in builds that work). The CUDA apt repo carries
# every nsight-compute release as its own package (nsight-compute-<yyyy.m.p>); the newest one is
# installed when no present ncu passes the probe. The winner is linked to /usr/local/bin/ncu
# (profile_cell.sh finds `ncu` on PATH) and named in LOCAL_PREENV as GPUDIAGNOSTICS_COUNTER_TOOL.
# RmProfilingAdminOnly=1 only restricts NON-admin users: root on a full VM collects regardless,
# a rootless container (no CAP_SYS_ADMIN) does not — the txt records both facts.
# Exit status: 0 when a probed-good ncu is on PATH, 3 when none could be found (the driver
# decides: VERDA_PRELUDE_FAIL=keep continues without counters, =teardown stops billing).
set -uo pipefail
camp=${1:?usage: vm_prelude_nvidia_ncu.sh <campaign dir>}; mkdir -p "$camp"
out="$camp/prelude_ncu.txt"
log() { echo "[prelude] $*"; echo "$*" >> "$out"; }
: > "$out"
log "captured: $(date -u +%FT%TZ) on $(uname -r), user $(id -un) uid $(id -u)"
log "driver: $(nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
log "RmProfilingAdminOnly: $(grep -o 'RmProfilingAdminOnly: [0-9]*' /proc/driver/nvidia/params 2>/dev/null | awk '{print $2}')"
log "CAP_SYS_ADMIN: $(capsh --print 2>/dev/null | grep -q cap_sys_admin && echo yes || echo no)"

# Which ncu can profile THIS card? `--list-chips` is not the test: the B300 (CC 10.3) is profiled
# fine by ncu 2025.3.1 and 2026.3.0 although neither lists a "gb103" (2026-09-16, chip id 419) —
# so probe empirically: one tiny CUDA.jl launch (the warm's scripts env has CUDA) under the
# candidate with a DFMA count, accepted when the CSV carries a kernel row with the metric.
export PATH="$HOME/.juliaup/bin:$PATH"
[ -f "$HOME/edm-orch/config.env" ] && . "$HOME/edm-orch/config.env"
for kv in ${LOCAL_PREENV:-}; do export "$kv"; done          # JULIA_DEPOT_PATH=…
probe() {   # probe <ncu path>: does it collect a counter on a real launch here?
    local out; out=$(cd "$HOME/EDM" && timeout -k 20 300 "$1" --metrics smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \
        --target-processes all --csv --page raw --print-units base \
        julia +"${JULIA_CHANNEL:-release}" --startup=no --project=scripts \
        -e 'using CUDA; x = CUDA.rand(Float64, 1<<16); y = x .* 1.5 .+ 2.0; CUDA.synchronize(); println(sum(y))' 2>&1)
    echo "$out" | grep -q '^"[0-9]*","[0-9]*",' && ! echo "$out" | grep -q 'ERR_NVGPUCTRPERM\|==ERROR=='
}
candidates() {   # every ncu on the box, newest install dirs first
    ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -rV
    command -v ncu 2>/dev/null
    ls -d /usr/local/cuda*/bin/ncu 2>/dev/null | sort -rV
}
pick() {   # stdout = the chosen path ONLY (callers capture it); progress goes to stderr + the txt
    local c
    for c in $(candidates | awk '!seen[$0]++'); do
        log "probing $c ($("$c" --version 2>/dev/null | grep -o 'Version [0-9.]*' | head -1))" >&2
        if probe "$c"; then log "  → collects" >&2; echo "$c"; return 0; else log "  → no" >&2; fi
    done
    return 1
}

if ncu=$(pick); then
    log "ncu already usable: $ncu ($("$ncu" --version 2>/dev/null | grep -o 'Version [0-9.]*' | head -1))"
else
    log "no ncu on the box profiles this card — installing the newest nsight-compute from the CUDA apt repo"
    while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done
    if ! apt-cache policy 2>/dev/null | grep -q developer.download.nvidia.com; then
        . /etc/os-release; rel="ubuntu${VERSION_ID//./}"
        log "adding the CUDA apt repo ($rel)"
        curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$rel/x86_64/cuda-keyring_1.1-1_all.deb" \
            && dpkg -i /tmp/cuda-keyring.deb >/dev/null || log "cuda-keyring install failed"
    fi
    apt-get update -qq 2>&1 | tail -1
    pkg=$(apt-cache search --names-only '^nsight-compute-[0-9]' 2>/dev/null | awk '{print $1}' | sort -rV | head -1)
    if [ -n "$pkg" ]; then
        log "installing $pkg"
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" >/dev/null 2>&1 || log "apt install $pkg failed"
    else
        log "no nsight-compute-* package in the apt repo"
    fi
    if ncu=$(pick); then
        log "ncu now usable: $ncu ($("$ncu" --version 2>/dev/null | grep -o 'Version [0-9.]*' | head -1))"
    else
        log "FAILED: no ncu profiles this card; candidates: $(candidates | awk '!seen[$0]++' | paste -sd' ')"
        exit 3
    fi
fi
[ -x "$ncu" ] || { log "FAILED: chosen ncu is not executable: '$ncu'"; exit 3; }
ln -sfn "$ncu" /usr/local/bin/ncu
log "PATH ncu → $(readlink -f /usr/local/bin/ncu)"
# Belt and braces: profile_cell.sh runs under `env $LOCAL_PREENV`, so name the tool there too
# (the generated config.env does not carry it — that cost a counter sweep on an H200 pod).
cfg="$HOME/edm-orch/config.env"
if [ -f "$cfg" ] && ! grep -q GPUDIAGNOSTICS_COUNTER_TOOL "$cfg"; then
    sed -i "s|^LOCAL_PREENV=\(.*\)$|LOCAL_PREENV=\1 GPUDIAGNOSTICS_COUNTER_TOOL=/usr/local/bin/ncu|" "$cfg"
    log "config.env LOCAL_PREENV += GPUDIAGNOSTICS_COUNTER_TOOL=/usr/local/bin/ncu"
fi
exit 0
