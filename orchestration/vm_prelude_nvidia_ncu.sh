#!/usr/bin/env bash
# vm_prelude_nvidia_ncu.sh — VM prelude for an NVIDIA counter campaign (run as root on the VM,
# after warm, before any lane launches; verda.sh VERDA_VM_PRELUDE). Makes sure an Nsight Compute
# that KNOWS THE CHIP is on PATH and records whether counter collection is permitted, so the
# EDM_PROFILE cells (profile_cell.sh → ncu) collect instead of failing one by one.
#
#   bash orchestration/vm_prelude_nvidia_ncu.sh <campaign dir>      # writes <campaign dir>/prelude_ncu.txt
#
# Why: the B300 (GB103, CC 10.3) is newer than the ncu shipped with CUDA 12.x/13.0 images —
# ncu 2025.1 answers "unsupported chip Unknown" for it (RunPod, 2026-09-16). The CUDA apt repo
# carries every nsight-compute release as its own package (nsight-compute-<yyyy.m.p>); the
# newest one is installed when the present ncu does not list the chip, and linked to
# /usr/local/bin/ncu (profile_cell.sh finds `ncu` on PATH; GPUDIAGNOSTICS_COUNTER_TOOL also works).
# RmProfilingAdminOnly=1 only restricts NON-admin users: root on a full VM collects regardless,
# a rootless container (no CAP_SYS_ADMIN) does not — the txt records both facts.
# Exit status: 0 when an ncu listing the chip is on PATH, 3 when none could be found (the driver
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

# The chip code ncu uses (gb103 for the B300, gb100/gb200/gb202 …): derived from the CC when
# nvidia-smi does not spell it; the listing is matched case-insensitively.
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
case "$cc" in
    10.3) chip=gb103 ;; 10.0) chip=gb100 ;; 12.0) chip=gb202 ;; 12.1) chip=gb10 ;; 9.0) chip=gh100 ;; *) chip="" ;;
esac
log "compute capability $cc → chip ${chip:-unknown}"

knows_chip() {   # knows_chip <ncu path>: does this ncu list the chip?
    [ -n "$chip" ] || return 0   # unknown CC: any ncu will do (the cell tells)
    "$1" --list-chips 2>/dev/null | tr ',' '\n' | tr -d ' ' | grep -qix "$chip"
}
candidates() {   # every ncu on the box, newest install dirs first
    command -v ncu 2>/dev/null
    ls -d /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/bin/ncu /usr/local/cuda*/nsight-compute*/ncu 2>/dev/null | sort -rV
}
pick() { local c; for c in $(candidates | awk '!seen[$0]++'); do knows_chip "$c" && { echo "$c"; return 0; }; done; return 1; }

if ncu=$(pick); then
    log "ncu already usable: $ncu ($("$ncu" --version 2>/dev/null | grep -o 'Version [0-9.]*' | head -1))"
else
    log "no ncu on the box lists chip ${chip:-?} — installing the newest nsight-compute from the CUDA apt repo"
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
        log "FAILED: no ncu lists chip ${chip:-?}; candidates: $(candidates | awk '!seen[$0]++' | paste -sd' ')"
        log "list-chips of the newest: $(candidates | head -1 | xargs -r -I{} {} --list-chips 2>&1 | head -c 400)"
        exit 3
    fi
fi
ln -sfn "$ncu" /usr/local/bin/ncu
log "PATH ncu → $(readlink -f /usr/local/bin/ncu)"
exit 0
