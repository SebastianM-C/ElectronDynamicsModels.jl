#!/usr/bin/env bash
# vm_postlude_gpudiag_suite.sh — VM postlude (verda.sh VERDA_VM_POSTLUDE): run the GPUDiagnostics.jl
# real-hardware test suite (test/gpu/runtests.jl — conformance + the vendor paths CPU CI cannot
# reach: resource report, native/IR instruction mix, FP64 issue floor, GPM sampler, the ncu
# counter probe) on the campaign's card, after every lane is DONE and before the download, so the
# log rides home with the campaign products. The suite has never run on CC 10.3 (B300).
#
#   bash orchestration/vm_postlude_gpudiag_suite.sh <campaign dir>   # → <campaign dir>/gpudiag_suite.log
#
# Uses the warm's depot (JULIA_DEPOT_PATH from ~/edm-orch/config.env's LOCAL_PREENV) and Julia
# channel, so CUDA.jl and its artifacts are already there; the suite env resolves on top of it.
# GPUDIAG_REPO/GPUDIAG_BRANCH pick the checkout (default: the package's GitHub main).
# Exit status = the suite's (a failing test is a finding, not a campaign failure — the driver
# reports it and still downloads).
set -uo pipefail
camp=${1:?usage: vm_postlude_gpudiag_suite.sh <campaign dir>}; mkdir -p "$camp"
log="$camp/gpudiag_suite.log"
REPO="${GPUDIAG_REPO:-https://github.com/SebastianM-C/GPUDiagnostics.jl}"; BR="${GPUDIAG_BRANCH:-main}"
[ -f "$HOME/edm-orch/config.env" ] && . "$HOME/edm-orch/config.env"
for kv in ${LOCAL_PREENV:-}; do export "$kv"; done          # JULIA_DEPOT_PATH=…
export PATH="$HOME/.juliaup/bin:$PATH"
case "${LOCAL_BACKEND:-cuda}" in rocm) vendor=rocm ;; *) vendor=cuda ;; esac
{
    echo "[suite] $(date -u +%FT%TZ) GPUDiagnostics $BR on $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    rm -rf ~/GPUDiagnostics && git clone --quiet --branch "$BR" "$REPO" ~/GPUDiagnostics || { echo "[suite] clone failed"; exit 1; }
    cd ~/GPUDiagnostics && echo "[suite] at $(git rev-parse --short HEAD); ncu: $(command -v ncu || echo none)"
    GPUDIAGNOSTICS_GPU=$vendor timeout -k 60 "${SUITE_TIMEOUT:-3600}" \
        julia +"${JULIA_CHANNEL:-release}" --startup=no --project=test/gpu \
        -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); include("test/gpu/runtests.jl")'
    rc=$?
    echo "[suite] $(date -u +%FT%TZ) exit $rc"
    exit $rc
} 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
