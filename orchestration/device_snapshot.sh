#!/usr/bin/env bash
# device_snapshot.sh — capture what the run manifests cannot see from inside the process: the
# vendor driver's sysfs/proc state of the GPU the cell ran on, as files registered in the cell's
# [outputs] so they ride to the dashboard with the run. Meant as a POST_HOOK (receives the uuid):
#     POST_HOOK="bash orchestration/device_snapshot.sh \"\$CAMP\""
#
# AMD:    gpu_metrics_<uuid>_<pci>.bin  the amdgpu `gpu_metrics` blob of every card (idle, after the cell) — the
#                                  MI300 throttle residencies / per-XCD clocks GPUDiagnostics decodes;
#                                  the first bytes say format/content/size, and whether an SR-IOV
#                                  virtual function exposes it at all
#         device_<uuid>.txt        performance level (and whether it is writable), power cap, the
#                                  `amd-smi metric --violation` output where the tool exists
# NVIDIA: device_<uuid>.txt        RmProfilingAdminOnly (counter collection permitted?), the
#                                  clocks / power / throttle sections of `nvidia-smi -q`, `ncu` presence
# Registers `outputs.gpu_metrics_idle` / `outputs.device_snapshot` in run_<uuid>.toml. Never fails a
# cell: every probe is best-effort and a missing tool just leaves its section short. Contains no
# hostnames: what it records is device state, and the manifest already carries provenance.
set -uo pipefail
camp=${1:?usage: device_snapshot.sh <campaign dir> <uuid>}; uuid=${2:?uuid}
manifest="$camp/run_$uuid.toml"; [ -f "$manifest" ] || { echo "[snapshot] no manifest for $uuid" >&2; exit 0; }
backend=$(grep -m1 '^gpu_backend = ' "$manifest" | sed 's/.*= "\(.*\)"/\1/')
txt="$camp/device_$uuid.txt"
{
    echo "backend: ${backend:-?}"
    echo "captured: $(date -u +%FT%TZ) (after the cell, device idle)"
    case "$backend" in
        rocm)
            # Every amdgpu card the host exposes (a workstation card beside an APU is common;
            # a cloud VF exposes one): one section and one blob per card, named by PCI slot,
            # so the reader picks the run's device by name/slot rather than by card index.
            for d in /sys/class/drm/card*/device; do
                [ -f "$d/gpu_metrics" ] || continue
                grep -qi "DRIVER=amdgpu" "$d/uevent" 2>/dev/null || continue
                slot=$(grep PCI_SLOT_NAME "$d/uevent" 2>/dev/null | cut -d= -f2)
                echo "--- card: $d (PCI $slot)"
                echo "vram total (bytes): $(cat "$d/mem_info_vram_total" 2>/dev/null)"
                echo "gpu_metrics header (size lo, size hi, format, content): $(head -c 4 "$d/gpu_metrics" | od -An -tu1)"
                echo "gpu_metrics bytes: $(wc -c < "$d/gpu_metrics")"
                lvl="$d/power_dpm_force_performance_level"
                echo "performance level: $(cat "$lvl" 2>/dev/null || echo unavailable) (writable: $([ -w "$lvl" ] && echo yes || echo no))"
                echo "power cap (µW): $(cat "$d"/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)"
                echo "sclk: $(grep '\*' "$d/pp_dpm_sclk" 2>/dev/null | head -1)"
                cp "$d/gpu_metrics" "$camp/gpu_metrics_${uuid}_${slot}.bin" 2>/dev/null
            done
            if command -v amd-smi >/dev/null; then
                echo "--- amd-smi metric --violation (bare-metal API; unsupported on a VF is a finding too)"
                timeout 20 amd-smi metric --violation 2>&1 | head -40
                echo "--- amd-smi static --vbios --board"
                timeout 20 amd-smi static --vbios --board 2>&1 | head -20
            fi ;;
        cuda)
            echo "RmProfilingAdminOnly: $(grep -o 'RmProfilingAdminOnly: [0-9]' /proc/driver/nvidia/params 2>/dev/null || echo unknown)"
            echo "ncu: $(command -v ncu 2>/dev/null || ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1 || echo none)"
            if command -v nvidia-smi >/dev/null; then
                echo "--- nvidia-smi -q (clocks, power, performance state)"
                timeout 20 nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | head -120
            fi ;;
        *) echo "no vendor snapshot for backend ${backend:-?}" ;;
    esac
} > "$txt" 2>&1
# register the files in [outputs] (TOML stdlib only — no project needed); gpu_metrics_idle is
# the list of per-card blobs (one on a cloud VF, several on a multi-GPU host)
blobs=$(cd "$camp" && ls gpu_metrics_"$uuid"_*.bin 2>/dev/null | tr '\n' ' ')
julia --startup=no -e '
    using TOML
    m = TOML.parsefile(ARGS[1]); outs = get!(m, "outputs", Dict{String, Any}())
    outs["device_snapshot"] = ARGS[2]
    blobs = String.(split(ARGS[3]))
    isempty(blobs) ? delete!(outs, "gpu_metrics_idle") : (outs["gpu_metrics_idle"] = blobs)
    open(io -> TOML.print(io, m; sorted = true), ARGS[1], "w")' "$manifest" "$(basename "$txt")" "$blobs" \
    || echo "[snapshot] could not register the snapshot in $manifest" >&2
echo "[snapshot] $uuid → $(basename "$txt")${blobs:+, $blobs}"
