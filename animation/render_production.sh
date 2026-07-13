#!/usr/bin/env bash
# Production render of the RPR Thomson animation (final look, 2026-07-13).
# HybridPro leaks ~1.7 GB VRAM per frame until process exit, so the 400 play
# frames run in chunks of CHUNK per julia process; frames are resumable
# (existing files skip), and the loop runs twice so a crashed chunk's missing
# frames self-heal on the second pass. Encodes the mp4 at the end and drops a
# DONE marker. Run detached:
#   setsid nohup bash animation/render_production.sh > /tmp/render_prod.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
CHUNK=${CHUNK:-16}
TOTAL=${TOTAL:-400}
OUTDIR=${OUTDIR:-animation/rpr_frames_v3}
MP4=${MP4:-animation/thomson_rpr_v3.mp4}
mkdir -p "$OUTDIR"

# final look (memory: rpr-look-locked)
export EDM_RPR_PLUGIN=hybridpro EDM_RPR_QUALITY=ultra EDM_RPR_BG=room
export EDM_RPR_ITER=${EDM_RPR_ITER:-700}
export EDM_RPR_RIBBONS=glassglow EDM_RPR_EMIS=0.7 EDM_RPR_EMIS_COOL=0.25
export EDM_RPR_ELECTRONS=gold
export EDM_RPR_RAD_STYLE=striped EDM_RPR_RAD_MAT=glass EDM_RPR_RAD_TINT=0.5
export EDM_RPR_RAD_COLORS="1.0,0.75,0.2;0.55,0.3,0.9"
export EDM_RPR_AMBIENT=7.0 EDM_RPR_EXPOSURE=0.3 EDM_RPR_SAT=1.45
export EDM_RPR_SOFTBOX=6 EDM_RPR_RAY_DEPTH=16
export EDM_RPR_SCREEN_STYLE=striped EDM_RPR_SCREEN_FLOOR=0.12
export EDM_RPR_SCREEN_REFL=0.25 EDM_RPR_SCREEN_DEVELOP=live
# fixed seed: without it the sampler state advances across a chunk's frames and
# the per-frame grain boil reads as ±0.7 px camera judder (measured)
export EDM_RPR_SEED=42
# dissolve the laser as it reaches the detector (t = 14→18 T0, frames ~260-320)
export EDM_RPR_PULSE_FADE=${EDM_RPR_PULSE_FADE:-14,4}
export EDM_RPR_ELECTRON_ROUGH=0.45
export EDM_RPR_OUTDIR="$OUTDIR"

chunk_missing() {  # any frame of chunk $1..$2 absent?
    local i
    for i in $(seq "$1" "$2"); do
        [ -f "$OUTDIR/$(printf 'rpr_%04d.png' "$i")" ] || return 0
    done
    return 1
}

for pass in 1 2; do
    for a in $(seq 1 "$CHUNK" "$TOTAL"); do
        b=$((a + CHUNK - 1)); [ "$b" -gt "$TOTAL" ] && b=$TOTAL
        chunk_missing "$a" "$b" || continue
        echo "[$(date -u +%FT%TZ)] pass $pass chunk $a:$b"
        EDM_RPR_FRAMES=$a:$b julia -t 8 --startup=no --project=animation \
            animation/thomson_rpr.jl ||
            echo "[$(date -u +%FT%TZ)] chunk $a:$b rc=$? (resumes next pass)"
    done
done

n=$(ls "$OUTDIR"/rpr_*.png 2>/dev/null | wc -l)
echo "[$(date -u +%FT%TZ)] frames complete: $n/$TOTAL"
if [ "$n" -eq "$TOTAL" ]; then
    ffmpeg -y -framerate 30 -i "$OUTDIR/rpr_%04d.png" -c:v libx264 \
        -pix_fmt yuv420p -crf 18 "$MP4" &&
        echo "[$(date -u +%FT%TZ)] encoded $MP4"
else
    echo "[$(date -u +%FT%TZ)] INCOMPLETE — rerun this script to fill the gaps"
fi
touch "$OUTDIR/RENDER_DONE"
