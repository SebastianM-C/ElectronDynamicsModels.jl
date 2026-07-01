#!/usr/bin/env bash
# Shared run core for EDM campaigns — used by EVERY backend (local / SLURM / Hot Aisle).
# COMMITTED + generic: no infra, no secrets. Machine specifics come from orchestration/config.env
# (gitignored); secrets stay in ~/.config (referenced by path). Because this file ships in the repo,
# it's present on any machine the repo is cloned to (your box, a cluster node, a cloud VM).
#
# THE CONTRACT every backend honours, identically:
#   • mint a UUID run-tag (never readable tags — they collide in the 8-char derived-sidecar key)
#   • set EDM_RUN_TAG + EDM_OUTDIR, run the solver, tee/redirect to run_<uuid>.log beside the manifest
#   • record a canonical cells.tsv trail (label ⇆ uuid ⇆ script ⇆ backend ⇆ overrides)
#   • apply the cube-retention policy (KEEP_CUBE)
#   • optionally run a post-process hook (a reducer), given the uuid
#
# A CAMPAIGN is pure data — it sets BASE (baseline EDM_* env) + SCRIPT and lists CELLS; a BACKEND
# sets JL/PREENV/CAMP/BACKEND and decides where/how cells run. The split keeps campaigns and the
# core infra-free and portable.
#
# Caller sets before using run_cell:
#   REPO      EDM checkout (cd target; scripts env at $REPO/scripts)
#   CAMP      output dir for this campaign's runs (holds run_<uuid>.{toml,log,jls}, cells.tsv)
#   BACKEND   cuda | rocm                          (→ EDM_GPU_BACKEND)
#   JL        julia launcher+flags array           e.g. JL=(julia +release --startup=no -t auto)
#   SCRIPT    solver, default scripts/thomson_scattering.jl (scripts/lpwa.jl for the analytic path)
#   BASE      array of baseline EDM_* assignments  e.g. BASE=(EDM_NX=200 EDM_NSAMPLES=6000 ...)
#   PREENV    optional leading env array           e.g. PREENV=(HIP_VISIBLE_DEVICES=0)
#   KEEP_CUBE 1 keeps the field_*.jls cube (default 0 = delete after a successful post-process)
#   POST_HOOK optional command run synchronously after a successful cell; receives the uuid as $1
#   REDUCE_OVERLAP 1 runs the solver field-only (EDM_SKIP_POSTPROCESS=1) and reduces it in the
#             BACKGROUND, so this cell's reduction overlaps the NEXT cell's GPU compute (the paid
#             multi-cell win). Reaped (+ cube policy) by reap_reduces ⇒ requires the run_cells path.
#   REDUCE_HOOK optional command (receives the uuid as $1) replacing the default cube→products
#             reducer used in overlap mode (default: harmonic_products.jl + plot_screen_observables.jl)
# Usage: run_cell <label> [EDM_VAR=val ...]   (per-cell overrides win — env is last-wins)
: "${SCRIPT:=scripts/thomson_scattering.jl}"

# Load machine infra (gitignored). orchestration/ is this file's dir.
_ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_ORCH_DIR/config.env" ] && . "$_ORCH_DIR/config.env"
# REPO defaults to the repo that CONTAINS this orchestration/ dir (so it's right on any clone —
# your dev checkout, a cluster node, a cloud VM at $HOME/EDM). config.env's EDM_REPO overrides it.
: "${REPO:=${EDM_REPO:-$(cd "$_ORCH_DIR/.." && pwd)}}"

# ── notifications (optional, ALL backends) ───────────────────────────────────
# Creds come from $NTFY_ENV (set in config.env → e.g. ~/.config/ntfy/edm-campaigns.env); secrets stay
# external. No-op if unconfigured, so campaigns run fine without ntfy. mTLS cert/key are optional
# (the topic is mTLS-gated, so the URL alone is inert — safe even if it ever leaked).
[ -n "${NTFY_ENV:-}" ] && [ -f "${NTFY_ENV:-/nonexistent}" ] && . "$NTFY_ENV"
notify() {   # notify <tags> <priority> <title> <message>  — no-op unless NTFY_URL is set
    # NTFY_DISABLE=1 hard-disables (e.g. SLURM compute nodes have no outbound internet — pushing from
    # there would just hang the curl; cluster completion is handled by an external squeue poller instead).
    [ "${NTFY_DISABLE:-0}" = 1 ] && return 0
    [ -n "${NTFY_URL:-}" ] || return 0
    curl -sS --max-time 10 ${NTFY_CERT:+--cert "$NTFY_CERT"} ${NTFY_KEY:+--key "$NTFY_KEY"} \
        ${NTFY_TOKEN:+-H "Authorization: Bearer $NTFY_TOKEN"} \
        -H "Tags: $1" -H "Priority: $2" -H "Title: $3" -d "$4" "$NTFY_URL" >/dev/null 2>&1 \
        || echo "[warn] ntfy post failed: $3" >&2
}

# Default reducer for overlap mode: turn a finished field cube into its products (the step the
# solver does inline when EDM_SKIP_POSTPROCESS is unset). Runs BACKGROUNDED; appends to the cell's
# run_<uuid>.log (so the reduce output travels with the run) and drops a <uuid>.reduced /
# <uuid>.reduce_failed marker that reap_reduces waits on. Override with REDUCE_HOOK for a custom reducer.
_reduce_cell() {
    local uuid=$1 manifest="$CAMP/run_${uuid}.toml" cube rc=0
    ( cd "$REPO" && env ${PREENV[@]+"${PREENV[@]}"} "${JL[@]}" --project=scripts scripts/harmonic_products.jl "$manifest" ) \
        >> "$CAMP/run_${uuid}.log" 2>&1 || rc=1
    if [ "$rc" -eq 0 ]; then
        for cube in "$CAMP"/field_*_"$uuid".jls; do
            [ -e "$cube" ] || continue
            ( cd "$REPO" && env ${PREENV[@]+"${PREENV[@]}"} "${JL[@]}" --project=scripts scripts/plot_screen_observables.jl "$cube" ) \
                >> "$CAMP/run_${uuid}.log" 2>&1 || rc=1
        done
    fi
    [ "$rc" -eq 0 ] && touch "$CAMP/${uuid}.reduced" || touch "$CAMP/${uuid}.reduce_failed"
}

run_cell() {
    local label=$1; shift
    local uuid; uuid=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)
    mkdir -p "$CAMP"
    [ -f "$CAMP/cells.tsv" ] || printf 'label\tuuid\tscript\tbackend\toverrides\n' > "$CAMP/cells.tsv"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$uuid" "$(basename "$SCRIPT")" "${BACKEND:-?}" "$*" >> "$CAMP/cells.tsv"
    local log="$CAMP/run_${uuid}.log"
    # In overlap mode the field runs WITHOUT inline postprocess; _reduce_cell does it backgrounded.
    local skip=""; [ "${REDUCE_OVERLAP:-0}" = 1 ] && skip="EDM_SKIP_POSTPROCESS=1"
    echo "[$(date -u +%FT%TZ)] cell $label  [$(basename "$SCRIPT") ${BACKEND:-?}]  keep=${KEEP_CUBE:-0} overlap=${REDUCE_OVERLAP:-0}  $uuid :: ${*:-<baseline>}"
    # shellcheck disable=SC2086
    ( cd "$REPO" && env ${PREENV[@]+"${PREENV[@]}"} ${BASE[@]+"${BASE[@]}"} $skip \
          EDM_GPU_BACKEND="$BACKEND" EDM_OUTDIR="$CAMP" EDM_RUN_TAG="$uuid" "$@" \
          "${JL[@]}" --project=scripts "$SCRIPT" ) > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        CELLS_OK=$(( ${CELLS_OK:-0} + 1 ))
        [ "${NOTIFY_EACH:-0}" = 1 ] && notify white_check_mark default "EDM cell done" "${CAMPAIGN:-?}/$label ($(hostname))"
        if [ "${REDUCE_OVERLAP:-0}" = 1 ]; then
            echo "  field done $label ($uuid) — reduce backgrounded (overlaps next cell)"
            rm -f "$CAMP/${uuid}.reduced" "$CAMP/${uuid}.reduce_failed"
            ( eval "${REDUCE_HOOK:-_reduce_cell \"$uuid\"}" ) &
            REDUCE_PIDS+=("$!"); PENDING_REDUCE+=("$uuid|$label")
        else
            echo "  done $label ($uuid)"
            [ -n "${POST_HOOK:-}" ] && eval "$POST_HOOK \"$uuid\""
            [ "${KEEP_CUBE:-0}" = 1 ] || rm -f "$CAMP"/field_*_"$uuid".jls
        fi
    else
        echo "  FAILED $label rc=$rc — cube kept, see $log"
        CELLS_FAIL=$(( ${CELLS_FAIL:-0} + 1 ))
        notify rotating_light high "EDM cell FAILED" "${CAMPAIGN:-?}/$label rc=$rc on $(hostname) — see run_${uuid}.log"
    fi
    return 0   # one bad cell never aborts the sweep
}

# In overlap mode, join the backgrounded reductions launched by run_cell: wait them out, then for
# each apply the cube-retention policy. A field that ran but whose reduction failed is the awkward
# case — the GPU work (the expensive part) is done, and the cube is the only way to recover the
# products without re-running it, so the policy trades disk for compute.
reap_reduces() {
    [ "${REDUCE_OVERLAP:-0}" = 1 ] || return 0
    [ "${#PENDING_REDUCE[@]}" -gt 0 ] || return 0
    echo "[reduce] waiting on ${#PENDING_REDUCE[@]} backgrounded reduction(s)…"
    wait "${REDUCE_PIDS[@]}" 2>/dev/null
    local entry uuid label
    for entry in "${PENDING_REDUCE[@]}"; do
        uuid=${entry%%|*}; label=${entry#*|}
        if [ -f "$CAMP/${uuid}.reduced" ]; then
            echo "  reduce ok: $label ($uuid)"
            [ "${KEEP_CUBE:-0}" = 1 ] || rm -f "$CAMP"/field_*_"$uuid".jls
        else
            # Field ok but reduction failed: keep the cube (recoverable), alert, count the cell failed.
            CELLS_OK=$(( ${CELLS_OK:-0} - 1 )); CELLS_FAIL=$(( ${CELLS_FAIL:-0} + 1 ))
            echo "  REDUCE FAILED: $label ($uuid) — cube KEPT for recovery, see run_${uuid}.log"
            notify rotating_light high "EDM reduce FAILED" "${CAMPAIGN:-?}/$label ($uuid): field ok, reduce failed on $(hostname) — cube kept; re-reduce from run_${uuid}.toml"
        fi
    done
}

# Iterate a CELLS array of "label|EDM_VAR=val EDM_VAR2=val2" entries (the campaign's data).
# Used by the local + hotaisle backends; SLURM indexes CELLS by $SLURM_ARRAY_TASK_ID instead.
# ntfy: a campaign-start ping, per-cell-failure pings (always), and a pass/fail summary at the end.
run_cells() {
    CELLS_OK=0; CELLS_FAIL=0; PENDING_REDUCE=(); REDUCE_PIDS=()
    notify hourglass_flowing_sand default "EDM campaign started" "${CAMPAIGN:-?}: ${#CELLS[@]} cells, ${BACKEND:-?}@$(hostname)"
    local entry label overrides
    for entry in "${CELLS[@]}"; do
        label=${entry%%|*}; overrides=${entry#*|}
        [ "$overrides" = "$entry" ] && overrides=""   # no '|' ⇒ baseline cell
        # shellcheck disable=SC2086
        run_cell "$label" $overrides
    done
    reap_reduces
    local tag=white_check_mark pri=default
    [ "${CELLS_FAIL:-0}" -gt 0 ] && { tag=warning; pri=high; }
    notify "$tag" "$pri" "EDM campaign done" "${CAMPAIGN:-?}: ${CELLS_OK:-0}/${#CELLS[@]} ok, ${CELLS_FAIL:-0} failed (${BACKEND:-?}@$(hostname))"
}
