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
#   POST_HOOK optional command run after a successful cell; receives the uuid as $1
# Usage: run_cell <label> [EDM_VAR=val ...]   (per-cell overrides win — env is last-wins)
: "${SCRIPT:=scripts/thomson_scattering.jl}"

# Load machine infra (gitignored). orchestration/ is this file's dir.
_ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_ORCH_DIR/config.env" ] && . "$_ORCH_DIR/config.env"
: "${REPO:=${EDM_REPO:-$HOME/.julia/dev/ElectronDynamicsModels}}"

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

run_cell() {
    local label=$1; shift
    local uuid; uuid=$(uuidgen)
    mkdir -p "$CAMP"
    [ -f "$CAMP/cells.tsv" ] || printf 'label\tuuid\tscript\tbackend\toverrides\n' > "$CAMP/cells.tsv"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$uuid" "$(basename "$SCRIPT")" "${BACKEND:-?}" "$*" >> "$CAMP/cells.tsv"
    local log="$CAMP/run_${uuid}.log"
    echo "[$(date -u +%FT%TZ)] cell $label  [$(basename "$SCRIPT") ${BACKEND:-?}]  keep=${KEEP_CUBE:-0}  $uuid :: ${*:-<baseline>}"
    ( cd "$REPO" && env ${PREENV[@]+"${PREENV[@]}"} ${BASE[@]+"${BASE[@]}"} \
          EDM_GPU_BACKEND="$BACKEND" EDM_OUTDIR="$CAMP" EDM_RUN_TAG="$uuid" "$@" \
          "${JL[@]}" --project=scripts "$SCRIPT" ) > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "  done $label ($uuid)"
        CELLS_OK=$(( ${CELLS_OK:-0} + 1 ))
        [ "${NOTIFY_EACH:-0}" = 1 ] && notify white_check_mark default "EDM cell done" "${CAMPAIGN:-?}/$label ($(hostname))"
        [ -n "${POST_HOOK:-}" ] && eval "$POST_HOOK \"$uuid\""
        [ "${KEEP_CUBE:-0}" = 1 ] || rm -f "$CAMP"/field_*_"$uuid".jls
    else
        echo "  FAILED $label rc=$rc — cube kept, see $log"
        CELLS_FAIL=$(( ${CELLS_FAIL:-0} + 1 ))
        notify rotating_light high "EDM cell FAILED" "${CAMPAIGN:-?}/$label rc=$rc on $(hostname) — see run_${uuid}.log"
    fi
    return 0   # one bad cell never aborts the sweep
}

# Iterate a CELLS array of "label|EDM_VAR=val EDM_VAR2=val2" entries (the campaign's data).
# Used by the local + hotaisle backends; SLURM indexes CELLS by $SLURM_ARRAY_TASK_ID instead.
# ntfy: a campaign-start ping, per-cell-failure pings (always), and a pass/fail summary at the end.
run_cells() {
    CELLS_OK=0; CELLS_FAIL=0
    notify hourglass_flowing_sand default "EDM campaign started" "${CAMPAIGN:-?}: ${#CELLS[@]} cells, ${BACKEND:-?}@$(hostname)"
    local entry label overrides
    for entry in "${CELLS[@]}"; do
        label=${entry%%|*}; overrides=${entry#*|}
        [ "$overrides" = "$entry" ] && overrides=""   # no '|' ⇒ baseline cell
        # shellcheck disable=SC2086
        run_cell "$label" $overrides
    done
    local tag=white_check_mark pri=default
    [ "${CELLS_FAIL:-0}" -gt 0 ] && { tag=warning; pri=high; }
    notify "$tag" "$pri" "EDM campaign done" "${CAMPAIGN:-?}: ${CELLS_OK:-0}/${#CELLS[@]} ok, ${CELLS_FAIL:-0} failed (${BACKEND:-?}@$(hostname))"
}
