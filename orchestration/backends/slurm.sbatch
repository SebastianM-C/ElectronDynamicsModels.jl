#!/usr/bin/env bash
#SBATCH --job-name=edm-campaign
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# Node-side: run ONE campaign cell — the array task's index into CELLS — via the shared core.
# All #SBATCH resource directives (partition/gres/cpus/mem/time/array) are supplied by
# slurm_submit.sh from config.env (CLI flags override the placeholders above). The repo (with
# orchestration/ + a machine-local config.env) must be present on the cluster.
set -uo pipefail
ORCH="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
. "$ORCH/run_cell.sh"                        # sources config.env + defines run_cell/notify
: "${CAMPAIGN_FILE:?CAMPAIGN_FILE must be exported by slurm_submit.sh}"
. "$CAMPAIGN_FILE"                            # sets CAMPAIGN, SCRIPT, BASE, CELLS

BACKEND="${SLURM_BACKEND:-cuda}"             # cluster GPUs are NVIDIA
JL=(julia +"${JULIA_CHANNEL:-release}" --startup=no -t "${SLURM_CPUS_PER_TASK:-auto}")
CAMP="${SLURM_OUTROOT:-$REPO/runs}/${CAMPAIGN}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$CAMP"
NTFY_DISABLE=1   # compute nodes have no outbound internet — completion ntfy comes from the external
                 # squeue poller (e.g. ~/.local/bin/edm-campaign-notify on a box with egress), not from here

idx="${SLURM_ARRAY_TASK_ID:?not an array job — submit via slurm_submit.sh}"
entry="${CELLS[$idx]}"; label=${entry%%|*}; overrides=${entry#*|}
[ "$overrides" = "$entry" ] && overrides=""
echo "[slurm] task $idx/$((${#CELLS[@]} - 1)): $CAMPAIGN/$label on $(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?} -> $CAMP"
# shellcheck disable=SC2086
run_cell "$label" $overrides
