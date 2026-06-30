#!/usr/bin/env bash
# SLURM backend (submit side) — run a campaign as a job array on the cluster.
#   bash orchestration/backends/slurm_submit.sh orchestration/campaigns/<campaign>.sh
# Sizes the array from the campaign's CELLS and pulls all SLURM directives from config.env
# (SLURM_PARTITION/GRES/CPUS/MEM/TIME/CONCURRENCY/OUTROOT). The node-side work is slurm.sbatch.
set -uo pipefail
ORCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ORCH/run_cell.sh"                       # sources config.env (SLURM_* + EDM_REPO)
CAMPAIGN_FILE="${1:?usage: slurm_submit.sh <campaign.sh>}"
. "$CAMPAIGN_FILE"                           # sets CAMPAIGN, CELLS, ...
n=${#CELLS[@]}; last=$((n - 1))
echo "[slurm] submit $CAMPAIGN: $n cells, array 0-${last}%${SLURM_CONCURRENCY:-2}, partition=${SLURM_PARTITION:-?} gres=${SLURM_GRES:-?}"
jobid=$(sbatch --parsable \
  --job-name="edm-$CAMPAIGN" \
  --partition="${SLURM_PARTITION:?set SLURM_PARTITION in config.env}" \
  --gres="${SLURM_GRES:-gpu:1}" \
  --cpus-per-task="${SLURM_CPUS:-32}" \
  --mem="${SLURM_MEM:-192G}" \
  --time="${SLURM_TIME:-12:00:00}" \
  --array="0-${last}%${SLURM_CONCURRENCY:-2}" \
  --output="%x-%A_%a.out" --error="%x-%A_%a.err" \
  --export="ALL,CAMPAIGN_FILE=$(realpath "$CAMPAIGN_FILE")" \
  "$ORCH/backends/slurm.sbatch")
echo "[slurm] submitted $CAMPAIGN as job $jobid ($n tasks)"
# Login-side ping (works only where the login node has egress; harmless no-op/warn otherwise).
# COMPLETION notifications come from the external squeue poller, not the cluster.
notify hourglass_flowing_sand default "EDM SLURM submitted" "$CAMPAIGN: $n cells, job $jobid on $(hostname). Completion via the squeue poller."
