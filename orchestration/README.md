# orchestration/

Portable campaign framework: a campaign is pure data (`campaigns/*.sh`), `run_cell.sh` is the
shared core, `backends/` decides where cells run (local GPU / SLURM / Hot Aisle / RunPod).
Machine infra lives in the gitignored `config.env`; secrets stay under `~/.config`.

## R2 cube pipeline

Field cubes (`field_*.jls`, ~86 GB each) are too big for any ssh path off a cloud VM
(OpenSSH's channel window pins every stream at ~12–16 MB/s). Instead they ride a Cloudflare
R2 bucket (`simulation-storage`, multipart HTTPS at NIC speed) with end-to-end sha256
verification:

- **VM side** — `cube_drain_r2.sh`: watches `~/EDM/runs/*/field_*.jls`, uploads each cube
  (+ `.sha256` sidecar) once its `<uuid>.reduced` marker exists, drops `.drained_<basename>`
  sentinels. The hotaisle/runpod backends auto-start it on campaign launch when the campaign
  sets `KEEP_CUBE=1` (copied to the VM's `$HOME` first, so a branch sync can't yank it).
- **Teardown gate** — both cloud backends refuse `teardown` while any drain-eligible cube
  lacks its `.drained_` sentinel (the VM disk is the only copy). `FORCE_TEARDOWN=1` overrides.
- **Archive side** — `cube_pull_r2.sh` (+ its systemd unit), `cube_pull.sh` and
  `cube_inventory.sh` moved to the private results-dashboard repo 2026-07-22: they run only
  on the trusted archive box and feed the dashboard's status pipeline, so they live next to
  their consumer. Archive-box setup instructions moved with them. The credentials contract
  is unchanged: `~/.config/edm-r2.env` with the `RCLONE_CONFIG_R2_*` exports for a
  bucket-scoped R2 API token — see the header of `cube_drain_r2.sh` for the exact variables
  (bucket-scoped tokens can't `ListBuckets`, hence `--s3-no-check-bucket` everywhere).
