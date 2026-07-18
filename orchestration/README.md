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
- **Archive side** — `cube_pull_r2.sh`: downloads each cube, verifies against the VM-computed
  sha256, and only then frees the bucket slot (`purge`). R2 egress is free, so a failed
  verify re-pulls at zero cost.

### Archive-box setup (once)

1. Credentials: write `~/.config/edm-r2.env` (same file the drain side ships to VMs) with the
   `RCLONE_CONFIG_R2_*` exports for a bucket-scoped R2 API token — see the header of
   `cube_drain_r2.sh` for the exact variables. Bucket-scoped tokens can't `ListBuckets`,
   which is why every rclone call in these scripts carries `--s3-no-check-bucket`.
2. rclone: any install works; `~/bin/rclone` is found even from non-interactive shells, or
   point `RCLONE=/path/to/rclone` at it.
3. Destination: pick `PULL_DEST` (e.g. `/storage/pool/smc/edm_cubes`).
4. Run the puller as a systemd user service:

   ```sh
   mkdir -p ~/.config/systemd/user
   cp orchestration/cube_pull_r2.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user edit cube_pull_r2      # override PULL_DEST / ExecStart repo path if needed
   systemctl --user enable --now cube_pull_r2
   loginctl enable-linger "$USER"          # keep it running with no login session
   journalctl --user -u cube_pull_r2 -f    # watch it work
   ```
