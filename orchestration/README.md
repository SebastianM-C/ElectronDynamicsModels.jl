# orchestration/

Portable campaign framework: a campaign is pure data (`campaigns/*.sh`), `run_cell.sh` is the
shared core, `backends/` decides where cells run (local GPU / SLURM / Hot Aisle / RunPod).
Machine infra lives in the gitignored `config.env`; secrets stay under `~/.config`.

## Multi-GPU lanes (RunPod)

`accumulate_field_sharded` splits the electrons across every device the solver can see and
streams the per-device partials into one host cube (exact by linearity; host peak ≈ 1.1× cube
whatever the device count). Two knobs turn that into a benchmark or a big-N campaign:

- `RUNPOD_GPU_COUNT=<1-8>` (config.env or inline) — the pod's GPU count.
- **Lanes**: `runpod.sh run a.sh b.sh …` launches several campaign files CONCURRENTLY on one
  pod (one sequential `local.sh` lane each; each owns `runs/<stem>.out/.pid`) and monitors /
  downloads them together. Lanes may share a `CAMPAIGN` dir; cells pin themselves to device
  subsets with a per-cell `CUDA_VISIBLE_DEVICES=…` override (the CUDA extension maps ordinals
  to NVML uuids, so telemetry stays correct) and name their logical sweep with `EDM_SWEEP=…`.

Reference recipe: `campaigns/mgpu_bench_*.sh` (weak + strong scaling on 8×H200, an H100
point, a capacity cell — every cell doubles as a physics point; the header of
`mgpu_bench_common.sh` says which). Launch sequence, from the driver box:

```bash
export RUNPOD_BRANCH=<the branch carrying this recipe> RUNPOD_GPU_COUNT=8 RUNPOD_DC="" \
    RUNPOD_GPU_CANDIDATES="NVIDIA H200,NVIDIA H200 NVL" RUNPOD_DISK_GB=400 RUNPOD_DRAIN_DELETE_LOCAL=1
#   the pod resolves scripts/ FRESH at warm (no tracked manifest) — the branch must carry the current
#   [compat] caps; if the grab log shows the create rejected for the disk size, lower RUNPOD_DISK_GB
#   (DRAIN_DELETE_LOCAL=1 keeps the footprint at ~2 cubes). One pod = one GPU type for both phases.
B=orchestration/backends/runpod.sh; C=orchestration/campaigns
# size first: estimate_run.sh --nx 601 --ns 1660 --n 16000 --devices 8 --mode total --direct 1 --vram-gb 141 --thr 5.5e8
#   (phase 1 ≈ 3 × 82 GiB strong lanes + 36 GiB weak lane + overlapped reduces ≈ 300 GiB host;
#    phase 2 peaks at the 1101² cell, ≈ 235 GiB; cubes on disk ≈ 245 GB if never freed)
bash $B run $C/mgpu_bench_smoke.sh                      # ~3 min; also the pod's first fresh resolve+precompile
julia --project=scripts scripts/compare_hmaps.jl ~/campaign_out/smoke/run_<D1>.toml ~/campaign_out/smoke/run_<D2>.toml
#   → D=1 vs D=2 must agree to roundoff; the D=1 manifest's [timing].field gives the per-device
#     throughput (N·N_samples·Nx²/t): re-run estimate_run.sh --thr <it> for the real lane B length
#     (the plan assumes 5.5e8; below ~4e8 drop strong D=1 or lower EDM_N) before phase 1.
#   → check the "[pod] pod shape" line against the budget above; below ~350 GB RAM run
#     l0 + l12 first and l3456 + l7 after.
bash $B run $C/mgpu_bench_l0.sh $C/mgpu_bench_l12.sh $C/mgpu_bench_l3456.sh $C/mgpu_bench_l7.sh
bash $B run $C/mgpu_bench_p2.sh                         # all 8 GPUs, after phase 1 is DONE
bash $B teardown                                        # gate waits for the R2 drainer
# H100 point — its own pod/STATE:
RUNPOD_STATE=~/.config/runpod/pod_h100 RUNPOD_GPU_COUNT=1 RUNPOD_GPU_CANDIDATES="NVIDIA H100 80GB HBM3" \
    bash $B run $C/mgpu_bench_h100.sh && RUNPOD_STATE=~/.config/runpod/pod_h100 bash $B teardown
julia --project=scripts scripts/scaling_report.jl ~/campaign_out/mgpu_bench ~/campaign_out/mgpu_bench_h100 \
    ~/campaign_out/rest_departure_bridge_refix          # tables + scaling_*.png; extra dirs = reference throughput rows
bash orchestration/run_diagnostics.sh ~/campaign_out/mgpu_bench ~/campaign_out/mgpu_bench_h100
#   validity chips per run (CPU): pixel traces (field sampling), worldlines + final-position
#   histograms (kick-out, mass-shell, Δγ), lattice-alias check (r_alias vs measured halo onset)
julia --project=scripts scripts/compare_hmaps.jl --sweep mgpu_strong ~/campaign_out/mgpu_bench   # D=1/2/4/8 ≡ to roundoff
julia --project=scripts scripts/compare_hmaps.jl ~/campaign_out/mgpu_bench/run_<weak_N2000_D1>.toml \
    ~/campaign_out/mgpu_bench_h100/run_<h100>.toml ~/campaign_out/rest_departure_bridge_refix/run_486a637e-*.toml
#   the same cell on H200 / H100 / MI300X: one answer, three vendors; then re-publish the campaigns
```

Report field-phase times (`[timing].field`, GPU-bound, barely affected by neighbouring
lanes) separately from end-to-end cell times (which include the ~4 min of un-sharded
Julia load / serialize / reduce, and DO see lane contention on the host).

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
