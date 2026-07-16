# RayMakie/Hikari vs RPRMakie/HybridPro — experiment notes & verdict

Evaluating `../RayDemo`'s stack (Makie `sd/lava`, Hikari `sd/vk-hw-accel`,
Lava `sd/nvidia` — Julia-native spectral path tracing over Vulkan) as a
replacement for the RPRMakie/HybridPro production pipeline
(`animation/thomson_rpr.jl` + `render_production.sh`).

## Verdict: adopt for the volume look — with a lookdev pass first

**RayMakie volume mode renders the animation at 3.5 s/frame in one process —
3–4× faster than RPR HybridPro (10–15 s/frame), with no VRAM-leak chunking,
no manual tonemap, and a genuinely volumetric look RPR cannot produce at
all.** The mesh (striped-glass) look, however, is currently blocked by
upstream performance bugs, so an exact like-for-like port of the locked RPR
style is not viable today. Recommended path: keep `rpr_frames_v3` as the
shipped mesh-style render, develop the next animation (and the planned
inverse-scattering one) on RayMakie volume mode after a proper lookdev
session, and file the upstream issues below.

## Measured numbers (W7900, RADV; 1280×960 unless noted)

| configuration | per frame | notes |
|---|---|---|
| RPR HybridPro production (700 iter, ultra) | 10–15 s | + 1.7 GB/frame leak → 16-frame chunks, 2 passes |
| RayMakie **volume style, SW BVH, 64 spp** | **3.5 s** | 61-frame clip, single process, scene reuse |
| RayMakie volume style, HW RT, 64 spp | 6.3 s | works (small BLAS), but slower — volume cost dominates |
| RayMakie mesh style, fresh scene | 209 s | SW BLAS build over ~1.6 M stripe triangles |
| RayMakie mesh style, observable mesh swap | 2282 s | pathological upstream path (see issues) |
| RayMakie mesh style, steady state (nothing changed) | 0.5 s (640×480×32) | tracing itself is fast |
| physics payloads (`precompute_frames.jl`, animation env) | 0.3 s/frame (volume) / ~6–9 s (mesh+volume stills) | CPU, runs in parallel with the GPU |

One-time costs per render process: ~100 s kernel compile (SW) / ~120 s (HW).
Clip evidence: `thomson_ray_clip.mp4` (frames 120–180, t = 0→6 T0, flash).

## What worked well

- **First-try look parity was close** for the static port: same room/grid,
  gold electrons, striped screen, camera path (`ray_t-5.00.png`,
  `ray_t+3.00.png`, `ray_t+26.00.png` vs `rpr_frames_v3/rpr_{0071,0151,0380}.png`).
  Materials are typed structs; glassglow is ONE node
  (`MediumInterface(Dielectric; emission=…)`), `Emissive(two_sided=true)`
  replaces RPR's DOUBLESIDED mode, capture is just
  `colorbuffer(...; exposure, tonemap=:aces, gamma)`.
- **Emissive volumes from raw Julia arrays** (`RGBGridMedium` with per-voxel
  σ_a/Le): the radiation cube and the analytic pulse render as true glowing
  media that light the room (floor caustic patterns in the clip) — the
  original "the flash lights the room" idea, impossible in RPR.
- **Scene reuse across frames actually works** for volumes + meshscatter:
  grid re-upload ~2 s, electron position updates cheap, camera via
  observables. 61 frames, one process, no leak.

## Bring-up recipe (headless W7900)

- GLFW.jl hardcodes X11 on Linux → run under a virtual compositor:
  `kwin_wayland --virtual --no-global-shortcuts --socket=wayland-edm &`,
  then `WAYLAND_DISPLAY=wayland-edm JULIA_GLFW_PLATFORM=wayland`.
  xkbcommon Compose errors are harmless.
- **Env split forced**: Hikari pins StructArrays 0.6; SciMLBase 3 needs
  RecursiveArrayTools 4 → StructArrays 0.7. Physics (setup.jl) runs in
  `--project=animation` via `precompute_frames.jl` → plain-array payloads
  (13–81 MB/frame) → `thomson_ray.jl` renders from `--project=animation/raymakie`.

## Upstream findings (file as issues on Hikari/RayMakie)

1. **`RGBGridMedium(σ_s_grid = nothing)` renders as uniform fog** filling the
   whole bounding box. Explicit zero σ_s grid restores emission-only
   behavior. Minimal reproducer: `mwe_medium.jl`.
2. **Observable mesh swap is ~10× slower than a full scene rebuild**
   (2282 s vs 209 s for the same ~1.6 M-triangle content, SW BVH).
3. **HW RT (`hw_accel=true`) DEVICE_LOSTs** (`rt_indirect` batch) on the
   ~1.6 M-triangle mesh scene; fine on small scenes and the volume-style
   scene (~100 k triangles).
4. (Documented, not a bug: a Dielectric volume boundary shadows the box —
   use `MediumInterface(NullMaterial(); inside=medium)`, per RayMakie
   `plots/volume.jl`.)

## Remaining work before production adoption

- [ ] Volume lookdev with a human eye: current mapping
  (`EDM_RAY_VOL_FLOOR/GAMMA/LE/SIGMA`, pulse floor 0.45 ≙ the mesh ±0.5
  isolevel) shows structure but cores still trend white; grading
  (ambient/softbox/exposure) not yet matched to the locked RPR level.
- [ ] Screen `develop` sequence + camera pull-back check over the full
  400-frame window.
- [ ] spp / denoiser trade (64 spp is grainy; `DenoiseConfig` untested).
- [ ] Optional: report + track the upstream issues; mesh style becomes
  viable if the BLAS-rebuild and mesh-update paths are fixed.

## Files

- `Project.toml` — render env (ray stack only, GeometryBasics =0.5.10 pin)
- `precompute_frames.jl` — physics side (animation env): payloads per frame
- `thomson_ray.jl` — renderer: EDM_RAY_* knobs, mesh|volume styles, reuse
- `sweep_glass.jl`, `sweep_buildtrace.jl`, `sweep_volume.jl` — measurements
- `mwe_medium.jl` — σ_s_grid=nothing reproducer
- `frame_cache/`, `ray_frames/`, `*.png`, `thomson_ray_clip.mp4` — outputs (untracked)
