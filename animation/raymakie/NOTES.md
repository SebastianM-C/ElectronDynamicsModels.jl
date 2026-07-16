# RayMakie/Hikari vs RPRMakie/HybridPro — experiment notes

Evaluating `../RayDemo`'s stack (Makie `sd/lava`, Hikari `sd/vk-hw-accel`,
Lava `sd/nvidia` — Julia-native spectral path tracing over Vulkan) as a
replacement for the RPRMakie/HybridPro production pipeline
(`animation/thomson_rpr.jl` + `render_production.sh`).

## Bring-up (W7900, headless)

- GLFW.jl **hardcodes X11 on Linux**; with no X session everything fails at
  precompile. Recipe: `kwin_wayland --virtual --no-global-shortcuts
  --socket=wayland-edm &` then `WAYLAND_DISPLAY=wayland-edm
  JULIA_GLFW_PLATFORM=wayland` (no sudo needed). xkbcommon Compose errors in
  logs are harmless.
- Lava sees the W7900 via RADV, `rt_pipeline_properties` present (HW RT
  available). RayDemo Materials demo: **1.45 s warm** at 600×450×8 spp
  (~220 s one-time kernel compile per process).
- **Env split forced**: Hikari pins StructArrays 0.6; SciMLBase 3 needs
  RecursiveArrayTools 4 → StructArrays 0.7. Physics (setup.jl, MTK + Vern9)
  cannot share an env with the ray stack. Solution:
  `precompute_frames.jl` (animation env) → plain-array payloads →
  `thomson_ray.jl` (this env). Payloads: 13–81 MB/frame, 0.5–9 s/frame to
  produce.

## First light (3 stills, 1280×960, 64 spp, SW BVH)

Look is close to the RPR reference out of the box — same composition,
tile room + grid, gold electrons, gold/violet stripes, striped screen.
Grading differences (walls darker at ambient 1.0/aces; laser ribbons more
prominent) are lookdev-level. Images: `ray_t-5.00.png`, `ray_t+3.00.png`,
`ray_t+26.00.png` vs `animation/rpr_frames_v3/rpr_{0071,0151,0380}.png`.

## Performance findings (640×480, 32 spp, flash frame t=+3)

| measurement | mesh style | volume style |
|---|---|---|
| `build_scene` | 7.6 s | 0.8 s |
| first `colorbuffer` (upload + BLAS + trace) | 209 s | 34 s |
| steady-state `colorbuffer` (nothing changed) | **0.5 s** | **0.6 s** |
| `colorbuffer` after observable update | **2282 s** (!) | **2.5 s** |

- Tracing itself is *fast* — the scene steady-states at half a second.
- Mesh style is dominated by SW-BVH build over the ~1.6 M-triangle stripe
  stacks (209 s), and RayMakie's **observable mesh-swap path is
  pathological** (2282 s — worse than a full rebuild; this is what the
  "39-minute frame" actually was). Interface model and path depth are
  secondary (glass sweep: dielectric d16 175 s → thin/interface d8 ~103 s,
  all dominated by the same floor).
- **HW RT `DEVICE_LOST`s** (`rt_indirect`) on the mesh-style scene; fine on
  the small Materials demo. Untested against the volume-style scene (few
  triangles) — TODO.
- Volume style (radiation + pulse as emissive `RGBGridMedium`, the thing RPR
  cannot do at all): per-frame grid re-upload costs ~2.5 s at test res.
  First mapping attempt (σ_a = |s| everywhere) rendered as an optically
  thick fog block — needs a floor cut so only wavefront crests participate
  (fixed; sweep over floor/sigma/Le in `sweep_volume.jl`).

## Cost model for a 400-frame production render (preliminary)

- RPR HybridPro reference: ~10–15 s/frame + 1.7 GB/frame leak → chunked
  16-frame processes, two passes (`render_production.sh`).
- RayMakie mesh style: not viable (≥209 s/frame via rebuild; observable path
  worse).
- RayMakie volume style: build once + ~2.5 s/frame updates at test res
  (full-res × spp scaling TBD) in ONE process — no leak, no chunking.
  Electron meshscatter updates are cheap (included in the 2.5 s).

## Upstream findings (candidates for issues on Hikari/RayMakie)

1. **`RGBGridMedium(σ_s_grid = nothing)` renders as uniform fog** filling the
   whole bounding box, drowning the actual grid contents. An explicit
   zero-valued σ_s grid restores correct emission-only behavior.
   Minimal reproducer: `mwe_medium.jl` (emitting ball in a box — fog block
   without the zero grid, clean glowing ball with it).
2. **A Dielectric volume boundary shadows the box** — shadow rays treat the
   cube as opaque (documented in RayMakie `plots/volume.jl`); use
   `MediumInterface(NullMaterial(); inside=medium)`.
3. **Observable mesh swap is ~10× slower than a full scene rebuild**
   (2282 s vs 209 s for the same 1.6 M-triangle content, SW BVH) — looks like
   each updated plot triggers its own full accel rebuild.
4. **HW RT (`hw_accel=true`) DEVICE_LOSTs** (`rt_indirect`) on the
   ~1.6 M-triangle mesh scene; fine on the small Materials demo.

## Open items

- [ ] Volume-mode look sweep (floor/sigma/le) → pick production mapping
- [ ] Full-res volume-mode still + timing
- [ ] HW RT retry on volume-style scene
- [ ] Short animation clip (scene reuse) + VRAM stability
- [ ] Grading pass (ambient/softbox/exposure) to match the locked RPR look
