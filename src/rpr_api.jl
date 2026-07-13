# RadeonProRender rendering support: the backend plumbing hard-won during the
# Thomson animation work (see animation/thomson_rpr.jl), generic enough for any
# EDM visualization. Implementations live in package extensions, loaded on
# demand: ext/EDMRPRMakieExt.jl (needs RPRMakie) and ext/EDMIsoMeshExt.jl
# (needs MarchingCubes + GeometryBasics).

"""    rpr_capture(screen; hybrid, exposure = 0.2, sat = 1.0, dump_raw = "") -> Matrix{RGBf}

Capture a rendered RPRMakie `Screen` to an image. On Northstar (`hybrid = false`)
this is plain `colorbuffer`. On the Hybrid plugins (`hybrid = true`), where
`colorbuffer`'s resolve returns black, it reads the raw HDR framebuffer,
normalizes by per-pixel sample count (alpha), applies an exposure-scaled
LUMINANCE Reinhard (per-channel compression drags saturated highlights to
white) + sRGB gamma, then an optional post-tonemap saturation `sat` in gamma
space (≈1.25 recovers Northstar's per-channel photolinear vibrancy without
blowing highlights). `dump_raw` serializes the raw HDR buffer for offline
regrading."""
function rpr_capture end

"""    rpr_tune!(screen; quality = nothing, denoiser = nothing, ray_depth = nothing)

Set HybridPro context parameters on an RPRMakie `Screen` (call right after
constructing it; Hybrid plugins only — Northstar uses different knobs):

- `quality`: "low" | "medium" | "high" | "ultra" — RPR_CONTEXT_RENDER_QUALITY
  (0x1001), gates HybridPro's algorithmic shortcuts.
- `denoiser`: "none" | "svgf" | "asvgf" | "ml" — RPR_CONTEXT_PT_DENOISER
  (0x102D; ml needs RadeonImageFilters libs the jll doesn't ship).
- `ray_depth`: Int — max_recursion + refraction/glossy_refraction depths
  (glossy capped at 8). Nested translucent shells want 12-16: rays that
  exhaust the preset budget terminate BLACK. Probed: HybridPro ACCEPTS the
  recursion/diffuse/glossy/refraction depth family and REJECTS shadow depth,
  Russian roulette, sampler type, and adaptive sampling."""
function rpr_tune! end

"""    rpr_enable_multiframe!()

Work around the HybridPro `rprSceneClear` segfault when rendering many frames
in one process: RPRMakie's singleton `Context` releases the previous context on
every new `Screen`, and HybridPro crashes in the release. After this call, new
contexts are created non-singleton — they LEAK (measured ~1.7 GB VRAM/frame on
heavy scenes) but never tear down. Bound the leak by chunking long frame
ranges across processes (~16 frames/process on a 48 GB card)."""
function rpr_enable_multiframe! end

"""    iso_mesh(vol, Xs, Ys, Zs, level) -> GeometryBasics.Mesh | nothing

Marching-cubes isosurface of `vol` on the grids `(Xs, Ys, Zs)` (dim 1 ↔ Xs),
as a `GeometryBasics.Mesh` ready for `mesh!`, or `nothing` when the level cuts
no surface. This is the RPR-compatible replacement for `contour!`/`volume!`
(unsupported / GPU-segfaulting on RPR backends), and renders identically on
GLMakie."""
function iso_mesh end
