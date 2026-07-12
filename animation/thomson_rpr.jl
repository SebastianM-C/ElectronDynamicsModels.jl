# Ray-traced (RadeonProRender via RPRMakie) rendering of the Thomson-scattering
# scene from thomson_animation.jl: same physics (setup.jl), different renderer.
# RPRMakie has no Contour conversion and Northstar's OpenCL backend segfaults on
# volume grids, so the signed pulse field becomes explicit isosurface meshes
# (marching cubes) — which also makes the scene mesh-only and therefore
# GPU-renderable. Materials are RPR-native: color-alpha transparency is ignored
# by the ray tracer, so the wavefront ribbons carry an emissive/glass material
# and the electrons a metallic gold Uber material.
#
# GPU note (W7900 / Linux): Northstar ships no HIP loader on Linux — a bare
# ENABLE_GPU0 context renders silently black. ENABLE_OPENCL must be OR'ed in.
#
# Run (single still):    julia -t auto --startup=no --project=animation animation/thomson_rpr.jl
# Run (animation frames): EDM_RPR_FRAMES=all julia -t auto ... animation/thomson_rpr.jl
#   then: ffmpeg -framerate 30 -i animation/rpr_frames/rpr_%04d.png -c:v libx264 \
#           -pix_fmt yuv420p -crf 18 animation/thomson_rpr.mp4
# Knobs:
#   EDM_RPR_RESOURCE=gpu|cpu  render device            (default gpu; CPU is ~7× slower, same image)
#   EDM_RPR_ITER=200          ray-tracing iterations   (more = less noise)
#   EDM_RPR_T=0.0             frame time in units of T0 (single-still mode)
#   EDM_RPR_FRAMES=a:b|all    render play_times frames a..b (1-based) into
#                             EDM_RPR_OUTDIR (default animation/rpr_frames/);
#                             existing files are skipped, so runs are resumable
#   EDM_RPR_RADIUS=0.14       electron sphere radius in units of λ (sunflower
#                             spacing is ~0.30λ — radii ≳0.15λ merge into a plate)
#   EDM_RPR_LIGHTS=laser|studio|dim  lighting rig (default laser: the pulse
#                             itself is the key light — the electrons are lit by
#                             the laser). dim (room mode): dimmed-lab chamber lit
#                             mostly by the pulse/radiation emission — lantern
#                             projections on the tiles, the flash lights the room.
#   EDM_RPR_BG=dark|room      dark: blue-gray backdrop (compositing; fast).
#                             room: gray 3-plane corner with a 2w₀ square grid;
#                             the seams are the coordinate axes. Slower (~2-4×,
#                             light bounce). Note: a Makie EnvironmentLight
#                             suppresses the composited backdrop (missed rays
#                             turn opaque black), so a light background needs
#                             real geometry.
#   EDM_RPR_RIBBONS=emissive|plastic|glass|glassglow  wavefront material.
#                             emissive glows (and in room mode lantern-projects
#                             its pattern on the walls); plastic = matte model;
#                             glass = tinted thin-sheet glass; glassglow = glass
#                             blended with 25% emission — translucent shells
#                             that read as made of light (recommended for room
#                             mode). plastic/glass/glassglow are room-mode
#                             materials — near-invisible unlit in dark mode.
#   EDM_RPR_OVERSAMPLE=2      isosurface grid densification factor. setup.jl's
#                             shared grid is only ~8 pts/λ transverse — visible
#                             marching-cubes stair-stepping on the ribbons at 1.
#   EDM_RPR_RADIATION=1       Stage C: radiated far-field shells from the
#                             precomputed cube (same file/transfer as
#                             thomson_animation.jl; EDM_RADIATION_CUBE overrides
#                             the path). RPR volumes segfault the GPU backend, so
#                             the log-compressed |E_far|² is rendered as nested
#                             emissive isosurfaces instead (inner hot/solid,
#                             outer faint/translucent). t ≈ 2–6 T0 is the flash.
#   EDM_RPR_RAD_LEVELS=0.85,0.65  shell isolevels in the log transfer
#                             (0.85 ≈ 35%, 0.65 ≈ 9% of peak intensity)
#   EDM_RPR_SCREEN=1          detector plate at z_screen textured with the LW
#                             far-field time series (screen_timeseries.jls, or
#                             the cube's +X edge as fallback). The camera path
#                             holds the hero framing through the flash, then
#                             pulls back to hold box + screen (t = 5→14 T0).
#   EDM_RPR_OUT=path.png      output file, single-still mode (default animation/rpr_frame.png)

include(joinpath(@__DIR__, "setup.jl"))

using RPRMakie
using RPRMakie: RPR
using MarchingCubes
using MarchingCubes: MC, march
import GeometryBasics
using FileIO
using Serialization
using Printf

# matte single-color material via Uber — RPR_MATERIAL_NODE_DIFFUSE is
# UNSUPPORTED on the Hybrid plugins, and Uber-diffuse looks identical
flat_material(matsys, c) = RPR.UberMaterial(matsys;
    color = to_color(c), diffuse_weight = Vec4f(1), reflection_weight = Vec4f(0))

# ── Emission-aware Volume conversion (method overwrite of RPRMakie's) ──
# RPR's VolumeMaterial has an `emission` input RPRMakie never wires — with it a
# volume GLOWS proportionally to its density grid, which is exactly what the
# radiated field needs. RAD_VOL_EMISSION[] carries the emission color into the
# conversion (volume! rejects unknown attributes). Identical to RPRMakie's
# version otherwise. Volumes only render on HybridPro/CPU (Northstar-GPU
# segfaults on volume grids).
const RAD_VOL_EMISSION = Ref(Makie.Vec4f(0, 0, 0, 0))
function RPRMakie.to_rpr_object(context, matsys, scene, plot::Makie.Volume)
    volume = plot.volume[]
    cube = RPR.VolumeCube(context)
    mi = (minimum(plot.x[]), minimum(plot.y[]), minimum(plot.z[]))
    w = (maximum(plot.x[]) - mi[1], maximum(plot.y[]) - mi[2], maximum(plot.z[]) - mi[3])
    m2 = Makie.Mat4f(w[1], 0, 0, 0, 0, w[2], 0, 0, 0, 0, w[3], 0, mi[1], mi[2], mi[3], 1)
    RPR.transform!(cube, convert(Makie.Mat4f, plot.model[]) * m2)
    lo, hi = extrema(volume)
    grid = RPR.VoxelGrid(context, (volume .- lo) ./ max(hi - lo, eps(Float32)))
    gridsampler = RPR.GridSamplerMaterial(matsys)
    gridsampler.data = grid
    color_sampler = RPR.ImageTextureMaterial(matsys)
    color_sampler.data = RPR.Image(context, reverse(to_colormap(plot.colormap[])))
    color_sampler.uv = gridsampler
    volmat = RPR.VolumeMaterial(matsys)
    volmat.density = Vec4f(plot.absorption[], 0, 0, 0)
    volmat.densitygrid = gridsampler
    volmat.color = color_sampler
    if RAD_VOL_EMISSION[][4] > 0
        volmat.emission = Vec4f(RAD_VOL_EMISSION[][1], RAD_VOL_EMISSION[][2],
            RAD_VOL_EMISSION[][3], 0)
    end
    RPR.set!(cube, volmat)
    return [cube]
end

# ── Pulse scalar + sampler (same as thomson_animation.jl) ──
pulse_scalar(E, B) = E[1]

function sample_pulse!(buf, fe, xs, ys, zs, t)
    Threads.@threads for j in eachindex(ys)
        y = ys[j]
        for (i, x) in enumerate(xs), (k, z) in enumerate(zs)
            E, B = fe([t, x, y, z])
            buf[k, i, j] = Float32(pulse_scalar(E, B))
        end
    end
    return buf
end

# ── Isosurface extraction ──
# vol is in SCENE order (dim1 = scene X = physics z grid, dim2 = scene Y =
# physics x, dim3 = scene Z = physics y); MC maps dim1↔x, so the grids are
# passed in that same order and the mesh comes out in scene coordinates.
function iso_mesh(vol, Xs, Ys, Zs, level)
    m = MC(vol; x = collect(Float32, Xs), y = collect(Float32, Ys), z = collect(Float32, Zs))
    march(m, level)
    isempty(m.triangles) && return nothing
    return MarchingCubes.makemesh(GeometryBasics, m)
end

# ── Parameters ──
iterations = parse(Int, get(ENV, "EDM_RPR_ITER", "200"))
r_electron = parse(Float64, get(ENV, "EDM_RPR_RADIUS", "0.14")) * λ
gpu = get(ENV, "EDM_RPR_RESOURCE", "gpu") == "gpu"
resource = gpu ?
    RPR.RPR_CREATION_FLAGS_ENABLE_GPU0 | RPR.RPR_CREATION_FLAGS_ENABLE_OPENCL :
    RPR.RPR_CREATION_FLAGS_ENABLE_CPU
laser_lit = get(ENV, "EDM_RPR_LIGHTS", "laser") == "laser"
dim_room = get(ENV, "EDM_RPR_LIGHTS", "") == "dim"
white_room = get(ENV, "EDM_RPR_BG", "dark") == "room"
ribbon_style = get(ENV, "EDM_RPR_RIBBONS", "emissive")
rad_levels = parse.(Float32, split(get(ENV, "EDM_RPR_RAD_LEVELS", "0.85,0.65"), ","))
# shells = emissive isosurfaces (works everywhere); volume = true emissive
# participating medium (HybridPro or CPU only — Northstar-GPU segfaults)
rad_style = get(ENV, "EDM_RPR_RAD_STYLE", "shells")

# Densified grids for the isosurface only — setup.jl's grids stay canonical for
# the GLMakie animation and the radiation precompute.
oversample = parse(Float64, get(ENV, "EDM_RPR_OVERSAMPLE", "2"))
nxr, nyr, nzr = round.(Int, oversample .* (nx, ny, nz))
xsr = LinRange(first(xs), last(xs), nxr)
ysr = LinRange(first(ys), last(ys), nyr)
zsr = LinRange(first(zs), last(zs), nzr)
vol_buf = Array{Float32}(undef, nzr, nxr, nyr)

# ── Stage C data: radiation cube + detector screen (loaded once; ceilings are
# global over the whole time series, matching thomson_animation.jl) ──
radiation_cube = get(ENV, "EDM_RADIATION_CUBE", joinpath(@__DIR__, "radiation_cube.jls"))
RAD = nothing
rad_win = nothing
rad_floor = 0.0f0
const rad_decades = 3.0f0
if isfile(radiation_cube) && get(ENV, "EDM_RPR_RADIATION", "1") == "1"
    @info "loading radiation cube $radiation_cube"
    RAD = deserialize(radiation_cube)
    _rs = filter(>(0.0f0), vec(@view RAD.rad[1:3:end, 1:5:end, 1:5:end, 1:5:end]))
    rad_ceil = partialsort!(_rs, max(1, round(Int, 0.005 * length(_rs))); rev = true)
    rad_floor = rad_ceil / exp10(rad_decades)
    edgewindow(n, frac = 0.1f0) = begin
        m = max(2, round(Int, frac * n))
        w = ones(Float32, n)
        for i in 1:m
            s = Float32(i - 1) / m
            w[i] = w[n + 1 - i] = s * s * (3 - 2s)
        end
        w
    end
    rad_win = reshape(edgewindow(size(RAD.rad, 2)), :, 1, 1) .*
        reshape(edgewindow(size(RAD.rad, 3)), 1, :, 1) .*
        reshape(edgewindow(size(RAD.rad, 4)), 1, 1, :)
end
rad_slice(t) = begin   # log-compressed, edge-faded transfer of the frame's cube slice
    f = clamp(searchsortedlast(RAD.frame_times, t), 1, length(RAD.frame_times))
    @. rad_win * clamp(
        log10(max($(@view RAD.rad[f, :, :, :]), rad_floor) / rad_floor) / rad_decades, 0.0f0, 1.0f0)
end

screen_file = get(ENV, "EDM_SCREEN_TIMESERIES", joinpath(@__DIR__, "screen_timeseries.jls"))
show_screen = get(ENV, "EDM_RPR_SCREEN", "1") == "1"
SCR = if show_screen && isfile(screen_file)
    @info "loading screen time series $screen_file"
    deserialize(screen_file)
elseif show_screen && RAD !== nothing
    @info "no dedicated screen data — using the radiation cube's +X edge slice"
    (; scr = RAD.rad[:, end, :, :], txs = RAD.txs, tys = RAD.tys,
        z_screen = last(RAD.slice_zs), frame_times = RAD.frame_times)
else
    nothing
end
scr_floor = 0.0f0
if SCR !== nothing
    _ss = filter(>(0.0f0), vec(@view SCR.scr[1:2:end, 1:3:end, 1:3:end]))
    scr_ceil = partialsort!(_ss, max(1, round(Int, 0.002 * length(_ss))); rev = true)
    global scr_floor = scr_ceil / exp10(3.0f0)
end
scr_slice(t) = begin
    f = clamp(searchsortedlast(SCR.frame_times, t), 1, length(SCR.frame_times))
    @. clamp(log10(max($(@view SCR.scr[f, :, :]), scr_floor) / scr_floor) / 3.0f0, 0.0f0, 1.0f0)
end
# camera span for the zoomed-out box+screen framing (0 = hero framing)
span_scr = SCR === nothing ? 0.0f0 : Float32(SCR.z_screen - first(zs))

# ── Camera path: hold the hero framing through approach/crossing/flash, then
# pull back to hold box + detector together while the burst propagates out.
# NB the wide framing is NOT the GLMakie formula (eye_x = cx + 0.5span puts the
# eye exactly in the plate's plane — the detector renders edge-on and
# vanishes); keep the eye short of z_screen so the display face reads obliquely.
smoothstep(x) = x * x * (3 - 2x)
function camera_for(t)
    hero = (Vec3f(2.5w₀, -8w₀, 3w₀), Vec3f(0, 0, 0))
    SCR === nothing && return hero
    cx = Float32((first(zs) + SCR.z_screen) / 2)
    wide = (Vec3f(cx + 0.1span_scr, -1.2span_scr, 0.32span_scr), Vec3f(cx, 0, 0))
    s = Float32(smoothstep(clamp((t - 5T0) / 9T0, 0, 1)))
    return ((1 - s) .* hero[1] .+ s .* wide[1], (1 - s) .* hero[2] .+ s .* wide[2])
end

# ── Scene ──
# max_recursion 20 (default 10): rays cross 10+ translucent ribbon layers, and
# recursion truncation visibly darkens the pulse interior at the default.
# EDM_RPR_PLUGIN=northstar|hybridpro|hybrid — Northstar = ground-truth path
# tracer (Linux GPU via OpenCL: AMD-only in practice, crashes on NVIDIA);
# HybridPro = Vulkan hardware-RT (vendor-neutral incl. NVIDIA RTX, GPU-only,
# reduced material feature set).
plugin = let p = lowercase(get(ENV, "EDM_RPR_PLUGIN", "northstar"))
    p == "hybridpro" ? RPR.HybridPro : p == "hybrid" ? RPR.Hybrid : RPR.Northstar
end
# Hybrid plugins: Vulkan HW-RT, reduced feature set — DIFFUSE material nodes and
# thin-surface/caustics glass inputs are unsupported, and (crucially)
# rprContextResolveFrameBuffer silently zeroes, so colorbuffer() returns black:
# read the RAW framebuffer and normalize RGB by alpha (= accumulated sample
# count), then gamma — see capture() below.
hybrid_rt = plugin !== RPR.Northstar
RPRMakie.activate!(; iterations, max_recursion = 20, plugin, resource)

function capture(screen)
    hybrid_rt || return colorbuffer(screen)
    colorbuffer(screen)   # drives the render loop; its resolved output is black on Hybrid
    raw = RPR.get_data(screen.framebuffer1)
    # calibration aid: dump the raw HDR buffer so exposure/tonemap variants can
    # be regenerated offline (lookdev/tonemap_sweep.jl) without re-rendering
    dump_raw = get(ENV, "EDM_RPR_DUMP_RAW", "")
    isempty(dump_raw) || serialize(dump_raw, (; raw, fb_size = screen.fb_size))
    # Hybrid also skips Northstar's tonemapping operator, so the HDR emission
    # values blow out under a plain clamp. LUMINANCE-based Reinhard, not
    # per-channel: per-channel compression drags saturated highlights to white
    # (the amber radiation shells desaturate to cream). EDM_RPR_EXPOSURE ≈ 0.2
    # lands near the Northstar photolinear look.
    ex = parse(Float32, get(ENV, "EDM_RPR_EXPOSURE", "0.2"))
    img = map(raw) do c
        a = max(c.alpha, 1.0f-6)
        r, g, b = ex * c.r / a, ex * c.g / a, ex * c.b / a
        L = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
        s = L > 0 ? (L / (1 + L)) / L : 0.0f0
        Makie.RGBf(clamp(s * r, 0, 1)^0.4545f0, clamp(s * g, 0, 1)^0.4545f0,
            clamp(s * b, 0, 1)^0.4545f0)
    end
    return permutedims(reshape(img, screen.fb_size))
end

function render_frame(t, outpath)
    sample_pulse!(vol_buf, fe, xsr, ysr, zsr, t)
    cmax = maximum(abs, vol_buf)
    epos = electron_positions(trajs, t)

    fig = Figure(size = (1280, 960), backgroundcolor = RGBf(0.055, 0.065, 0.09))
    # The scene lives in atomic units (w₀ ≈ 6e4), and RPR point lights fall off
    # physically as 1/r² — radiance must scale with the squared scene distance
    # or the lights contribute nothing and only the ambient term is visible.
    radiance = Float32((20w₀)^2)
    lights = if white_room
        # ambient IS the environment dome in RPRMakie (an explicit
        # EnvironmentLight gets displaced by the ambient env light — upstream
        # bug). Point lights are avoided: they cast hard fan shadows through
        # the layered translucent ribbons. The pulse still "lantern-projects"
        # its stripe pattern onto the walls — genuine light transport.
        # EDM_RPR_LIGHTS=dim: dimmed-lab variant — the chamber is lit mostly by
        # the pulse/radiation emission itself (lantern projections become the
        # feature; the flash visibly lights the room); the small ambient floor
        # keeps the tile grid readable in the dark tail act.
        dim_room ? [AmbientLight(RGBf(0.15, 0.15, 0.16))] :
            [AmbientLight(RGBf(0.9, 0.9, 0.92))]
    elseif laser_lit
        # the pulse's own emission is the key light — the electrons are lit by
        # the laser; keep only a weak fill + ambient so the dark side isn't dead
        [PointLight(RGBf(0.12radiance, 0.12radiance, 0.14radiance), Point3f(2w₀, -9w₀, 5w₀)),
            AmbientLight(RGBf(0.08, 0.08, 0.09))]
    else
        [
            # key: camera side, so the disk face we see is the lit one
            PointLight(RGBf(radiance, radiance, radiance), Point3f(2w₀, -9w₀, 5w₀)),
            PointLight(RGBf(0.4radiance, 0.4radiance, 0.5radiance), Point3f(-6w₀, 4w₀, -3w₀)),
            AmbientLight(RGBf(0.3, 0.3, 0.32)),
        ]
    end
    ax = LScene(fig[1, 1]; show_axis = false, scenekw = (lights = lights,))

    screen = RPRMakie.Screen(ax.scene)
    matsys = screen.matsys

    # HybridPro exposes render-quality presets (RadeonProRender_Baikal.h:
    # RPR_CONTEXT_RENDER_QUALITY = 0x1001, low=0 … ultra=3) that gate its
    # algorithmic shortcuts — EDM_RPR_QUALITY=low|medium|high|ultra (hybrid only)
    if hybrid_rt
        q = get(ENV, "EDM_RPR_QUALITY", "")
        if !isempty(q)
            qv = UInt32(findfirst(==(q), ["low", "medium", "high", "ultra"]) - 1)
            RPR.rprContextSetParameterByKey1u(screen.context.pointer,
                reinterpret(RPR.rpr_context_info, Int32(0x1001)), qv)
        end
    end

    if white_room
        # gray 3-plane corner (real geometry is the only way to get a light
        # backdrop). The three seams run along the scene axes: floor∩back =
        # propagation (X), back∩left = vertical (Z), floor∩left = Y; darker
        # strips mark them as explicit axis lines.
        C = (-10w₀, 8w₀, -2w₀)
        th = 0.05w₀
        # extents adapt to the zoomed-out box+screen framing so the pulled-back
        # camera stays inside the room
        xhi = max(32w₀, (SCR === nothing ? 0.0f0 : Float32(SCR.z_screen)) + 0.45f0 * span_scr)
        ylo_w = min(-16w₀, -1.2f0 * span_scr - 3w₀)
        zhi = max(18w₀, C[3] + 0.65f0 * span_scr)
        xlen, ylen, zlen = xhi - C[1], C[2] - ylo_w, zhi - C[3]
        # satin finish: diffuse base + broad rough sheen → bright high-key
        # walls. NB reflection_weight=1 is load-bearing — lower values shift
        # the PBR diffuse/specular balance and collapse the walls to dark gray.
        satin(col) = RPR.UberMaterial(matsys;
            color = to_color(col), diffuse_weight = Vec4f(1),
            reflection_color = Vec4f(1), reflection_weight = Vec4f(1),
            reflection_roughness = Vec4f(0.45),
            reflection_mode = RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
            reflection_ior = Vec4f(1.5))
        for (slab, col) in (
                (Rect3f(Point3f(C[1], ylo_w, C[3] - th), Vec3f(xlen, ylen, th)), RGBf(0.6, 0.6, 0.62)),   # floor
                (Rect3f(Point3f(C[1], C[2], C[3]), Vec3f(xlen, th, zlen)), RGBf(0.72, 0.72, 0.74)),       # back
                (Rect3f(Point3f(C[1] - th, ylo_w, C[3]), Vec3f(th, ylen, zlen)), RGBf(0.66, 0.66, 0.68)), # left
            )
            mesh!(ax, slab; color = col, material = satin(col))
        end
        s = 0.09w₀
        axis_gray = RGBf(0.32, 0.33, 0.36)
        for a in (
                Rect3f(Point3f(C[1], C[2] - s, C[3]), Vec3f(xlen, s, s)),   # X: propagation
                Rect3f(Point3f(C[1], ylo_w, C[3]), Vec3f(s, ylen, s)),      # Y
                Rect3f(Point3f(C[1], C[2] - s, C[3]), Vec3f(s, s, zlen)),   # Z
            )
            mesh!(ax, a; color = axis_gray, material = flat_material(matsys, axis_gray))
        end
        # thin square grid, spacing 2w₀, registered to the origin: reads as
        # calibrated space and keeps the walls visually uniform
        g = 0.02w₀
        grid_gray = RGBf(0.5, 0.5, 0.53)
        gmat = flat_material(matsys, grid_gray)
        gridmults(lo, hi) = (2 * ceil(Int, lo / 2w₀):2:2 * floor(Int, hi / 2w₀)) .* w₀
        strips = Rect3f[]
        for xg in gridmults(C[1], xhi)      # floor (∥Y) and back wall (∥Z)
            push!(strips, Rect3f(Point3f(xg - g / 2, ylo_w, C[3]), Vec3f(g, ylen, g)))
            push!(strips, Rect3f(Point3f(xg - g / 2, C[2] - g, C[3]), Vec3f(g, g, zlen)))
        end
        for yg in gridmults(ylo_w, C[2])    # floor (∥X) and left wall (∥Z)
            push!(strips, Rect3f(Point3f(C[1], yg - g / 2, C[3]), Vec3f(xlen, g, g)))
            push!(strips, Rect3f(Point3f(C[1], yg - g / 2, C[3]), Vec3f(g, g, zlen)))
        end
        for zg in gridmults(C[3], zhi)      # back wall (∥X) and left wall (∥Y)
            push!(strips, Rect3f(Point3f(C[1], C[2] - g, zg - g / 2), Vec3f(xlen, g, g)))
            push!(strips, Rect3f(Point3f(C[1], ylo_w, zg - g / 2), Vec3f(g, ylen, g)))
        end
        for st in strips
            mesh!(ax, st; color = grid_gray, material = gmat)
        end
    end

    # Wavefront ribbons: translucent double-sided emissive glow in the :balance
    # endpoints' red/blue. Double-sided because marching-cubes normals follow
    # the field gradient, so the −lobe surface faces away from the camera;
    # translucent (Uber transparency input — color alpha is a no-op in RPR) so
    # the electron disk reads through the pulse.
    function ribbon_material(c)
        ribbon_style == "plastic" &&
            return RPR.Plastic(matsys; color = to_color(c), transparency = Vec4f(0.35))
        # thin-sheet tinted glass: the isosurfaces are open sheets, so
        # refraction_thin_surface is the physical model; tint both transmission
        # and reflection or the white reflections wash the lobes to neutral
        # Hybrid plugins reject thin_surface/caustics AND the RPR.Glass preset
        # itself (its defaults include reflection_mode/thin_surface/caustics) —
        # build the glass Uber from scratch there with only supported inputs
        tinted_glass() = hybrid_rt ?
            RPR.UberMaterial(matsys;
                color = Vec4f(0), diffuse_weight = Vec4f(0),
                reflection_color = Vec4f(0.3 + 0.7c.r, 0.3 + 0.7c.g, 0.3 + 0.7c.b, 1),
                reflection_weight = Vec4f(1), reflection_roughness = Vec4f(0),
                reflection_ior = Vec4f(1.5),
                refraction_color = Vec4f(c.r, c.g, c.b, 1),
                refraction_weight = Vec4f(1), refraction_roughness = Vec4f(0),
                refraction_ior = Vec4f(1.5)) :
            RPR.Glass(matsys;
                refraction_color = Vec4f(c.r, c.g, c.b, 1),
                reflection_color = Vec4f(0.3 + 0.7c.r, 0.3 + 0.7c.g, 0.3 + 0.7c.b, 1),
                refraction_thin_surface = true,
                refraction_caustics = false,
            )
        ribbon_style == "glass" && return tinted_glass()
        if ribbon_style == "glassglow"
            # stronger inner glow when the room is lit BY the pulse (dim mode)
            emc, emw = dim_room ? (3.0f0, 0.35f0) : (1.5f0, 0.25f0)
            # BLEND nodes are unsupported on the Hybrid plugins — fold the
            # emissive layer into the glass Uber itself (same node carries
            # refraction + emission); Northstar keeps the LayerMaterial blend
            hybrid_rt || return RPR.LayerMaterial(tinted_glass(),
                RPR.EmissiveMaterial(matsys; color = Vec4f(emc * c.r, emc * c.g, emc * c.b, 1));
                weight = Vec4f(emw))
            g = tinted_glass()
            g.emission_color = Vec4f(emc * c.r, emc * c.g, emc * c.b, 1)
            g.emission_weight = Vec4f(emw)
            return g
        end
        # 5× emission in laser mode: bright enough to light the electrons, low
        # enough that tone mapping doesn't bleach the stripe colors; 3× when
        # the room/studio provides the light
        mult = ((laser_lit && !white_room) || dim_room) ? 5 : 3
        # NB: mode enums must be passed RAW (not UInt(...)-wrapped): RPR.jl
        # routes plain integers through the float setter (Vec4f coercion),
        # which Northstar tolerates but HybridPro rejects with
        # INVALID_PARAMETER_TYPE; enum values dispatch to the correct U setter.
        return RPR.UberMaterial(matsys;
            diffuse_weight = Vec4f(0),
            reflection_weight = Vec4f(0),
            emission_color = Vec4f(mult * c.r, mult * c.g, mult * c.b, 1),
            emission_weight = Vec4f(1),
            emission_mode = RPR.RPR_UBER_MATERIAL_EMISSION_MODE_DOUBLESIDED,
            transparency = Vec4f(0.55),
        )
    end

    for (level, c) in ((+0.5f0 * cmax, RGBf(0.85, 0.25, 0.15)),
                       (-0.5f0 * cmax, RGBf(0.2, 0.4, 0.95)))
        msh = iso_mesh(vol_buf, zsr, xsr, ysr, level)
        msh === nothing && continue
        mesh!(ax, msh; color = c, material = ribbon_material(c))
    end

    # Stage C: radiated far-field shells — nested emissive isosurfaces of the
    # log-compressed |E_far|² (RPR volumes segfault the GPU). Levels live in the
    # [0,1] transfer: 0.85 ≈ 35% and 0.65 ≈ 9% of peak intensity — much lower
    # levels wrap the whole box in faint wisps and bury the scene.
    # volume style is Northstar-CPU ONLY: Northstar-GPU segfaults on volume
    # grids and HybridPro SILENTLY IGNORES them (verified against a CPU control
    # — identical scene renders the volume on CPU, background-only on hybrid).
    # The emission input also showed no effect on CPU; shells remain the
    # production radiation representation.
    use_rad_volume = rad_style == "volume" && !hybrid_rt && !gpu
    rad_style == "volume" && !use_rad_volume &&
        @warn "EDM_RPR_RAD_STYLE=volume requires EDM_RPR_RESOURCE=cpu with Northstar — rendering shells instead"
    if RAD !== nothing && use_rad_volume
        u = rad_slice(t)
        RAD_VOL_EMISSION[] = Makie.Vec4f(2.4, 1.7, 0.9, 1)
        volume!(ax, (first(RAD.slice_zs), last(RAD.slice_zs)),
            (first(RAD.txs), last(RAD.txs)), (first(RAD.tys), last(RAD.tys)), u;
            algorithm = :absorption, absorption = 6.0f0, colormap = :afmhot)
    elseif RAD !== nothing
        u = rad_slice(t)
        for (lvl, emis, transp) in ((rad_levels[1], Vec4f(6.0, 5.0, 3.2, 1), 0.4f0),
                                    (rad_levels[2], Vec4f(2.2, 1.4, 0.8, 1), 0.72f0))
            m = MC(u; x = collect(Float32, RAD.slice_zs), y = collect(Float32, RAD.txs),
                z = collect(Float32, RAD.tys))
            march(m, lvl)
            isempty(m.triangles) && continue
            msh = MarchingCubes.makemesh(GeometryBasics, m)
            shellmat = RPR.UberMaterial(matsys;
                diffuse_weight = Vec4f(0), reflection_weight = Vec4f(0),
                emission_color = emis, emission_weight = Vec4f(1),
                emission_mode = RPR.RPR_UBER_MATERIAL_EMISSION_MODE_DOUBLESIDED,
                transparency = Vec4f(transp))
            mesh!(ax, msh; color = RGBf(1, 0.85, 0.5), material = shellmat)
        end
    end

    # Detector screen plate. NOT surface! — RPRMakie 0.10.13's Surface
    # conversion still reads the removed :calculated_colors attribute (KeyError
    # under Makie 0.24). Instead: colormap the image on the CPU and put it on a
    # manual quad as an emissive texture — a glowing detector display.
    if SCR !== nothing
        su = scr_slice(t)
        Xs_scr = Float32(SCR.z_screen)
        ylo, yhi = Float32(first(SCR.txs)), Float32(last(SCR.txs))
        zlo, zhi = Float32(first(SCR.tys)), Float32(last(SCR.tys))
        cmap = Makie.to_colormap(:magma)
        scr_img = [RGBf(cmap[clamp(round(Int, v * (length(cmap) - 1)) + 1, 1, length(cmap))])
                   for v in su]
        pts = Point3f[(Xs_scr, ylo, zlo), (Xs_scr, yhi, zlo), (Xs_scr, yhi, zhi), (Xs_scr, ylo, zhi)]
        fcs = [GeometryBasics.GLTriangleFace(1, 2, 3), GeometryBasics.GLTriangleFace(1, 3, 4)]
        uvs = GeometryBasics.Vec2f[(0, 0), (1, 0), (1, 1), (0, 1)]
        nrm = [GeometryBasics.Vec3f(-1, 0, 0) for _ in 1:4]
        plate = GeometryBasics.Mesh(pts, fcs; uv = uvs, normal = nrm)
        scr_tex = RPR.Texture(matsys, scr_img')
        scr_mat = RPR.UberMaterial(matsys;
            color = Vec4f(0.02, 0.02, 0.03, 1), diffuse_weight = Vec4f(0.15),
            reflection_weight = Vec4f(0),
            emission_weight = Vec4f(1),
            emission_mode = RPR.RPR_UBER_MATERIAL_EMISSION_MODE_DOUBLESIDED)
        scr_mat.emission_color = scr_tex
        mesh!(ax, plate; color = :black, material = scr_mat)
    end

    # Electrons: gold — diffuse base under a metallic specular lobe, so the
    # disk reads gold under direct light instead of only mirroring the sky.
    gold = RPR.UberMaterial(matsys;
        color = to_color(:gold),
        diffuse_weight = Vec4f(white_room ? 0.4 : 0.6),
        reflection_color = Vec4f(1, 0.85, 0.45, 1),
        reflection_weight = Vec4f(1),
        reflection_roughness = Vec4f(0.25),
        reflection_metalness = Vec4f(1),
        reflection_mode = RPR.RPR_UBER_MATERIAL_IOR_MODE_METALNESS,
    )
    meshscatter!(ax, epos; markersize = r_electron, color = :gold, material = gold)

    eye, lookat = camera_for(t)
    update_cam!(ax.scene, eye, lookat, Vec3f(0, 0, 1))

    t_render = @elapsed img = capture(screen)
    save(outpath, img)
    return t_render
end

# ── Entry: single still (EDM_RPR_T) or frame range (EDM_RPR_FRAMES) ──
frames_spec = get(ENV, "EDM_RPR_FRAMES", "")
if isempty(frames_spec)
    t_snap = parse(Float64, get(ENV, "EDM_RPR_T", "0.0")) * T0
    outfile = get(ENV, "EDM_RPR_OUT", joinpath(@__DIR__, "rpr_frame.png"))
    @info "rendering still at t = $(t_snap / T0) T0, $(iterations) iterations on $(gpu ? "GPU" : "CPU")"
    tr = render_frame(t_snap, outfile)
    @info "rendered in $(round(tr; digits = 1)) s — saved $outfile"
else
    rng = frames_spec == "all" ? (1:length(play_times)) :
        (:)(parse.(Int, split(frames_spec, ":"))...)
    outdir = get(ENV, "EDM_RPR_OUTDIR", joinpath(@__DIR__, "rpr_frames"))
    mkpath(outdir)
    @info "rendering frames $(first(rng))..$(last(rng)) of $(length(play_times)) into $outdir"
    function render_range(rng, outdir)
        total = 0.0
        done = 0
        for i in rng
            outpath = joinpath(outdir, @sprintf("rpr_%04d.png", i))
            isfile(outpath) && continue   # resumable
            tr = render_frame(play_times[i], outpath)
            done += 1
            total += tr
            eta = (length(rng) - (i - first(rng) + 1)) * total / done
            @info @sprintf("frame %d/%d  t = %+.2f T0  %.1f s  (mean %.1f s, ETA %.0f min)",
                i, last(rng), play_times[i] / T0, tr, total / done, eta / 60)
        end
        return done
    end
    n = render_range(rng, outdir)
    @info "done: $n frames rendered"
end
