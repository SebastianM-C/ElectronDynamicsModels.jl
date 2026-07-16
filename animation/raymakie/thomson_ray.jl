# RayMakie/Hikari (Lava Vulkan) renderer for the Thomson-scattering scene —
# the experiment answering "does RayMakie beat RPRMakie/HybridPro for the
# production animation?". Consumes the plain-array frame payloads produced by
# animation/raymakie/precompute_frames.jl (the physics stack cannot share this
# env: Hikari pins StructArrays 0.6, SciMLBase 3 needs 0.7 — see Project.toml).
#
# What RayMakie changes vs the RPR pipeline:
#   * observables are wired: the scene is built ONCE and mutated per frame
#     (EDM_RAY_REUSE=1, default) instead of rebuilt with a fresh context —
#     EDM_RAY_REUSE=0 rebuilds per frame as a control
#   * colorbuffer() returns a tonemapped image directly (exposure/tonemap/gamma
#     kwargs) — no raw-framebuffer capture / manual Reinhard
#   * materials are typed Hikari structs — glassglow is ONE node: glass BSDF +
#     area-light emission on the same faces (RPR needed LayerMaterial / a
#     folded Uber, with different builds per plugin)
#   * EDM_RAY_STYLE=volume renders the radiation as a true emissive medium
#     (RGBGridMedium) — the thing RPR could never do on GPU
#
# Headless (no X session): GLFW.jl hardcodes X11 on Linux, so run under a
# virtual compositor —
#   kwin_wayland --virtual --no-global-shortcuts --socket=wayland-edm &
#   export WAYLAND_DISPLAY=wayland-edm JULIA_GLFW_PLATFORM=wayland
#
# Run:  julia --startup=no --project=animation/raymakie \
#         animation/raymakie/thomson_ray.jl
# Knobs:
#   EDM_RAY_T=0.0             still time (T0) — reads still_<t>.jls from the cache
#   EDM_RAY_FRAMES=a:b|all    frame range — reads frame_%04d.jls (resumable out)
#   EDM_RAY_CACHE=dir         payload dir (default animation/raymakie/frame_cache)
#   EDM_RAY_OUT / EDM_RAY_OUTDIR   output png / frames dir
#   EDM_RAY_SPP=64 EDM_RAY_MAX_DEPTH=16 EDM_RAY_HW=1 EDM_RAY_RES=1280x960
#   EDM_RAY_REUSE=1           build scene once + mutate observables per frame
#   EDM_RAY_STYLE=mesh|volume radiation stripes as glass iso-meshes (production
#                             look) or as an emissive RGBGridMedium
#   EDM_RAY_PULSE_STYLE=mesh|volume  same for the laser pulse
#   EDM_RAY_BG=room|dark      gray corner room (production) or dark backdrop
#   EDM_RAY_AMBIENT=1.0       room dome intensity (Hikari scale — RPR's 7.0 was
#                             tuned against its Reinhard capture, do not copy)
#   EDM_RAY_SOFTBOX=1.5       ceiling panel Le (0 = off)
#   EDM_RAY_EMIS=1.0 EDM_RAY_EMIS_COOL=0.25 EDM_RAY_GLOW=0.25   glassglow
#   EDM_RAY_ELECTRON_ROUGH=0.45
#   EDM_RAY_RAD_TINT=0.5 EDM_RAY_RAD_COLORS="1.0,0.75,0.2;0.55,0.3,0.9"
#   EDM_RAY_VOL_LE=4.0 EDM_RAY_VOL_SIGMA=8.0   volume-mode emission/absorption
#   EDM_RAY_EXPOSURE=1.0 EDM_RAY_TONEMAP=aces EDM_RAY_GAMMA=2.2

using Makie
using RayMakie
using Hikari
using Lava
using GPUSelect
import GeometryBasics
using FileIO
using Serialization
using Printf

# ── Parameters ──
spp = parse(Int, get(ENV, "EDM_RAY_SPP", "64"))
max_depth = parse(Int, get(ENV, "EDM_RAY_MAX_DEPTH", "16"))
use_hw = get(ENV, "EDM_RAY_HW", "1") == "1"
res = Tuple(parse.(Int, split(get(ENV, "EDM_RAY_RES", "1280x960"), "x")))
reuse_scene = get(ENV, "EDM_RAY_REUSE", "1") == "1"
rad_style3d = get(ENV, "EDM_RAY_STYLE", "mesh")          # mesh | volume
pulse_style3d = get(ENV, "EDM_RAY_PULSE_STYLE", "mesh")  # mesh | volume
white_room = get(ENV, "EDM_RAY_BG", "room") == "room"
ambient_room = parse(Float32, get(ENV, "EDM_RAY_AMBIENT", "1.0"))
sbox = parse(Float32, get(ENV, "EDM_RAY_SOFTBOX", "1.5"))
emis_scale = parse(Float32, get(ENV, "EDM_RAY_EMIS", "1.0"))
emis_cool = parse(Float32, get(ENV, "EDM_RAY_EMIS_COOL", "0.25"))
glow_w = parse(Float32, get(ENV, "EDM_RAY_GLOW", "0.25"))
electron_rough = parse(Float32, get(ENV, "EDM_RAY_ELECTRON_ROUGH", "0.45"))
rad_tint = parse(Float32, get(ENV, "EDM_RAY_RAD_TINT", "0.5"))
rad_colors = let s = get(ENV, "EDM_RAY_RAD_COLORS", "1.0,0.75,0.2;0.55,0.3,0.9")
    a, b = split(s, ";")
    (RGBf(parse.(Float32, split(a, ","))...), RGBf(parse.(Float32, split(b, ","))...))
end
vol_le = parse(Float32, get(ENV, "EDM_RAY_VOL_LE", "2.0"))
vol_sigma = parse(Float32, get(ENV, "EDM_RAY_VOL_SIGMA", "6.0"))
vol_floor = parse(Float32, get(ENV, "EDM_RAY_VOL_FLOOR", "0.2"))
vol_gamma = parse(Float32, get(ENV, "EDM_RAY_VOL_GAMMA", "1.5"))
# the pulse envelope core sits at |s|≈1 over a large volume (unlike the thin
# radiation crests) — cut it at the mesh look's ±0.5 isolevel or it saturates
pulse_vol_floor = parse(Float32, get(ENV, "EDM_RAY_PULSE_VOL_FLOOR", "0.45"))
pulse_vol_le = parse(Float32, get(ENV, "EDM_RAY_PULSE_VOL_LE", "1.0"))
exposure = parse(Float32, get(ENV, "EDM_RAY_EXPOSURE", "1.0"))
tonemap = Symbol(get(ENV, "EDM_RAY_TONEMAP", "aces"))
gamma = parse(Float32, get(ENV, "EDM_RAY_GAMMA", "2.2"))

cache_dir = get(ENV, "EDM_RAY_CACHE", joinpath(@__DIR__, "frame_cache"))
ST = deserialize(joinpath(cache_dir, "static.jls"))
const sw = ST.sw
span_scr = ST.span_scr

DEVICE = GPUSelect.Backend(:Lava)
hw_accel = use_hw && DEVICE isa Lava.LavaBackend &&
    Lava.vk_context().rt_pipeline_properties !== nothing
integrator = Hikari.VolPath(; samples = spp, max_depth, hw_accel,
    regularize = true, max_component_value = 10.0f0)
@info "RayMakie" DEVICE hw_accel spp max_depth res reuse_scene rad_style3d pulse_style3d

# ── Payload reconstruction ──
rebuild_mesh(pm) = pm === nothing ? nothing :
    GeometryBasics.Mesh(
        [GeometryBasics.Point3f(p...) for p in pm.pts],
        [GeometryBasics.GLTriangleFace(f...) for f in pm.fcs];
        normal = [GeometryBasics.Vec3f(n...) for n in pm.nrm])
rebuild_img(pi) = [RGBf(c...) for c in pi]

# degenerate stand-in far below the floor: a lobe can vanish (payload nothing),
# but a live plot needs SOME mesh to keep its observable graph alive
const NULL_MESH = GeometryBasics.Mesh(
    [GeometryBasics.Point3f(0, 0, -1e4), GeometryBasics.Point3f(1e-3, 0, -1e4),
        GeometryBasics.Point3f(0, 1e-3, -1e4)],
    [GeometryBasics.GLTriangleFace(1, 2, 3)];
    normal = [GeometryBasics.Vec3f(0, 0, 1) for _ in 1:3])
mesh_or_null(m) = m === nothing ? NULL_MESH : m

# ── Hikari materials ──
spec(c::RGBf) = Hikari.RGBSpectrum(c.r, c.g, c.b)
pale(c, tint) = RGBf(1 - tint * (1 - c.r), 1 - tint * (1 - c.g), 1 - tint * (1 - c.b))

# Emissive() applies pbrt's photometric normalization to scale — route the
# emission NamedTuple through it so brightness matches Emissive materials
function emission_info(c; scale = 1.0f0, two_sided = true)
    em = Hikari.Emissive(Le = (c.r, c.g, c.b), scale = scale, two_sided = two_sided)
    (Le = em.Le, scale = em.scale, two_sided = em.two_sided)
end
emissive_mat(c; scale = 1.0f0, two_sided = true) =
    Hikari.MediumInterface(Hikari.Emissive(; Le = (c.r, c.g, c.b), scale, two_sided))

# tinted glass: transmission carries the lobe color, reflections lightly tinted
# (the RPR recipe: reflection_color = 0.3 + 0.7c). The iso-surface sheets are
# OPEN surfaces — a full Dielectric tracks inside/outside and can TIR-loop to
# the depth budget on every path, so EDM_RAY_GLASS picks the interface model:
#   dielectric  index-1.5 glass (Fresnel + refraction; slowest, deepest look)
#   thin        ThinDielectric — pbrt's thin-surface model, RPR's
#               refraction_thin_surface=true equivalent (untinted)
#   interface   index-1.0 tinted pass-through (no bend, no TIR; fastest)
glass_kind = get(ENV, "EDM_RAY_GLASS", "dielectric")
tinted_glass(c) =
    glass_kind == "thin" ? Hikari.ThinDielectric(eta = 1.5f0) :
    glass_kind == "interface" ? Hikari.Dielectric(
        Kr = Hikari.RGBSpectrum(0.0f0), Kt = spec(c), roughness = 0.0f0, index = 1.0f0) :
    Hikari.Dielectric(
        Kr = spec(RGBf(0.3f0 + 0.7f0 * c.r, 0.3f0 + 0.7f0 * c.g, 0.3f0 + 0.7f0 * c.b)),
        Kt = spec(c), roughness = 0.0f0, index = 1.5f0)

# glassglow: ONE node — glass BSDF + folded area-light emission
function glassglow(c; emis = 1.0f0, glow = glow_w, fade = 0.0f0)
    e = emis * (1 - fade)
    c.b > c.r && (e *= emis_cool)   # cool lobe glows less (Fresnel depth cues)
    Hikari.MediumInterface(tinted_glass(c);
        emission = emission_info(RGBf(e * c.r, e * c.g, e * c.b); scale = glow))
end

gold_electron() = Hikari.Gold(roughness = electron_rough)
flat_mat(c) = Hikari.Diffuse(Kd = spec(c))
satin_mat(c) = Hikari.CoatedDiffuse(reflectance = spec(c), roughness = 0.45f0)
# pbrt "interface" semantics — a Dielectric boundary (even Kr=0/Kt=1/index=1)
# makes SHADOW rays treat the cube as opaque and paints it as a shadowed block
# (see RayMakie plots/volume.jl); NullMaterial only fires the medium swap
transparent_boundary() = Hikari.NullMaterial()

# signed volume → emissive RGBGridMedium: ± wavefront crests in the given
# palette. Only |s| above vol_floor participates — pbrt's volumetric emission
# contributes as σ_a·Le, and without the cut the box-filling numerical floor
# goes optically thick over the ~30 sw domain (renders as a white fog block).
function signed_medium(s, bounds; le_scale = vol_le, sigma = vol_sigma,
        colors = rad_colors, floor = vol_floor, gamma = vol_gamma)
    dims = size(s)
    σ_a = Array{Hikari.RGBSpectrum, 3}(undef, dims)
    Le = Array{Hikari.RGBSpectrum, 3}(undef, dims)
    # σ_s_grid=nothing is mishandled upstream (whole box renders as uniform
    # fog — see mwe_medium.jl); an explicit zero grid restores emission-only
    σ_s = fill(Hikari.RGBSpectrum(0.0f0), dims)
    pos, neg = colors
    @inbounds for i in eachindex(s)
        v = s[i]
        a = abs(v)
        w = a <= floor ? 0.0f0 : ((a - floor) / (1 - floor))^gamma
        c = v ≥ 0 ? pos : neg
        σ_a[i] = Hikari.RGBSpectrum(w)
        Le[i] = Hikari.RGBSpectrum(w * c.r, w * c.g, w * c.b)
    end
    bmin, bmax = bounds
    Hikari.RGBGridMedium(; σ_a_grid = σ_a, σ_s_grid = σ_s, Le_grid = Le,
        sigma_scale = sigma, Le_scale = le_scale, g = 0.0f0,
        bounds = Hikari.Bounds3(GeometryBasics.Point3f(bmin...), GeometryBasics.Point3f(bmax...)))
end
vol_boundary_mat(s, bounds; kw...) = Hikari.MediumInterface(transparent_boundary();
    inside = signed_medium(s, bounds; kw...), outside = nothing)

pulse_colors = (RGBf(0.85, 0.25, 0.15), RGBf(0.2, 0.4, 0.95))
rad_pale = (pale(rad_colors[1], rad_tint), pale(rad_colors[2], rad_tint))

screen_material(img) = Hikari.MediumInterface(
    Hikari.Diffuse(Kd = spec(RGBf(0.02, 0.02, 0.03)));
    emission = (Le = Hikari.Texture(img'), scale = emission_info(RGBf(1, 1, 1)).scale,
        two_sided = true))

load_payload(path) = deserialize(path)

# ── Scene assembly ──
function build_scene(fc)
    lights = if white_room
        Makie.AbstractLight[Makie.AmbientLight(RGBf(ambient_room, ambient_room, 1.022f0 * ambient_room))]
    else
        radiance = 30.0f0
        Makie.AbstractLight[
            Makie.PointLight(RGBf(0.12radiance, 0.12radiance, 0.14radiance), Point3f(2sw, -9sw, 5sw)),
            Makie.AmbientLight(RGBf(0.08, 0.08, 0.09))]
    end
    scene = Scene(; size = res, backgroundcolor = RGBf(0.055, 0.065, 0.09), lights)
    cam3d!(scene)
    cc = Makie.cameracontrols(scene)
    cc.settings.center[] = false   # bbox re-centering steps when a lobe pops (RPR fix, still applies)

    if white_room
        C = (-10sw, 8sw, -2sw)
        th = 0.05sw
        xhi = max(32sw, (ST.screen === nothing ? 0.0f0 : ST.screen.x) + 0.45f0 * span_scr)
        ylo_w = min(-16sw, -1.2f0 * span_scr - 3sw)
        zhi = max(18sw, C[3] + 0.65f0 * span_scr)
        xlen, ylen, zlen = xhi - C[1], C[2] - ylo_w, zhi - C[3]
        for (slab, col) in (
                (Rect3f(Point3f(C[1], ylo_w, C[3] - th), Vec3f(xlen, ylen, th)), RGBf(0.6, 0.6, 0.62)),
                (Rect3f(Point3f(C[1], C[2], C[3]), Vec3f(xlen, th, zlen)), RGBf(0.72, 0.72, 0.74)),
                (Rect3f(Point3f(C[1] - th, ylo_w, C[3]), Vec3f(th, ylen, zlen)), RGBf(0.66, 0.66, 0.68)),
            )
            mesh!(scene, slab; color = col, material = satin_mat(col))
        end
        if sbox > 0
            panel = Rect3f(Point3f(-4sw, -8sw, 12sw), Vec3f(16sw, 12sw, 0.05sw))
            mesh!(scene, panel; color = RGBf(1, 1, 1),
                material = emissive_mat(RGBf(1, 1, 1.03f0); scale = sbox))
        end
        s = 0.09sw
        axis_gray = RGBf(0.32, 0.33, 0.36)
        for a in (
                Rect3f(Point3f(C[1], C[2] - s, C[3]), Vec3f(xlen, s, s)),
                Rect3f(Point3f(C[1], ylo_w, C[3]), Vec3f(s, ylen, s)),
                Rect3f(Point3f(C[1], C[2] - s, C[3]), Vec3f(s, s, zlen)),
            )
            mesh!(scene, a; color = axis_gray, material = flat_mat(axis_gray))
        end
        g = 0.02sw
        grid_gray = RGBf(0.5, 0.5, 0.53)
        gridmults(lo, hi) = (2 * ceil(Int, lo / 2sw):2:2 * floor(Int, hi / 2sw)) .* sw
        strips = Rect3f[]
        for xg in gridmults(C[1], xhi)
            push!(strips, Rect3f(Point3f(xg - g / 2, ylo_w, C[3]), Vec3f(g, ylen, g)))
            push!(strips, Rect3f(Point3f(xg - g / 2, C[2] - g, C[3]), Vec3f(g, g, zlen)))
        end
        for yg in gridmults(ylo_w, C[2])
            push!(strips, Rect3f(Point3f(C[1], yg - g / 2, C[3]), Vec3f(xlen, g, g)))
            push!(strips, Rect3f(Point3f(C[1], yg - g / 2, C[3]), Vec3f(g, g, zlen)))
        end
        for zg in gridmults(C[3], zhi)
            push!(strips, Rect3f(Point3f(C[1], C[2] - g, zg - g / 2), Vec3f(xlen, g, g)))
            push!(strips, Rect3f(Point3f(C[1], ylo_w, zg - g / 2), Vec3f(g, ylen, g)))
        end
        for st in strips
            mesh!(scene, st; color = grid_gray, material = flat_mat(grid_gray))
        end
    end

    # laser wavefront ribbons (± signed pulse isosurfaces, glassglow)
    pulse_plots = if pulse_style3d == "mesh"
        map((1, 2)) do i
            c = pulse_colors[i]
            mesh!(scene, mesh_or_null(rebuild_mesh(fc.pulse_meshes[i])); color = c,
                material = glassglow(c; emis = emis_scale, fade = fc.pf))
        end
    else
        nothing
    end
    # radiation stripes (± signed Ex_far isosurfaces, pale tinted glass)
    rad_plots = if rad_style3d == "mesh"
        map((1, 2)) do i
            mesh!(scene, mesh_or_null(rebuild_mesh(fc.rad_meshes[i])); color = rad_pale[i],
                material = tinted_glass(rad_pale[i]))
        end
    else
        nothing
    end
    # volume variants: emissive media inside transparent boxes
    pulse_vol_plot = if pulse_style3d == "volume" && fc.pulse_vol !== nothing
        bmin, bmax = ST.pulse_vol_bounds
        mesh!(scene,
            GeometryBasics.normal_mesh(Rect3f(Point3f(bmin...), Vec3f((bmax .- bmin)...)));
            material = vol_boundary_mat(fc.pulse_vol, ST.pulse_vol_bounds;
                le_scale = pulse_vol_le * emis_scale * (1 - fc.pf), colors = pulse_colors,
                floor = pulse_vol_floor, gamma = 2.0f0))
    else
        nothing
    end
    rad_vol_plot = if rad_style3d == "volume" && fc.rad_vol !== nothing
        bmin, bmax = ST.rad_vol_bounds
        mesh!(scene,
            GeometryBasics.normal_mesh(Rect3f(Point3f(bmin...), Vec3f((bmax .- bmin)...)));
            material = vol_boundary_mat(fc.rad_vol, ST.rad_vol_bounds))
    else
        nothing
    end

    # detector plate: textured emissive quad (manual mesh, same as RPR)
    screen_plot = if ST.screen !== nothing && fc.scr_img !== nothing
        S = ST.screen
        pts = GeometryBasics.Point3f[(S.x, S.ylo, S.zlo), (S.x, S.yhi, S.zlo),
            (S.x, S.yhi, S.zhi), (S.x, S.ylo, S.zhi)]
        fcs = [GeometryBasics.GLTriangleFace(1, 2, 3), GeometryBasics.GLTriangleFace(1, 3, 4)]
        uvs = GeometryBasics.Vec2f[(0, 0), (1, 0), (1, 1), (0, 1)]
        nrm = [GeometryBasics.Vec3f(-1, 0, 0) for _ in 1:4]
        plate = GeometryBasics.Mesh(pts, fcs; uv = uvs, normal = nrm)
        mesh!(scene, plate; color = :black, material = screen_material(rebuild_img(fc.scr_img)))
    else
        nothing
    end

    electrons = meshscatter!(scene, [GeometryBasics.Point3f(p...) for p in fc.epos];
        marker = GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1.0f0),
        markersize = Vec3f(ST.r_electron_scn), color = :gold, material = gold_electron())

    eye, lookat = fc.cam
    update_cam!(scene, Vec3f(eye...), Vec3f(lookat...), Vec3f(0, 0, 1))

    return (; scene, pulse_plots, rad_plots, pulse_vol_plot, rad_vol_plot,
        screen_plot, electrons)
end

# in-place frame update: mutate plot observables, never rebuild (the RayMakie
# capability RPRMakie lacks)
function update_scene!(S, fc)
    if S.pulse_plots !== nothing
        for i in (1, 2)
            S.pulse_plots[i].mesh[] = mesh_or_null(rebuild_mesh(fc.pulse_meshes[i]))
            S.pulse_plots[i].material[] =
                glassglow(pulse_colors[i]; emis = emis_scale, fade = fc.pf)
        end
    end
    if S.rad_plots !== nothing
        for i in (1, 2)
            S.rad_plots[i].mesh[] = mesh_or_null(rebuild_mesh(fc.rad_meshes[i]))
        end
    end
    S.pulse_vol_plot !== nothing && fc.pulse_vol !== nothing &&
        (S.pulse_vol_plot.material[] = vol_boundary_mat(fc.pulse_vol, ST.pulse_vol_bounds;
            le_scale = pulse_vol_le * emis_scale * (1 - fc.pf), colors = pulse_colors,
                floor = pulse_vol_floor, gamma = 2.0f0))
    S.rad_vol_plot !== nothing && fc.rad_vol !== nothing &&
        (S.rad_vol_plot.material[] = vol_boundary_mat(fc.rad_vol, ST.rad_vol_bounds))
    S.screen_plot !== nothing && fc.scr_img !== nothing &&
        (S.screen_plot.material[] = screen_material(rebuild_img(fc.scr_img)))
    Makie.update!(S.electrons; arg1 = [GeometryBasics.Point3f(p...) for p in fc.epos])
    eye, lookat = fc.cam
    update_cam!(S.scene, Vec3f(eye...), Vec3f(lookat...), Vec3f(0, 0, 1))
    return S
end

SCENE_STATE = Ref{Any}(nothing)
function render_frame(fc, outpath)
    S = SCENE_STATE[]
    if S === nothing || !reuse_scene
        S = build_scene(fc)
        SCENE_STATE[] = S
    else
        update_scene!(S, fc)
    end
    t_render = @elapsed img = Makie.colorbuffer(S.scene; backend = RayMakie,
        device = DEVICE, integrator, exposure, tonemap, gamma, update = false)
    save(outpath, img)
    return t_render
end

# ── Entry: single still (EDM_RAY_T) or frame range (EDM_RAY_FRAMES) ──
frames_spec = get(ENV, "EDM_RAY_FRAMES", "")
if get(ENV, "EDM_RAY_ENTRY", "1") == "0"
    @info "EDM_RAY_ENTRY=0 — loaded as library, no render"
elseif isempty(frames_spec)
    for tT in parse.(Float64, split(get(ENV, "EDM_RAY_T", "0.0"), ","))
        payload = joinpath(cache_dir, @sprintf("still_t%+.2f.jls", tT))
        outfile = get(ENV, "EDM_RAY_OUT",
            joinpath(@__DIR__, @sprintf("ray_t%+.2f.png", tT)))
        @info "rendering still t = $tT T0 from $payload ($spp spp, hw_accel=$hw_accel)"
        fc = load_payload(payload)
        tr = render_frame(fc, outfile)
        @info "rendered in $(round(tr; digits = 1)) s — saved $outfile"
    end
else
    rng = frames_spec == "all" ? (1:length(ST.play_times)) :
        (:)(parse.(Int, split(frames_spec, ":"))...)
    outdir = get(ENV, "EDM_RAY_OUTDIR", joinpath(@__DIR__, "ray_frames"))
    mkpath(outdir)
    @info "rendering frames $(first(rng))..$(last(rng)) into $outdir"
    total = 0.0
    done = 0
    for i in rng
        outpath = joinpath(outdir, @sprintf("ray_%04d.png", i))
        isfile(outpath) && continue   # resumable
        fc = load_payload(joinpath(cache_dir, @sprintf("frame_%04d.jls", i)))
        tr = render_frame(fc, outpath)
        done += 1
        total += tr
        eta = (length(rng) - (i - first(rng) + 1)) * total / done
        @info @sprintf("frame %d/%d  t = %+.2f T0  %.1f s  (mean %.1f s, ETA %.0f min)",
            i, last(rng), fc.t / ST.T0, tr, total / done, eta / 60)
    end
    @info "done: $done frames rendered"
end
