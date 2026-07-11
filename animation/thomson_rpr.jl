# Ray-traced (RadeonProRender via RPRMakie) still frame of the Thomson-scattering
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
# Run: julia -t auto --startup=no --project=animation animation/thomson_rpr.jl
# Knobs:
#   EDM_RPR_RESOURCE=gpu|cpu  render device            (default gpu; CPU is ~7× slower, same image)
#   EDM_RPR_ITER=200          ray-tracing iterations   (more = less noise)
#   EDM_RPR_T=0.0             frame time in units of T0
#   EDM_RPR_RADIUS=0.14       electron sphere radius in units of λ (sunflower
#                             spacing is ~0.30λ — radii ≳0.15λ merge into a plate)
#   EDM_RPR_LIGHTS=laser|studio  lighting rig (default laser: the pulse itself is
#                             the key light — the electrons are lit by the laser)
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
#   EDM_RPR_OUT=path.png      output file              (default animation/rpr_frame.png)

include(joinpath(@__DIR__, "setup.jl"))

using RPRMakie
using RPRMakie: RPR
using MarchingCubes
using MarchingCubes: MC, march
import GeometryBasics
using FileIO

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

# ── Frame parameters ──
t_snap = parse(Float64, get(ENV, "EDM_RPR_T", "0.0")) * T0
iterations = parse(Int, get(ENV, "EDM_RPR_ITER", "200"))
r_electron = parse(Float64, get(ENV, "EDM_RPR_RADIUS", "0.14")) * λ
gpu = get(ENV, "EDM_RPR_RESOURCE", "gpu") == "gpu"
resource = gpu ?
    RPR.RPR_CREATION_FLAGS_ENABLE_GPU0 | RPR.RPR_CREATION_FLAGS_ENABLE_OPENCL :
    RPR.RPR_CREATION_FLAGS_ENABLE_CPU
outfile = get(ENV, "EDM_RPR_OUT", joinpath(@__DIR__, "rpr_frame.png"))

# Densified grids for the isosurface only — setup.jl's grids stay canonical for
# the GLMakie animation and the radiation precompute.
oversample = parse(Float64, get(ENV, "EDM_RPR_OVERSAMPLE", "2"))
nxr, nyr, nzr = round.(Int, oversample .* (nx, ny, nz))
xsr = LinRange(first(xs), last(xs), nxr)
ysr = LinRange(first(ys), last(ys), nyr)
zsr = LinRange(first(zs), last(zs), nzr)

@info "sampling pulse volume at t = $(t_snap / T0) T0, $(nzr)×$(nxr)×$(nyr) ($(Threads.nthreads()) threads)"
vol = sample_pulse!(Array{Float32}(undef, nzr, nxr, nyr), fe, xsr, ysr, zsr, t_snap)
epos = electron_positions(trajs, t_snap)
cmax = maximum(abs, vol)

# ── Scene ──
# max_recursion 20 (default 10): rays cross 10+ translucent ribbon layers, and
# recursion truncation visibly darkens the pulse interior at the default.
RPRMakie.activate!(; iterations, max_recursion = 20, plugin = RPR.Northstar, resource)

laser_lit = get(ENV, "EDM_RPR_LIGHTS", "laser") == "laser"
white_room = get(ENV, "EDM_RPR_BG", "dark") == "room"
ribbon_style = get(ENV, "EDM_RPR_RIBBONS", "emissive")

fig = Figure(size = (1280, 960), backgroundcolor = RGBf(0.055, 0.065, 0.09))
# The scene lives in atomic units (w₀ ≈ 6e4), and RPR point lights fall off
# physically as 1/r² — radiance must scale with the squared scene distance or
# the lights contribute nothing and only the ambient term is visible.
radiance = Float32((20w₀)^2)
lights = if white_room
    # ambient IS the environment dome in RPRMakie (an explicit EnvironmentLight
    # gets displaced by the ambient env light — upstream bug). Point lights are
    # avoided: they cast hard fan shadows through the layered translucent
    # ribbons. The pulse still "lantern-projects" its stripe pattern onto the
    # walls — that's genuine light transport from the emissive ribbons.
    [AmbientLight(RGBf(0.9, 0.9, 0.92))]
elseif laser_lit
    # the pulse's own emission is the key light — the electrons are lit by the
    # laser; keep only a weak fill + ambient so the dark side isn't dead
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

if white_room
    # gray 3-plane corner (real geometry is the only way to get a light
    # backdrop — see EDM_RPR_BG note above). The three seams run along the
    # scene axes: floor∩back = propagation (X), back∩left = vertical (Z),
    # floor∩left = Y; darker strips mark them as explicit axis lines.
    C = (-10w₀, 8w₀, -2w₀)
    t = 0.05w₀
    # satin finish: diffuse base + broad rough sheen → bright high-key walls.
    # NB reflection_weight=1 is load-bearing — lower values shift the PBR
    # diffuse/specular balance and collapse the walls to dark gray.
    satin(col) = RPR.UberMaterial(matsys;
        color = to_color(col), diffuse_weight = Vec4f(1),
        reflection_color = Vec4f(1), reflection_weight = Vec4f(1),
        reflection_roughness = Vec4f(0.45),
        reflection_mode = UInt(RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR),
        reflection_ior = Vec4f(1.5))
    for (slab, col) in (
            (Rect3f(Point3f(C[1], -16w₀, C[3] - t), Vec3f(42w₀, 24w₀, t)), RGBf(0.6, 0.6, 0.62)),   # floor
            (Rect3f(Point3f(C[1], C[2], C[3]), Vec3f(42w₀, t, 20w₀)), RGBf(0.72, 0.72, 0.74)),       # back
            (Rect3f(Point3f(C[1] - t, -16w₀, C[3]), Vec3f(t, 24w₀, 20w₀)), RGBf(0.66, 0.66, 0.68)),  # left
        )
        mesh!(ax, slab; color = col, material = satin(col))
    end
    s = 0.09w₀
    axis_gray = RGBf(0.32, 0.33, 0.36)
    for a in (
            Rect3f(Point3f(C[1], C[2] - s, C[3]), Vec3f(42w₀, s, s)),   # X: propagation
            Rect3f(Point3f(C[1], -16w₀, C[3]), Vec3f(s, 24w₀, s)),      # Y
            Rect3f(Point3f(C[1], C[2] - s, C[3]), Vec3f(s, s, 20w₀)),   # Z
        )
        mesh!(ax, a; color = axis_gray, material = RPR.DiffuseMaterial(matsys; color = to_color(axis_gray)))
    end
    # thin square grid, spacing 2w₀, registered to the origin: reads as
    # calibrated space and keeps the walls visually uniform
    g = 0.02w₀
    grid_gray = RGBf(0.5, 0.5, 0.53)
    gmat = RPR.DiffuseMaterial(matsys; color = to_color(grid_gray))
    strips = Rect3f[]
    for xg in (-10:2:32) .* w₀   # floor (∥Y) and back wall (∥Z)
        push!(strips, Rect3f(Point3f(xg - g / 2, -16w₀, C[3]), Vec3f(g, 24w₀, g)))
        push!(strips, Rect3f(Point3f(xg - g / 2, C[2] - g, C[3]), Vec3f(g, g, 20w₀)))
    end
    for yg in (-16:2:8) .* w₀    # floor (∥X) and left wall (∥Z)
        push!(strips, Rect3f(Point3f(C[1], yg - g / 2, C[3]), Vec3f(42w₀, g, g)))
        push!(strips, Rect3f(Point3f(C[1], yg - g / 2, C[3]), Vec3f(g, g, 20w₀)))
    end
    for zg in (-2:2:18) .* w₀    # back wall (∥X) and left wall (∥Y)
        push!(strips, Rect3f(Point3f(C[1], C[2] - g, zg - g / 2), Vec3f(42w₀, g, g)))
        push!(strips, Rect3f(Point3f(C[1], -16w₀, zg - g / 2), Vec3f(g, 24w₀, g)))
    end
    for st in strips
        mesh!(ax, st; color = grid_gray, material = gmat)
    end
end

# Wavefront ribbons: translucent double-sided emissive glow in the :balance
# endpoints' red/blue. Double-sided because marching-cubes normals follow the
# field gradient, so the −lobe surface faces away from the camera; translucent
# (Uber transparency input — color alpha is a no-op in RPR) so the electron
# disk reads through the pulse like the GLMakie contour's alpha did.
function ribbon_material(c)
    ribbon_style == "plastic" &&
        return RPR.Plastic(matsys; color = to_color(c), transparency = Vec4f(0.35))
    # thin-sheet tinted glass: the isosurfaces are open sheets, so
    # refraction_thin_surface is the physical model (no ray bending, no
    # volumetric absorption); tint both transmission and reflection or the
    # white reflections wash the lobe colors to neutral smoke
    tinted_glass() = RPR.Glass(matsys;
        refraction_color = Vec4f(c.r, c.g, c.b, 1),
        reflection_color = Vec4f(0.3 + 0.7c.r, 0.3 + 0.7c.g, 0.3 + 0.7c.b, 1),
        refraction_thin_surface = true,
        refraction_caustics = false,
    )
    ribbon_style == "glass" && return tinted_glass()
    ribbon_style == "glassglow" &&
        return RPR.LayerMaterial(tinted_glass(),
            RPR.EmissiveMaterial(matsys; color = Vec4f(1.5c.r, 1.5c.g, 1.5c.b, 1));
            weight = Vec4f(0.25))
    # 5× emission in laser mode: bright enough to light the electrons, low
    # enough that tone mapping doesn't bleach the stripe colors (8× goes white);
    # 3× when the room/studio provides the light
    mult = (laser_lit && !white_room) ? 5 : 3
    return RPR.UberMaterial(matsys;
        diffuse_weight = Vec4f(0),
        reflection_weight = Vec4f(0),
        emission_color = Vec4f(mult * c.r, mult * c.g, mult * c.b, 1),
        emission_weight = Vec4f(1),
        emission_mode = UInt(RPR.RPR_UBER_MATERIAL_EMISSION_MODE_DOUBLESIDED),
        transparency = Vec4f(0.55),
    )
end

for (level, c) in ((+0.5f0 * cmax, RGBf(0.85, 0.25, 0.15)),
                   (-0.5f0 * cmax, RGBf(0.2, 0.4, 0.95)))
    msh = iso_mesh(vol, zsr, xsr, ysr, level)
    if msh === nothing
        @warn "empty isosurface at level $level"
        continue
    end
    @info "isosurface at $(sign(level) > 0 ? "+" : "−")0.5·cmax: $(length(GeometryBasics.coordinates(msh))) vertices"
    mesh!(ax, msh; color = c, material = ribbon_material(c))
end

# Electrons: gold — diffuse base under a metallic specular lobe, so the disk
# reads gold under direct light instead of only mirroring the (black) sky.
gold = RPR.UberMaterial(matsys;
    color = to_color(:gold),
    diffuse_weight = Vec4f(white_room ? 0.4 : 0.6),
    reflection_color = Vec4f(1, 0.85, 0.45, 1),
    reflection_weight = Vec4f(1),
    reflection_roughness = Vec4f(0.25),
    reflection_metalness = Vec4f(1),
    reflection_mode = UInt(RPR.RPR_UBER_MATERIAL_IOR_MODE_METALNESS),
)
meshscatter!(ax, epos; markersize = r_electron, color = :gold, material = gold)

# Same hero camera as the GLMakie static frame.
update_cam!(ax.scene, Vec3f(2.5w₀, -8w₀, 3w₀), Vec3f(0), Vec3f(0, 0, 1))

@info "rendering $(iterations) iterations on $(gpu ? "GPU (Northstar/OpenCL)" : "CPU (Northstar)")"
t_render = @elapsed img = colorbuffer(screen)
@info "rendered in $(round(t_render; digits = 1)) s"
save(outfile, img)
@info "saved $outfile"
