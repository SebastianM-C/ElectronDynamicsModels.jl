# Minimal medium sanity check: a 64³ grid with an emitting ball in the center,
# inside a NullMaterial MediumInterface box, dark scene, one ambient light.
# If the whole box renders instead of just the ball, the medium path is broken.
#   julia --startup=no --project=animation/raymakie animation/raymakie/mwe_medium.jl
using Makie, RayMakie, Hikari, Lava, GPUSelect
import GeometryBasics
using FileIO

DEVICE = GPUSelect.Backend(:Lava)
integrator = Hikari.VolPath(; samples = 64, max_depth = 8, hw_accel = false,
    regularize = true, max_component_value = 10.0f0)

n = 64
σ_a = Array{Hikari.RGBSpectrum, 3}(undef, n, n, n)
Le = Array{Hikari.RGBSpectrum, 3}(undef, n, n, n)
for k in 1:n, j in 1:n, i in 1:n
    r = sqrt((i - n / 2)^2 + (j - n / 2)^2 + (k - n / 2)^2) / (n / 2)
    w = r < 0.3 ? 1.0f0 : 0.0f0
    σ_a[i, j, k] = Hikari.RGBSpectrum(w)
    Le[i, j, k] = Hikari.RGBSpectrum(w * 1.0f0, w * 0.6f0, w * 0.1f0)   # orange ball
end
σ_s = fill(Hikari.RGBSpectrum(0.0f0), n, n, n)   # explicit zero-scattering grid:
# σ_s_grid=nothing appears to be mishandled (uniform fog) — testing that
med = Hikari.RGBGridMedium(; σ_a_grid = σ_a, σ_s_grid = σ_s, Le_grid = Le,
    sigma_scale = 6.0f0, Le_scale = 2.0f0, g = 0.0f0,
    bounds = Hikari.Bounds3(GeometryBasics.Point3f(-2, -2, -2), GeometryBasics.Point3f(2, 2, 2)))

scene = Scene(; size = (480, 360), backgroundcolor = RGBf(0.02, 0.02, 0.03),
    lights = Makie.AbstractLight[Makie.AmbientLight(RGBf(0.5, 0.5, 0.5))])
cam3d!(scene)
# gray floor for reference
mesh!(scene, Rect3f(Point3f(-8, -8, -2.6), Vec3f(16, 16, 0.1));
    material = Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.5f0)))
mesh!(scene, GeometryBasics.normal_mesh(Rect3f(Point3f(-2, -2, -2), Vec3f(4, 4, 4)));
    material = Hikari.MediumInterface(Hikari.NullMaterial(); inside = med, outside = nothing))
update_cam!(scene, Vec3f(6, -8, 4), Vec3f(0, 0, 0), Vec3f(0, 0, 1))

img = Makie.colorbuffer(scene; backend = RayMakie, device = DEVICE, integrator,
    exposure = 1.0f0, tonemap = :aces, gamma = 2.2f0, update = false)
save(joinpath(@__DIR__, "mwe_medium.png"), img)
println("saved mwe_medium.png")
