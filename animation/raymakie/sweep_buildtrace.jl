# Attribute the ~100 s flash-frame floor: BLAS/upload vs trace vs
# observable-update cost, for mesh style and volume style.
#   julia --startup=no --project=animation/raymakie animation/raymakie/sweep_buildtrace.jl
ENV["EDM_RAY_ENTRY"] = "0"
ENV["EDM_RAY_RES"] = get(ENV, "EDM_RAY_RES", "640x480")
ENV["EDM_RAY_GLASS"] = "thin"
include(joinpath(@__DIR__, "thomson_ray.jl"))

fc = load_payload(joinpath(cache_dir, "still_t+3.00.jls"))
integ() = Hikari.VolPath(; samples = 32, max_depth = 8, hw_accel = false,
    regularize = true, max_component_value = 10.0f0)
render(S) = Makie.colorbuffer(S.scene; backend = RayMakie, device = DEVICE,
    integrator = integrator, exposure, tonemap, gamma, update = false)

global integrator = integ()

println("=== MESH style (stripes as iso-mesh glass) ===")
global rad_style3d = "mesh"; global pulse_style3d = "mesh"
tb = @elapsed S = build_scene(fc)
println("build_scene:            ", round(tb; digits = 1), " s")
t1 = @elapsed render(S)
println("colorbuffer #1:         ", round(t1; digits = 1), " s   (upload + BLAS + trace)")
t2 = @elapsed render(S)
println("colorbuffer #2:         ", round(t2; digits = 1), " s   (steady state)")
t3 = @elapsed render(S)
println("colorbuffer #3:         ", round(t3; digits = 1), " s")
tu = @elapsed update_scene!(S, fc)
t4 = @elapsed render(S)
println("update_scene!:          ", round(tu; digits = 1), " s")
println("colorbuffer after upd:  ", round(t4; digits = 1), " s   (mesh-swap rebuild cost)")

println("=== VOLUME style (stripes+pulse as emissive media) ===")
global rad_style3d = "volume"; global pulse_style3d = "volume"
tbv = @elapsed SV = build_scene(fc)
println("build_scene:            ", round(tbv; digits = 1), " s")
v1 = @elapsed render(SV)
println("colorbuffer #1:         ", round(v1; digits = 1), " s")
v2 = @elapsed render(SV)
println("colorbuffer #2:         ", round(v2; digits = 1), " s")
tuv = @elapsed update_scene!(SV, fc)
v3 = @elapsed render(SV)
println("update_scene!:          ", round(tuv; digits = 1), " s")
println("colorbuffer after upd:  ", round(v3; digits = 1), " s   (grid re-upload cost)")
save(joinpath(@__DIR__, "sweep_volume_t+3.png"), render(SV))
println("volume still saved: sweep_volume_t+3.png")
