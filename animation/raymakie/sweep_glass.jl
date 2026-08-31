# Timing/quality sweep over the ribbon/stripe interface model and path depth —
# diagnosing the 2327 s flash frame (t=+3, both stripe stacks in scene).
# One process, one kernel compile; scene rebuilt per config.
#   julia --startup=no --project=animation/raymakie animation/raymakie/sweep_glass.jl
ENV["EDM_RAY_ENTRY"] = "0"
ENV["EDM_RAY_RES"] = get(ENV, "EDM_RAY_RES", "640x480")
include(joinpath(@__DIR__, "thomson_ray.jl"))

fc = load_payload(joinpath(cache_dir, "still_t+3.00.jls"))
results = Tuple{String, Int, Float64}[]
for (kind, depth) in (("dielectric", 16), ("dielectric", 8),
        ("thin", 16), ("thin", 8), ("interface", 8))
    global glass_kind = kind
    global integrator = Hikari.VolPath(; samples = 32, max_depth = depth,
        hw_accel = false, regularize = true, max_component_value = 10.0f0)
    SCENE_STATE[] = nothing   # force rebuild so the new materials apply
    out = joinpath(@__DIR__, "sweep_$(kind)_d$(depth).png")
    tr = render_frame(fc, out)
    push!(results, (kind, depth, tr))
    @info "config $kind depth=$depth → $(round(tr; digits = 1)) s"
end
println("\n=== sweep results (640x480, 32 spp, t=+3) ===")
for (k, d, tr) in results
    println(rpad(k, 12), " depth=", rpad(d, 3), round(tr; digits = 1), " s")
end
