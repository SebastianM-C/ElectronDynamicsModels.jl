# Volume-mode look sweep: (floor, sigma, le_scale) for the emissive
# RGBGridMedium mapping, flash frame t=+3, radiation + pulse as media.
#   julia --startup=no --project=animation/raymakie animation/raymakie/sweep_volume.jl
ENV["EDM_RAY_ENTRY"] = "0"
ENV["EDM_RAY_RES"] = get(ENV, "EDM_RAY_RES", "640x480")
ENV["EDM_RAY_STYLE"] = "volume"
ENV["EDM_RAY_PULSE_STYLE"] = "volume"
include(joinpath(@__DIR__, "thomson_ray.jl"))

fc = load_payload(joinpath(cache_dir, "still_t+3.00.jls"))
global integrator = Hikari.VolPath(; samples = 32, max_depth = 8,
    hw_accel = false, regularize = true, max_component_value = 10.0f0)
for le in (0.5f0, 1.0f0, 2.0f0, 4.0f0), sig in (2.0f0, 6.0f0, 15.0f0)
    global vol_floor = 0.2f0
    global vol_sigma = sig
    global vol_le = le
    global vol_gamma = 1.5f0
    SCENE_STATE[] = nothing
    out = joinpath(@__DIR__, "sweep_vol_le$(le)_s$(sig).png")
    tr = render_frame(fc, out)
    @info "le=$le sigma=$sig → $(round(tr; digits = 1)) s"
end
println("volume sweep done")
