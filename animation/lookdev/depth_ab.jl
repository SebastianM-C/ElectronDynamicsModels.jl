# Ray-depth A/B: the pale_bright faint-laser config at the HybridPro preset
# depth vs EDM_RPR_RAY_DEPTH=16. Nested glass stripes exhaust the preset's
# refraction budget — depth-starved rays terminate black inside the stack.
#   EDM_RPR_PLUGIN=hybridpro EDM_RPR_QUALITY=ultra EDM_RPR_BG=room EDM_RPR_ITER=500 \
#     julia -t auto --startup=no --project=animation animation/lookdev/depth_ab.jl

ENV["EDM_RPR_ENTRY"] = "0"
include(joinpath(@__DIR__, "..", "thomson_rpr.jl"))

function RPR.Context(; plugin = RPR.Northstar,
        resource = RPR.RPR_CREATION_FLAGS_ENABLE_GPU0, singleton = true)
    return RPR.Context(plugin, resource, false)
end

outdir = joinpath(@__DIR__, "faint_laser")
mkpath(outdir)

for depth in ("", "16")
    global rad_style = "striped"
    global rad_mat = "glass"
    global ribbon_style = "glassglow"
    global electron_style = "gold"
    global emis_scale = 0.7f0
    global ambient_room = 3.0f0
    global rad_tint = 0.5f0
    ENV["EDM_RPR_EXPOSURE"] = "0.25"
    ENV["EDM_RPR_RAY_DEPTH"] = depth
    tag = isempty(depth) ? "preset" : "depth$depth"
    out = joinpath(outdir, "travel_pale_bright_$(tag).png")
    isfile(out) && continue
    tr = render_frame(10T0, out)
    @info "ray depth=$(isempty(depth) ? "preset" : depth)  $(round(tr; digits = 1)) s"
end
@info "depth A/B rendered into $outdir"
