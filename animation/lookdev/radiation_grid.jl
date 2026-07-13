# Radiation-representation comparison: the four styles the v2 cube (signed
# Ex_far) unlocks, at two story moments — flash (t = 4 T0) and the wavefront
# in flight toward the detector (t = 10 T0). Renders in the animation's room +
# laser-lit look so the choice is judged in context.
#   EDM_RPR_PLUGIN=hybridpro EDM_RPR_QUALITY=ultra EDM_RPR_BG=room \
#     EDM_RPR_ITER=500 EDM_RPR_EXPOSURE=0.12 \
#     julia -t auto --startup=no --project=animation animation/lookdev/radiation_grid.jl

ENV["EDM_RPR_ENTRY"] = "0"
include(joinpath(@__DIR__, "..", "thomson_rpr.jl"))

outdir = joinpath(@__DIR__, "radiation_grid")
mkpath(outdir)

#            tag              rad_style   rad_mat
const VARIANTS = [("shells", "shells", "emissive"),
    ("striped_emissive", "striped", "emissive"),
    ("striped_glass", "striped", "glass")]

for (tag, t) in (("flash", 4T0), ("travel", 10T0))
    for (name, style, mat) in VARIANTS
        global rad_style = style
        global rad_mat = mat
        global ribbon_style = "glassglow"
        global electron_style = "gold"
        out = joinpath(outdir, "$(tag)_$(name).png")
        isfile(out) && continue
        tr = render_frame(t, out)
        @info "[$tag] $name  $(round(tr; digits = 1)) s"
    end
end
@info "radiation grid rendered into $outdir"
