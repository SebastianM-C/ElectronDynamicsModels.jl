# HybridPro material exploration grids: ribbon (laser) and electron variants,
# rendered at two story moments — approach (t = -5 T0, pre-radiation) and flash
# (t = +4 T0, radiation present). Loads thomson_rpr.jl once as a library
# (EDM_RPR_ENTRY=0) so each variant costs one ~10-15 s render, not a 5-minute
# process startup.
#
# Usage (env sets the base look; the script varies one knob at a time):
#   EDM_RPR_PLUGIN=hybridpro EDM_RPR_QUALITY=ultra EDM_RPR_BG=room \
#     EDM_RPR_ITER=500 EDM_RPR_EXPOSURE=0.12 \
#     julia -t auto --startup=no --project=animation animation/lookdev/hybrid_material_grid.jl

ENV["EDM_RPR_ENTRY"] = "0"
include(joinpath(@__DIR__, "..", "thomson_rpr.jl"))

outdir = joinpath(@__DIR__, "material_grid")
mkpath(outdir)

const RIBBON_VARIANTS = ["glassglow", "glass", "coated", "emissive"]
const ELECTRON_VARIANTS = ["gold", "mirror", "brushed", "copper"]

for (tag, t) in (("approach", -5T0), ("flash", 4T0))
    for rv in RIBBON_VARIANTS
        global ribbon_style = rv
        global electron_style = "gold"
        out = joinpath(outdir, "$(tag)_ribbon_$(rv).png")
        isfile(out) && continue
        tr = render_frame(t, out)
        @info "[$tag] ribbon=$rv  $(round(tr; digits = 1)) s"
    end
    for ev in ELECTRON_VARIANTS
        global ribbon_style = "glassglow"
        global electron_style = ev
        out = joinpath(outdir, "$(tag)_electron_$(ev).png")
        isfile(out) && continue
        tr = render_frame(t, out)
        @info "[$tag] electrons=$ev  $(round(tr; digits = 1)) s"
    end
end
@info "grids rendered into $outdir"
