# Faint-laser / translucent-radiation exploration: glassglow ribbons with the
# emission dialed down (EDM_RPR_EMIS via emis_scale) + pure-glass radiation
# stripes (no glow). The glass wavefronts need environment light to read, so a
# brighter-dome variant is included.
#   EDM_RPR_PLUGIN=hybridpro EDM_RPR_QUALITY=ultra EDM_RPR_BG=room \
#     EDM_RPR_ITER=500 EDM_RPR_EXPOSURE=0.12 \
#     julia -t auto --startup=no --project=animation animation/lookdev/faint_laser_grid.jl

ENV["EDM_RPR_ENTRY"] = "0"
include(joinpath(@__DIR__, "..", "thomson_rpr.jl"))

# HybridPro segfaults in rprSceneClear when a previous context is released —
# non-singleton contexts leak (~200 MB/render) but never tear down.
function RPR.Context(; plugin = RPR.Northstar,
        resource = RPR.RPR_CREATION_FLAGS_ENABLE_GPU0, singleton = true)
    return RPR.Context(plugin, resource, false)
end

outdir = joinpath(@__DIR__, "faint_laser")
mkpath(outdir)

#             tag                     emis   ambient  exposure  mat        tint
const VARIANTS = [
    ("emis0.4_amb0.9", 0.4f0, 0.9f0, "0.12", "glass", 1.0f0),
    ("emis0.2_amb0.9", 0.2f0, 0.9f0, "0.12", "glass", 1.0f0),
    ("emis0.4_amb1.8", 0.4f0, 1.8f0, "0.12", "glass", 1.0f0),
    ("pale_bright", 0.7f0, 3.0f0, "0.25", "glass", 0.5f0),
    ("pale_bright_whisper", 0.7f0, 3.0f0, "0.25", "glassglow", 0.5f0),
    ("pale_brighter", 0.7f0, 5.0f0, "0.3", "glass", 0.5f0),
]

for (tag, t) in (("flash", 4T0), ("travel", 10T0))
    for (name, emis, amb, expo, mat, tint) in VARIANTS
        global rad_style = "striped"
        global rad_mat = mat
        global ribbon_style = "glassglow"
        global electron_style = "gold"
        global emis_scale = emis
        global ambient_room = amb
        global rad_tint = tint
        ENV["EDM_RPR_EXPOSURE"] = expo
        out = joinpath(outdir, "$(tag)_$(name).png")
        isfile(out) && continue
        tr = render_frame(t, out)
        @info "[$tag] $name  $(round(tr; digits = 1)) s"
    end
end
@info "faint-laser grid rendered into $outdir"
