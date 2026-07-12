# Room-brightness A/B for the HybridPro material grids: one coated-ribbon
# approach frame at several (ambient dome, exposure) combos. The room mode is
# lit by the ambient dome alone, so the dome level is what gives metals and
# glass something to reflect; exposure only lifts the tonemap midtones.
#   EDM_RPR_PLUGIN=hybridpro EDM_RPR_QUALITY=ultra EDM_RPR_BG=room EDM_RPR_ITER=500 \
#     julia -t auto --startup=no --project=animation animation/lookdev/light_ab.jl

ENV["EDM_RPR_ENTRY"] = "0"
include(joinpath(@__DIR__, "..", "thomson_rpr.jl"))

# HybridPro segfaults in rprSceneClear when a previous context is released —
# non-singleton contexts leak (~200 MB/render) but never tear down.
function RPR.Context(; plugin = RPR.Northstar,
        resource = RPR.RPR_CREATION_FLAGS_ENABLE_GPU0, singleton = true)
    return RPR.Context(plugin, resource, false)
end

outdir = joinpath(@__DIR__, "ab_light")
mkpath(outdir)

const COMBOS = [(0.9f0, "0.12"), (1.8f0, "0.12"), (1.8f0, "0.18"), (3.0f0, "0.12"),
    (3.0f0, "0.25"), (5.0f0, "0.18"), (5.0f0, "0.3")]

for (amb, expo) in COMBOS
    global ambient_room = amb
    global ribbon_style = "coated"
    global electron_style = "gold"
    ENV["EDM_RPR_EXPOSURE"] = expo
    out = joinpath(outdir, "coated_amb$(amb)_exp$(expo).png")
    isfile(out) && continue
    tr = render_frame(-5T0, out)
    @info "ambient=$amb exposure=$expo  $(round(tr; digits = 1)) s"
end
@info "A/B rendered into $outdir"
