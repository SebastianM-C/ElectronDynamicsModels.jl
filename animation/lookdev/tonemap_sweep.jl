# Offline exposure/tonemap sweep over a raw HDR framebuffer dumped by
# thomson_rpr.jl (EDM_RPR_DUMP_RAW=path, HybridPro path only). Regenerates the
# tonemapped PNG at several exposures without re-rendering, so grading the
# HybridPro look against the Northstar reference costs seconds per iteration.
#
# Usage: julia --project=animation animation/lookdev/tonemap_sweep.jl raw.jls [out_prefix]
using Serialization
using FileIO
using Colors: RGB

d = deserialize(ARGS[1])
prefix = length(ARGS) >= 2 ? ARGS[2] : splitext(ARGS[1])[1]

# same operator as thomson_rpr.jl capture(): luminance Reinhard + gamma
function tonemap(raw, fb_size, ex)
    img = map(raw) do c
        a = max(c.alpha, 1.0f-6)
        r, g, b = ex * c.r / a, ex * c.g / a, ex * c.b / a
        L = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
        s = L > 0 ? (L / (1 + L)) / L : 0.0f0
        RGB{Float32}(clamp(s * r, 0, 1)^0.4545f0, clamp(s * g, 0, 1)^0.4545f0,
            clamp(s * b, 0, 1)^0.4545f0)
    end
    return permutedims(reshape(img, fb_size))
end

exposures = let s = get(ENV, "EDM_SWEEP_EXPOSURES", "")
    isempty(s) ? Float32[0.05, 0.08, 0.12, 0.2, 0.3] : parse.(Float32, split(s, ","))
end
for ex in exposures
    out = "$(prefix)_ex$(ex).png"
    save(out, tonemap(d.raw, d.fb_size, ex))
    println("wrote $out")
end
