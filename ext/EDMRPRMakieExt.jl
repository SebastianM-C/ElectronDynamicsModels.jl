# RPR rendering plumbing (see src/rpr_api.jl for docstrings). Everything here
# was established empirically on Northstar (W7900/OpenCL) + HybridPro (Vulkan
# HW-RT) during the Thomson animation work — the comments record the findings.
module EDMRPRMakieExt

using ElectronDynamicsModels
import ElectronDynamicsModels as EDM
using RPRMakie
using RPRMakie: RPR, Makie
using Serialization: serialize

function EDM.rpr_capture(screen::RPRMakie.Screen; hybrid::Bool,
        exposure::Real = 0.2, sat::Real = 1.0, dump_raw::AbstractString = "")
    hybrid || return Makie.colorbuffer(screen)
    Makie.colorbuffer(screen)   # drives the render loop; its resolve is black on Hybrid
    raw = RPR.get_data(screen.framebuffer1)
    isempty(dump_raw) || serialize(dump_raw, (; raw, fb_size = screen.fb_size))
    ex, st = Float32(exposure), Float32(sat)
    img = map(raw) do c
        a = max(c.alpha, 1.0f-6)   # alpha = per-pixel sample count
        r, g, b = ex * c.r / a, ex * c.g / a, ex * c.b / a
        L = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
        s = L > 0 ? (L / (1 + L)) / L : 0.0f0
        rr = clamp(s * r, 0, 1)^0.4545f0
        gg = clamp(s * g, 0, 1)^0.4545f0
        bb = clamp(s * b, 0, 1)^0.4545f0
        if st != 1
            Lg = 0.2126f0 * rr + 0.7152f0 * gg + 0.0722f0 * bb
            rr = clamp(Lg + st * (rr - Lg), 0, 1)
            gg = clamp(Lg + st * (gg - Lg), 0, 1)
            bb = clamp(Lg + st * (bb - Lg), 0, 1)
        end
        Makie.RGBf(rr, gg, bb)
    end
    return permutedims(reshape(img, screen.fb_size))
end

function EDM.rpr_tune!(screen::RPRMakie.Screen; quality = nothing,
        denoiser = nothing, ray_depth = nothing)
    ctx = screen.context.pointer
    if quality !== nothing
        qv = UInt32(findfirst(==(quality), ["low", "medium", "high", "ultra"]) - 1)
        RPR.rprContextSetParameterByKey1u(ctx,
            reinterpret(RPR.rpr_context_info, Int32(0x1001)), qv)
    end
    if denoiser !== nothing
        dv = UInt32(findfirst(==(denoiser), ["none", "svgf", "asvgf", "ml"]) - 1)
        RPR.rprContextSetParameterByKey1u(ctx,
            reinterpret(RPR.rpr_context_info, Int32(0x102D)), dv)
    end
    if ray_depth !== nothing
        n = UInt(ray_depth)
        RPR.rprContextSetParameterByKey1u(ctx, RPR.RPR_CONTEXT_MAX_RECURSION, n)
        RPR.rprContextSetParameterByKey1u(ctx, RPR.RPR_CONTEXT_MAX_DEPTH_REFRACTION, n)
        RPR.rprContextSetParameterByKey1u(ctx,
            RPR.RPR_CONTEXT_MAX_DEPTH_GLOSSY_REFRACTION, n)
        RPR.rprContextSetParameterByKey1u(ctx, RPR.RPR_CONTEXT_MAX_DEPTH_GLOSSY,
            min(n, UInt(8)))
    end
    return screen
end

# Non-singleton contexts, gated behind a flag so plain Northstar sessions keep
# RPRMakie's stock (leak-free) singleton behavior. Type piracy on RPR.Context's
# keyword constructor — the alternative is a segfault in HybridPro's
# rprSceneClear whenever the singleton frees the previous context.
const MULTIFRAME = Ref(false)
EDM.rpr_enable_multiframe!() = (MULTIFRAME[] = true; nothing)

function RPR.Context(; plugin = RPR.Northstar,
        resource = RPR.RPR_CREATION_FLAGS_ENABLE_GPU0, singleton = true)
    return RPR.Context(plugin, resource, MULTIFRAME[] ? false : singleton)
end

end
