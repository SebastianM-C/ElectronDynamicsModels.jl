# Physics side of the RayMakie experiment: everything thomson_rpr.jl computes
# per frame (pulse isosurfaces, radiation-cube slices/stripes, screen images,
# electron positions, camera path) evaluated here and dumped as plain-array
# payloads for animation/raymakie/thomson_ray.jl to render.
#
# Why a separate process: Hikari pins StructArrays 0.6 while SciMLBase 3
# (RecursiveArrayTools 4) needs StructArrays 0.7 — the SciML physics stack and
# the ray-tracing stack cannot resolve into one environment. Payloads are
# plain tuples/arrays (no GeometryBasics/EDM types) so the two envs never need
# matching package versions. Data/transfer blocks are copied verbatim from
# animation/thomson_rpr.jl to stay frame-comparable with the RPR renders.
#
# Run:  julia -t 8 --startup=no --project=animation \
#         animation/raymakie/precompute_frames.jl
# Knobs:
#   EDM_RAY_T=0.0             single still at t (T0 units) → still_<t>.jls
#   EDM_RAY_FRAMES=a:b|all    play_times frames a..b → frame_%04d.jls (resumable)
#   EDM_RAY_CACHE=dir         payload dir (default animation/raymakie/frame_cache)
#   EDM_RAY_CONTENT=mesh|volume|both   which radiation/pulse representations to
#                             precompute (default both for stills, mesh for ranges)
#   EDM_RAY_OVERSAMPLE=2 EDM_RAY_RAD_SLEVEL=0.3 EDM_RAY_RAD_UPSAMPLE=4
#   EDM_RAY_RAD_TINT=0.5 EDM_RAY_RAD_COLORS  EDM_RAY_SCREEN_STYLE=striped
#   EDM_RAY_SCREEN_FLOOR=0.12 EDM_RAY_SCREEN_DEVELOP=live EDM_RAY_PULSE_FADE=14,4
#   EDM_RADIATION_CUBE / EDM_SCREEN_TIMESERIES   data overrides (as everywhere)

include(joinpath(@__DIR__, "..", "setup.jl"))

using MarchingCubes   # triggers EDMIsoMeshExt (with GeometryBasics)
import GeometryBasics
using Serialization
using Printf
import DataInterpolations
using ElectronDynamicsModels: iso_mesh

# RGBf without a Makie dep (Colors is in the animation env)
import Colors
const RGBf = Colors.RGB{Float32}

# ── Shared look/content knobs (defaults = production render_production.sh) ──
rad_slevel = parse(Float32, get(ENV, "EDM_RAY_RAD_SLEVEL", "0.3"))
rad_upsample = parse(Int, get(ENV, "EDM_RAY_RAD_UPSAMPLE", "4"))
rad_tint = parse(Float32, get(ENV, "EDM_RAY_RAD_TINT", "0.5"))
rad_colors = let s = get(ENV, "EDM_RAY_RAD_COLORS", "1.0,0.75,0.2;0.55,0.3,0.9")
    a, b = split(s, ";")
    (RGBf(parse.(Float32, split(a, ","))...), RGBf(parse.(Float32, split(b, ","))...))
end
screen_style = get(ENV, "EDM_RAY_SCREEN_STYLE", "striped")
r_electron = parse(Float64, get(ENV, "EDM_RAY_RADIUS", "0.14")) * λ

# ── Pulse scalar + sampler (verbatim from thomson_rpr.jl) ──
pulse_scalar(E, B) = E[1]

function sample_pulse!(buf, fe, xs, ys, zs, t)
    Threads.@threads for j in eachindex(ys)
        y = ys[j]
        for (i, x) in enumerate(xs), (k, z) in enumerate(zs)
            E, B = fe([t, x, y, z])
            buf[k, i, j] = Float32(pulse_scalar(E, B))
        end
    end
    return buf
end

# Densified grids for the isosurface; pulse box reaches the detector (+16λ)
oversample = parse(Float64, get(ENV, "EDM_RAY_OVERSAMPLE", "2"))
z_pulse_hi = 16λ
nxr, nyr = round.(Int, oversample .* (nx, ny))
nzr = round(Int, oversample * nz * (z_pulse_hi - first(zs)) / (last(zs) - first(zs)))
xsr = LinRange(first(xs), last(xs), nxr)
ysr = LinRange(first(ys), last(ys), nyr)
zsr = LinRange(first(zs), z_pulse_hi, nzr)
vol_buf = Array{Float32}(undef, nzr, nxr, nyr)
# native-resolution buffer for the volume-mode pulse
nzn = round(Int, nz * (z_pulse_hi - first(zs)) / (last(zs) - first(zs)))
zsn = LinRange(first(zs), z_pulse_hi, nzn)
vol_buf_n = Array{Float32}(undef, nzn, nx, ny)

# ── Scene units ──
const SCN = Float64(w₀)
scn(x) = Float32(x / SCN)
const sw = 1.0f0

# ── Stage C data: radiation cube + screen (verbatim from thomson_rpr.jl) ──
default_cube = let p = joinpath(@__DIR__, "..", "radiation_cube.jls")
    isfile(p) ? p :
        expanduser("~/.julia/dev/ElectronDynamicsModels/animation/radiation_cube.jls")
end
radiation_cube = get(ENV, "EDM_RADIATION_CUBE", default_cube)
RAD = nothing
rad_win = nothing
rad_floor = 0.0f0
const rad_decades = 3.0f0
rad_s_ceil = 0.0f0
if isfile(radiation_cube) && get(ENV, "EDM_RAY_RADIATION", "1") == "1"
    @info "loading radiation cube $radiation_cube"
    RAD = deserialize(radiation_cube)
    _rs = filter(>(0.0f0), vec(@view RAD.rad[1:3:end, 1:5:end, 1:5:end, 1:5:end]))
    rad_ceil = partialsort!(_rs, max(1, round(Int, 0.005 * length(_rs))); rev = true)
    rad_floor = rad_ceil / exp10(rad_decades)
    edgewindow(n, frac = 0.1f0; lo = true, hi = true) = begin
        m = max(2, round(Int, frac * n))
        w = ones(Float32, n)
        for i in 1:m
            s = Float32(i - 1) / m
            lo && (w[i] = s * s * (3 - 2s))
            hi && (w[n + 1 - i] = s * s * (3 - 2s))
        end
        w
    end
    rad_win = reshape(edgewindow(size(RAD.rad, 2); hi = false), :, 1, 1) .*
        reshape(edgewindow(size(RAD.rad, 3)), 1, :, 1) .*
        reshape(edgewindow(size(RAD.rad, 4)), 1, 1, :)
    if haskey(RAD, :rad_s)
        _ss = filter(!iszero, vec(@view RAD.rad_s[1:3:end, 1:5:end, 1:5:end, 1:5:end]))
        map!(abs, _ss, _ss)
        rad_s_ceil = partialsort!(_ss, max(1, round(Int, 0.005 * length(_ss))); rev = true)
    end
end
function upsample_dim1(v, x, xu)
    out = Array{Float32}(undef, length(xu), size(v, 2), size(v, 3))
    Threads.@threads for k in axes(v, 3)
        for j in axes(v, 2)
            sp = DataInterpolations.CubicSpline(collect(@view v[:, j, k]), x)
            @inbounds for (i, xi) in enumerate(xu)
                out[i, j, k] = Float32(sp(xi))
            end
        end
    end
    return out
end
rad_s_grid() = LinRange(first(RAD.slice_zs), last(RAD.slice_zs),
    rad_upsample * (length(RAD.slice_zs) - 1) + 1)
rad_s_slice(t; upsample = true) = begin
    f = clamp(searchsortedlast(RAD.frame_times, t), 1, length(RAD.frame_times))
    s = upsample ?
        upsample_dim1(@view(RAD.rad_s[f, :, :, :]), RAD.slice_zs, rad_s_grid()) :
        Array{Float32}(@view RAD.rad_s[f, :, :, :])
    w1 = edgewindow(size(s, 1); hi = false)
    w2 = edgewindow(size(s, 2))
    w3 = edgewindow(size(s, 3))
    @. s = clamp(s / rad_s_ceil, -1.0f0, 1.0f0) *
        $(reshape(w1, :, 1, 1)) * $(reshape(w2, 1, :, 1)) * $(reshape(w3, 1, 1, :))
    s
end

default_scr = let p = joinpath(@__DIR__, "..", "screen_timeseries.jls")
    isfile(p) ? p :
        expanduser("~/.julia/dev/ElectronDynamicsModels/animation/screen_timeseries.jls")
end
screen_file = get(ENV, "EDM_SCREEN_TIMESERIES", default_scr)
show_screen = get(ENV, "EDM_RAY_SCREEN", "1") == "1"
SCR = if show_screen && isfile(screen_file)
    @info "loading screen time series $screen_file"
    deserialize(screen_file)
elseif show_screen && RAD !== nothing
    @info "no dedicated screen data — using the radiation cube's +X edge slice"
    (; scr = RAD.rad[:, end, :, :], txs = RAD.txs, tys = RAD.tys,
        z_screen = last(RAD.slice_zs), frame_times = RAD.frame_times)
else
    nothing
end
scr_floor = 0.0f0
if SCR !== nothing
    _ss = filter(>(0.0f0), vec(@view SCR.scr[1:2:end, 1:3:end, 1:3:end]))
    scr_ceil = partialsort!(_ss, max(1, round(Int, 0.002 * length(_ss))); rev = true)
    global scr_floor = scr_ceil / exp10(3.0f0)
end
scr_slice(t) = begin
    f = clamp(searchsortedlast(SCR.frame_times, t), 1, length(SCR.frame_times))
    @. clamp(log10(max($(@view SCR.scr[f, :, :]), scr_floor) / scr_floor) / 3.0f0, 0.0f0, 1.0f0)
end
scr_s_ceil = 0.0f0
scr_flu = nothing
scr_flu_ceil = 0.0f0
if RAD !== nothing && haskey(RAD, :rad_s)
    _sv = filter(!iszero, vec(@view RAD.rad_s[1:2:end, end, 1:3:end, 1:3:end]))
    map!(abs, _sv, _sv)
    isempty(_sv) ||
        (scr_s_ceil = partialsort!(_sv, max(1, round(Int, 0.002 * length(_sv))); rev = true))
    scr_flu = cumsum(abs2.(@view RAD.rad_s[:, end, :, :]); dims = 1)
    _fv = filter(>(0.0f0), vec(@view scr_flu[end, 1:2:end, 1:2:end]))
    isempty(_fv) ||
        (scr_flu_ceil = partialsort!(_fv, max(1, round(Int, 0.005 * length(_fv))); rev = true))
end
scr_flu_frac = scr_flu === nothing ? nothing :
    (fs = [sum(@view scr_flu[f, :, :]) for f in axes(scr_flu, 1)]; fs ./ max(fs[end], eps()))
scr_develop = let s = get(ENV, "EDM_RAY_SCREEN_DEVELOP", "live")
    isempty(s) || s == "off" ? nothing :
    s == "live" ? :live : Tuple(parse.(Float64, split(s, ",")))
end
dev_w(t) = scr_develop === nothing ? 0.0f0 :
    scr_develop === :live ?
    (scr_flu_frac === nothing ? 0.0f0 :
     Float32(scr_flu_frac[clamp(searchsortedlast(RAD.frame_times, t), 1,
        length(RAD.frame_times))])) :
    Float32(clamp((t / T0 - scr_develop[1]) / scr_develop[2], 0, 1)^2 *
            (3 - 2 * clamp((t / T0 - scr_develop[1]) / scr_develop[2], 0, 1)))
scr_developed_img(f) = begin
    flo = scr_flu_ceil / exp10(2.5f0)
    pos, neg = rad_colors
    map(@view scr_flu[f, :, :]) do v
        u = clamp(log10(max(v, flo) / flo) / 2.5f0, 0.0f0, 1.0f0)
        if u < 0.5f0
            k = 2u
            RGBf(k * neg.r, k * neg.g, k * neg.b)
        elseif u < 0.85f0
            k = (u - 0.5f0) / 0.35f0
            RGBf(neg.r + k * (pos.r - neg.r), neg.g + k * (pos.g - neg.g),
                neg.b + k * (pos.b - neg.b))
        else
            k = (u - 0.85f0) / 0.15f0
            RGBf(pos.r + k * (1 - pos.r), pos.g + k * (1 - pos.g),
                pos.b + k * (1 - pos.b))
        end
    end
end
pale(c, tint) = RGBf(1 - tint * (1 - c.r), 1 - tint * (1 - c.g), 1 - tint * (1 - c.b))
scr_striped_img(t) = begin
    f = clamp(searchsortedlast(RAD.frame_times, t), 1, length(RAD.frame_times))
    pos = pale(rad_colors[1], rad_tint)
    neg = pale(rad_colors[2], rad_tint)
    flo = parse(Float32, get(ENV, "EDM_RAY_SCREEN_FLOOR", "0.12"))
    inst = map(@view RAD.rad_s[f, end, :, :]) do s
        w = clamp((abs(s) / scr_s_ceil - flo) / (1 - flo), 0.0f0, 1.0f0)^0.45f0
        base = s >= 0 ? pos : neg
        hot = max(0.0f0, 2.5f0 * (w - 0.75f0))
        RGBf(clamp(w * base.r + hot, 0, 1), clamp(w * base.g + hot, 0, 1),
            clamp(w * base.b + hot, 0, 1))
    end
    (scr_flu !== nothing && scr_develop !== nothing) || return inst
    if scr_develop === :live
        dev = scr_developed_img(f)
        return map(inst, dev) do a, b
            RGBf(1 - (1 - a.r) * (1 - b.r), 1 - (1 - a.g) * (1 - b.g),
                1 - (1 - a.b) * (1 - b.b))
        end
    end
    wd = dev_w(t)
    wd > 0 || return inst
    dev = scr_developed_img(f)
    map(inst, dev) do a, b
        RGBf(a.r + wd * (b.r - a.r), a.g + wd * (b.g - a.g), a.b + wd * (b.b - a.b))
    end
end
span_scr = SCR === nothing ? 0.0f0 : scn(SCR.z_screen - first(zs))

# ── Camera path (verbatim) ──
smoothstep(x) = x * x * (3 - 2x)
function camera_for(t)
    hero = ((2.5f0 * sw, -8.0f0 * sw, 3.0f0 * sw), (0.0f0, 0.0f0, 0.0f0))
    SCR === nothing && return hero
    cx = scn((first(zs) + SCR.z_screen) / 2)
    wide = ((cx + 0.1f0 * span_scr, -1.2f0 * span_scr, 0.32f0 * span_scr), (cx, 0.0f0, 0.0f0))
    s = Float32(smoothstep(clamp((t - 5T0) / 9T0, 0, 1)))
    return ((1 - s) .* hero[1] .+ s .* wide[1], (1 - s) .* hero[2] .+ s .* wide[2])
end

pulse_fade(t) = let s = get(ENV, "EDM_RAY_PULSE_FADE", "14,4")
    if isempty(s) || s == "off"
        0.0f0
    else
        f0, fspan = parse.(Float64, split(s, ","))
        Float32(smoothstep(clamp((t / T0 - f0) / fspan, 0, 1)))
    end
end

# ── Plain-array payloads (env-boundary safe: tuples + Arrays only) ──
plain_mesh(m) = m === nothing ? nothing :
    (; pts = [Tuple(Float32.(p)) for p in GeometryBasics.coordinates(m)],
        nrm = [Tuple(Float32.(n)) for n in GeometryBasics.normals(m)],
        fcs = [Int32.(Tuple(f)) for f in GeometryBasics.faces(m)])
plain_img(img) = [(c.r, c.g, c.b) for c in img]

content_kind = get(ENV, "EDM_RAY_CONTENT", isempty(get(ENV, "EDM_RAY_FRAMES", "")) ? "both" : "mesh")
do_mesh = content_kind in ("mesh", "both")
do_vol = content_kind in ("volume", "both")

function frame_payload(t)
    pf = pulse_fade(t)
    pulse_meshes = if do_mesh && pf < 0.995f0
        sample_pulse!(vol_buf, fe, xsr, ysr, zsr, t)
        cmax = maximum(abs, vol_buf)
        (plain_mesh(iso_mesh(vol_buf, scn.(zsr), scn.(xsr), scn.(ysr), +0.5f0 * cmax)),
            plain_mesh(iso_mesh(vol_buf, scn.(zsr), scn.(xsr), scn.(ysr), -0.5f0 * cmax)))
    else
        (nothing, nothing)
    end
    pulse_vol = if do_vol && pf < 0.995f0
        sample_pulse!(vol_buf_n, fe, xs, ys, zsn, t)
        m = maximum(abs, vol_buf_n)
        m > 0 ? vol_buf_n ./ m : copy(vol_buf_n)
    else
        nothing
    end
    rad_meshes = if RAD !== nothing && do_mesh && haskey(RAD, :rad_s)
        s = rad_s_slice(t)
        zu = scn.(rad_s_grid())
        (plain_mesh(iso_mesh(s, zu, scn.(RAD.txs), scn.(RAD.tys), +rad_slevel)),
            plain_mesh(iso_mesh(s, zu, scn.(RAD.txs), scn.(RAD.tys), -rad_slevel)))
    else
        (nothing, nothing)
    end
    rad_vol = if RAD !== nothing && do_vol && haskey(RAD, :rad_s)
        rad_s_slice(t; upsample = false)
    else
        nothing
    end
    scr_img = if SCR !== nothing
        if screen_style == "striped" && RAD !== nothing && haskey(RAD, :rad_s) && scr_s_ceil > 0
            plain_img(scr_striped_img(t))
        else
            su = scr_slice(t)
            [(v, v, v) for v in su]   # gray fallback; production uses striped
        end
    else
        nothing
    end
    epos = [Tuple(Float32.(scn.(p))) for p in electron_positions(trajs, t)]
    return (; t, pf, pulse_meshes, pulse_vol, rad_meshes, rad_vol, scr_img, epos,
        cam = camera_for(t))
end

static_payload() = (;
    T0, sw, span_scr,
    play_times = collect(Float64, play_times),
    r_electron_scn = scn(r_electron),
    screen = SCR === nothing ? nothing : (;
        x = scn(SCR.z_screen),
        ylo = scn(first(SCR.txs)), yhi = scn(last(SCR.txs)),
        zlo = scn(first(SCR.tys)), zhi = scn(last(SCR.tys))),
    pulse_vol_bounds = ((scn(first(zsn)), scn(first(xs)), scn(first(ys))),
        (scn(last(zsn)), scn(last(xs)), scn(last(ys)))),
    rad_vol_bounds = RAD === nothing ? nothing :
        ((scn(first(RAD.slice_zs)), scn(first(RAD.txs)), scn(first(RAD.tys))),
        (scn(last(RAD.slice_zs)), scn(last(RAD.txs)), scn(last(RAD.tys)))),
)

# ── Entry ──
cache_dir = get(ENV, "EDM_RAY_CACHE", joinpath(@__DIR__, "frame_cache"))
mkpath(cache_dir)
serialize(joinpath(cache_dir, "static.jls"), static_payload())
@info "static payload written to $cache_dir"

frames_spec = get(ENV, "EDM_RAY_FRAMES", "")
if isempty(frames_spec)
    ts = parse.(Float64, split(get(ENV, "EDM_RAY_T", "0.0"), ","))
    for tT in ts
        out = joinpath(cache_dir, @sprintf("still_t%+.2f.jls", tT))
        el = @elapsed serialize(out, frame_payload(tT * T0))
        @info @sprintf("still t = %+.2f T0  %.1f s  → %s", tT, el, out)
    end
else
    rng = frames_spec == "all" ? (1:length(play_times)) :
        (:)(parse.(Int, split(frames_spec, ":"))...)
    for i in rng
        out = joinpath(cache_dir, @sprintf("frame_%04d.jls", i))
        isfile(out) && continue   # resumable
        el = @elapsed serialize(out, frame_payload(play_times[i]))
        @info @sprintf("frame %d/%d  t = %+.2f T0  %.1f s", i, last(rng), play_times[i] / T0, el)
    end
end
@info "precompute done"
