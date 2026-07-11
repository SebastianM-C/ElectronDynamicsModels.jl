# 3D animation of Thomson scattering: a Laguerre–Gauss pulse sweeps through an
# electron cloud at rest. Real simulation (same MTK machinery as scripts/), but
# with "hero" parameters chosen for visual legibility rather than production
# physics: w₀ = 4λ so the OAM helix is visible (production: 75λ), a0 = 1 for
# visible electron response (production: 0.1), few-cycle pulse, small N.
# Physics setup (params, solve, grids, frame times) lives in setup.jl, shared
# with precompute_radiation.jl.
#
# Stage A: trajectory solve + static scene prototypes (two PNGs).
# Stage B: record() animation loop over the observables + scripted camera path.
# Stage C: radiated far-field cube (precompute_radiation.jl output) as a second
#   emissive volume layer — loads automatically when animation/radiation_cube.jls
#   exists (override path with EDM_RADIATION_CUBE).
#
# Run: julia +release -t auto --startup=no --project=animation animation/thomson_animation.jl
# (-t auto matters: the per-frame field sampling is threaded.)

include(joinpath(@__DIR__, "setup.jl"))

using GLMakie

# ── Stage C: radiated far-field cube (optional) ──
# rad :: Float32 (n_frames, slices(X), x(Y), y(Z)) = |E_far|² at the frame times.
# Transfer: |E_far|² spans ~8 decades (1/R² decay + near-electron 1/R spikes on
# pixels that graze trajectories). Clip the ceiling at the 99.5th percentile
# (sacrifices only spike pixels) and log-compress RAD_DECADES below it into
# [0, 1] so the rendering shows the emission wavefronts, not the singularities.
const RADIATION_CUBE = get(ENV, "EDM_RADIATION_CUBE", joinpath(@__DIR__, "radiation_cube.jls"))
const HAS_RADIATION = isfile(RADIATION_CUBE)
const RAD_DECADES = 3.0f0

# Edge fade: the volume's clipped faces otherwise render the departing shell as
# a glowing solid box; smoothstep → 0 over the outer 10% of each axis turns the
# boundary crossing into fog instead.
function _edgewindow(n, frac = 0.1f0)
    m = max(2, round(Int, frac * n))
    w = ones(Float32, n)
    for i in 1:m
        s = Float32(i - 1) / m
        w[i] = w[n + 1 - i] = s * s * (3 - 2s)
    end
    return w
end

if HAS_RADIATION
    using Serialization
    @info "loading radiation cube $RADIATION_CUBE"
    const RAD = deserialize(RADIATION_CUBE)
    _rs = filter(>(0.0f0), vec(@view RAD.rad[1:3:end, 1:5:end, 1:5:end, 1:5:end]))
    const RAD_CEIL = partialsort!(_rs, max(1, round(Int, 0.005 * length(_rs))); rev = true)
    const RAD_FLOOR = RAD_CEIL / exp10(RAD_DECADES)
    const RAD_WIN = reshape(_edgewindow(size(RAD.rad, 2)), :, 1, 1) .*
        reshape(_edgewindow(size(RAD.rad, 3)), 1, :, 1) .*
        reshape(_edgewindow(size(RAD.rad, 4)), 1, 1, :)
    rad_frame_index(t) = clamp(searchsortedlast(RAD.frame_times, t), 1, length(RAD.frame_times))
    function rad_transfer!(dst, f)
        src = @view RAD.rad[f, :, :, :]
        @. dst = RAD_WIN * clamp(log10(max(src, RAD_FLOOR) / RAD_FLOOR) / RAD_DECADES, 0.0f0, 1.0f0)
        return dst
    end
end

# ── Pulse scalar: what the volume rendering shows ──
# A signed field component (E[1] = Ex, chosen) exposes the helical wavefronts as
# +/- lobes twisting around the axis; norm(E) would show only the donut envelope.
# The rendering auto-adapts: signed output → diverging colormap with symmetric
# colorrange; non-negative output → sequential colormap.
function pulse_scalar(E, B)
    E[1]
end

# In-place, threaded, and written directly in SCENE order (Z-fastest), so a
# frame update allocates nothing. The FieldEvaluator is thread-safe (setsym_oop
# is out-of-place); with -t auto this is ~30-50 ms instead of ~0.5 s.
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

# ── Scene ──
# Observables so the animation/slider update vol[]/epos[] in place.
# Takes the three grids in SCENE order (X = physics z, Y = physics x, Z = physics y).
function build_scene(Xs, Ys, Zs, vol0, epos0)
    fig = Figure(size = (1280, 960), backgroundcolor = :black)
    ax = LScene(fig[1, 1]; show_axis = false)

    vol = Observable(vol0)
    epos = Observable(epos0)

    # VolumeLike axes take (start, stop) endpoints; the array implies the spacing
    xe, ye, ze = (first(Xs), last(Xs)), (first(Ys), last(Ys)), (first(Zs), last(Zs))

    signed = minimum(vol0) < 0
    cmax = Float32(maximum(abs, vol0))
    if signed
        contour!(
            ax, xe, ye, ze, vol;
            levels = Float32[-0.5, 0.5] .* cmax,
            colormap = :balance, colorrange = (-cmax, cmax),
            alpha = 0.22, transparency = true,
        )
    else
        volume!(
            ax, xe, ye, ze, vol;
            algorithm = :absorption, absorption = 4.0f0,
            colormap = :inferno, colorrange = (0.0f0, cmax),
        )
    end

    # :additive + transparency — the only compositing combo where all three
    # layers survive: :absorption/OIT dims what's behind the box, and either
    # algorithm without the transparency flag writes depth and occludes the
    # pulse contour and electrons outright. The residual WBOIT quirk (an
    # ALL-ZERO volume still drags the weighted average dark, dimming the gold
    # disk) is handled in set_time! by hiding the plot on glow-free frames.
    radvol = nothing
    radplot = nothing
    if HAS_RADIATION
        radvol = Observable(zeros(Float32, size(RAD.rad, 2), size(RAD.rad, 3), size(RAD.rad, 4)))
        rXe = (first(RAD.slice_zs), last(RAD.slice_zs))
        rYe = (first(RAD.txs), last(RAD.txs))
        rZe = (first(RAD.tys), last(RAD.tys))
        # Alpha ramp from EXACTLY 0: zero-field fragments then carry zero OIT
        # weight, so empty regions of the box cannot dim geometry behind them.
        cmap = Makie.to_colormap(:afmhot)
        rad_cmap = [Makie.RGBAf(c.r, c.g, c.b, (i - 1) / (length(cmap) - 1)) for (i, c) in enumerate(cmap)]
        radplot = volume!(
            ax, rXe, rYe, rZe, radvol;
            algorithm = :additive,
            colormap = rad_cmap, colorrange = (0.0f0, 0.7f0),
            transparency = true, visible = false,
        )
    end

    meshscatter!(
        ax, epos;
        markersize = 0.28λ, color = :gold,
        shading = true,
    )

    # Scene axes put propagation on X, so the Makie default z-up camera and its
    # fixed-axis orbiting behave naturally: eye on the −Y side makes the pulse
    # travel left → right; distance frames the extended ±12λ box.
    update_cam!(ax.scene, Vec3f(2.5w₀, -8w₀, 3w₀), Vec3f(0), Vec3f(0, 0, 1))

    return fig, ax, vol, epos, radvol, radplot
end

set_time!(t) = begin
    sample_pulse!(vol[], fe, xs, ys, zs, t)
    notify(vol)
    epos[] = electron_positions(trajs, t)
    if HAS_RADIATION
        rad_transfer!(radvol[], rad_frame_index(t))
        notify(radvol)
        # The visible volume costs a small constant WBOIT dim on everything
        # behind it. Keep it ON through approach + crossing so that cost never
        # STEPS in at ignition (the brightest moment, where the eye catches
        # it); allow the hide only in the fading tail, where the empty-box dim
        # is all the volume would contribute (the olive-disk artifact) and the
        # scene is too dark for the toggle to read. Mean, not max: the log
        # transfer amplifies faint wisps that fill a tiny fraction of the box.
        # Hysteresis: the tail decays slowly, a single cut would flicker.
        m = sum(radvol[]) / length(radvol[])
        rad_visible[] = t <= 2T0 || (rad_visible[] ? (m > 4.0f-4) : (m > 8.0f-4))
        radplot.visible = rad_visible[]
    end
    return nothing
end

# ── Static prototypes: pulse approaching (Act 1 opening) + peak crossing ──
t_snap = 0.0
@info "sampling pulse volume at t = $t_snap ($(Threads.nthreads()) threads)"
vol0 = sample_pulse!(Array{Float32}(undef, nz, nx, ny), fe, xs, ys, zs, t_snap)
epos0 = electron_positions(trajs, t_snap)

fig, ax, vol, epos, radvol, radplot = build_scene(zs, xs, ys, vol0, epos0)
const rad_visible = Ref(true)

if HAS_RADIATION   # backfill the radiation layer for the t_snap frame
    rad_transfer!(radvol[], rad_frame_index(t_snap))
    notify(radvol)
    rad_visible[] = sum(radvol[]) / length(radvol[]) > 4.0f-4
    radplot.visible = rad_visible[]
end
outfile = joinpath(@__DIR__, "static_frame.png")
save(outfile, fig)
@info "saved $outfile"

set_time!(-5T0)
outfile2 = joinpath(@__DIR__, "static_frame_approach.png")
save(outfile2, fig)
@info "saved $outfile2"

if HAS_RADIATION   # flash mid-expansion: radiation shell chasing the departing pulse
    set_time!(4T0)
    outfile3 = joinpath(@__DIR__, "static_frame_radiation.png")
    save(outfile3, fig)
    @info "saved $outfile3"
end

# ── Interactive exploration ──
# Attach a time slider + camera-save button and open the window. Dragging the
# slider re-samples the pulse (throttled). "save camera" appends the current
# view as a ready-to-paste (eye, lookat, up) tuple (w₀ units) plus the matching
# s to animation/camera_keyframes.jl.
# The controls become part of the figure, so re-include before animate() to
# keep them out of the video.
const controls_attached = Ref(false)

function interactive()
    if !controls_attached[]
        controls_attached[] = true
        sg = SliderGrid(
            fig[2, 1],
            (label = "t / T₀", range = (t_start / T0):0.05:(t_end / T0), startvalue = t_snap / T0,
                format = v -> string(round(v; digits = 2))),
        )
        tobs = sg.sliders[1].value
        on(Makie.Observables.throttle(0.1, tobs)) do tT
            set_time!(tT * T0)
        end
        btn = Button(fig[3, 1]; label = "save camera", tellwidth = false, halign = :left)
        on(btn.clicks) do _
            cam = cameracontrols(ax.scene)
            inw₀(v) = "Vec3f(" * join(string.(round.(Float64.(v) ./ w₀; digits = 3)) .* "w₀", ", ") * ")"
            raw(v) = "Vec3f(" * join(string.(round.(Float64.(v); digits = 4)), ", ") * ")"
            s = (tobs[] * T0 - t_start) / (t_end - t_start)
            snippet = "# s = $(round(s; digits = 3))\n" *
                "(t = $(round(tobs[]; digits = 2))T0, eye = $(inw₀(cam.eyeposition[])), " *
                "lookat = $(inw₀(cam.lookat[])), up = $(raw(cam.upvector[]))),\n"
            kf = joinpath(@__DIR__, "camera_keyframes.jl")
            open(io -> write(io, snippet), kf, "a")
            @info "camera keyframe appended to $kf\n$snippet"
        end
    end
    return display(fig)
end

if get(ENV, "EDM_INTERACTIVE", "0") == "1"
    wait(interactive())
end

# ── Stage B: animation ──
# The pulse peak crosses the box left-to-right, then the long tail shows the
# scattered electrons drifting; each frame re-samples the analytic field and
# re-positions the electrons, then the camera moves along camera_path.

# Camera path: ease through saved keyframes (from interactive()'s "save camera"
# button — paste new entries here). Keyed by lab time, not s, so they survive
# changes of the animation window; the camera holds before the first and after
# the last keyframe. Segments blend with smoothstep easing.
const CAMERA_KEYFRAMES = [
    (t = -11.6T0, eye = Vec3f(2.695w₀, -9.753w₀, 0.837w₀), lookat = Vec3f(0), up = Vec3f(-0.022, 0.0795, 0.9966)),
    (t = 4.9T0, eye = Vec3f(4.895w₀, -8.811w₀, 1.221w₀), lookat = Vec3f(0), up = Vec3f(-0.0584, 0.1051, 0.9927)),
]

smoothstep(x) = x * x * (3 - 2x)

function camera_path(s)
    t = t_start + s * (t_end - t_start)
    kfs = CAMERA_KEYFRAMES
    t <= kfs[1].t && return (kfs[1].eye, kfs[1].lookat, kfs[1].up)
    t >= kfs[end].t && return (kfs[end].eye, kfs[end].lookat, kfs[end].up)
    i = findlast(k -> k.t <= t, kfs)
    a, b = kfs[i], kfs[i + 1]
    f = Float32(smoothstep((t - a.t) / (b.t - a.t)))
    lerp(u, v) = (1 - f) .* u .+ f .* v
    return (lerp(a.eye, b.eye), lerp(a.lookat, b.lookat), Vec3f(normalize(lerp(a.up, b.up))))
end

function animate(; file = joinpath(@__DIR__, "thomson.mp4"),
        t_range = frame_times, framerate = 30)
    controls_attached[] &&
        @warn "interactive controls are part of the figure and will show up in the video; re-include for a clean render"
    record(fig, file, t_range; framerate) do t
        set_time!(t)
        s = (t - t_start) / (t_end - t_start)
        eye, lookat, up = camera_path(s)
        update_cam!(ax.scene, eye, lookat, up)
    end
    @info "saved $file"
end

if get(ENV, "EDM_ANIMATE", "0") == "1"
    animate()
end
