# Trajectory-side products shared between the live solvers (thomson_scattering.jl,
# inverse_thomson_scattering.jl) and the backfill path (gammatau_backfill.jl): the γ(τ)/γ₀ trace
# reduction, its batch accumulator + cache + chip, the lab-time emission profile, and the as-run
# initial-conditions cache + chip. Included, not a module — the includer provides
# Serialization, CairoMakie, Printf, and RunManifests (write_derived), same contract as
# harmonic_products.jl.

# γ(τ)/γ₀ across the ensemble, sampled through the trajectory INTERPOLANTS (the uniform-saveat
# CubicSplines the radiation kernel integrates), read between knots at the caller's τ grid — so
# saveat-undersampling and spline artifacts stay visible instead of hidden by the solver's
# dense output. Returns (γmean, γmin, γmax, drain): disk mean/min/max at each τ plus the
# per-electron end-state drain 1 − γ(τf)/γ₀.
function gamma_trace(trajs, τs, c, γ0, τf)
    # One accumulator per electron CHUNK, not per threadid: threadid() can exceed nthreads()
    # (interactive pool) and tasks migrate, so threadid-indexed accumulators are unsound.
    nch = max(1, min(Threads.nthreads(), length(trajs)))
    chunks = collect(Iterators.partition(eachindex(trajs), cld(length(trajs), nch)))
    sums = [zeros(length(τs)) for _ in chunks]
    los = [fill(Inf, length(τs)) for _ in chunks]
    his = [fill(-Inf, length(τs)) for _ in chunks]
    drain = zeros(length(trajs))
    tasks = map(enumerate(chunks)) do (ci, ch)
        Threads.@spawn begin
            s, lo, hi = sums[ci], los[ci], his[ci]
            for e in ch
                tr = trajs[e]
                @inbounds for (k, τk) in enumerate(τs)
                    γ = tr(τk)[2][1] / c
                    s[k] += γ
                    γ < lo[k] && (lo[k] = γ)
                    γ > hi[k] && (hi[k] = γ)
                end
                drain[e] = 1 - tr(τf)[2][1] / ((γ0 isa Number ? γ0 : γ0[e]) * c)   # γ0: shared or per electron
            end
        end
    end
    foreach(wait, tasks)
    return reduce(+, sums) ./ length(trajs),
        reduce((a, b) -> min.(a, b), los), reduce((a, b) -> max.(a, b), his), drain
end

# The live solvers' batch-by-batch γ(τ) trace (thomson_scattering.jl and inverse_thomson_scattering.jl):
# a τ grid at `oversample`× the trajectory-knot rate, folded exactly across electron batches (sums and
# elementwise extrema; the drain vector in electron order), then one cache + chip per run.
#   acc = gamma_trace_acc(τi, τf, knot_dt, oversample)          # oversample = 0 ⇒ empty grid, trace off
#   fold_gamma!(acc, gamma_trace(trajs, acc.τs, c, γ0, τf), n)  # per batch
#   write_gamma_trace(outdir, run_tag, acc, N; γ0, ω, τ_pulse, knots_per_period)
function gamma_trace_acc(τi, τf, knot_dt, oversample)
    τs = oversample > 0 ? collect(τi:(knot_dt / oversample):τf) : Float64[]
    return (; τs, sum = zeros(length(τs)), lo = fill(Inf, length(τs)), hi = fill(-Inf, length(τs)),
        drain = Float64[], oversample)
end

function fold_gamma!(acc, γ, n)
    γm, γn, γx, γd = γ
    acc.sum .+= γm .* n   # batch means → ensemble mean (÷ N at write time)
    acc.lo .= min.(acc.lo, γn)
    acc.hi .= max.(acc.hi, γx)
    append!(acc.drain, γd)
    return acc
end

function write_gamma_trace(outdir, run_tag, acc, N; γ0, ω, τ_pulse, knots_per_period)
    gt = (; τs = acc.τs, γ0 = Float64(γ0), ω, τ_pulse, γmean = acc.sum ./ N, γmin = acc.lo, γmax = acc.hi,
        drain = acc.drain, oversample = acc.oversample, knots_per_period = Float64(knots_per_period))
    gtfile = joinpath(outdir, "gammatau_$(run_tag).jls")
    serialize(gtfile, gt)
    write_gamma_trace_chip(outdir, run_tag, gt)
    return gtfile
end

# Per-run chip from the cache (re-renders anywhere): γ/γ₀ mean with the min–max band over the disk vs
# τ, and the histogram of the per-electron net energy change γ_f − γ₀ = −γ₀·drain.
function write_gamma_trace_chip(outdir, run_tag, gt)
    Δγ = -gt.γ0 .* gt.drain
    fig = Figure(size = (1150, 420))
    ax = Axis(fig[1, 1]; title = "γ(τ)/γ₀ over the disk (mean, min–max band)", xlabel = "τ / τ_pulse", ylabel = "γ / γ₀")
    x = gt.τs ./ gt.τ_pulse
    band!(ax, x, gt.γmin ./ gt.γ0, gt.γmax ./ gt.γ0; color = (:steelblue, 0.3))
    lines!(ax, x, gt.γmean ./ gt.γ0; color = :steelblue, linewidth = 1.5)
    ax2 = Axis(fig[1, 2]; title = "net energy change per electron", xlabel = "γ_f − γ₀", ylabel = "electrons")
    hist!(ax2, Δγ; bins = 60, color = (:steelblue, 0.8))
    Label(fig[0, :], @sprintf("γ(τ) trace — %s  (γ₀ = %.6g; γ_f − γ₀ ∈ [%.3g, %.3g], mean %.3g)",
        first(run_tag, 8), gt.γ0, extrema(Δγ)..., sum(Δγ) / length(Δγ)); fontsize = 15, font = :bold)
    png = joinpath(outdir, "gammatau_$(run_tag).png")
    save(png, fig)
    write_derived(outdir; kind = "gamma_trace", label = "γ(τ) trace + net energy change", run_id = run_tag,
        plot = basename(png), source = "gammatau_$(run_tag).jls",
        plot_params = Dict("gamma0" => gt.γ0, "dgamma_min" => minimum(Δγ), "dgamma_max" => maximum(Δγ),
            "dgamma_mean" => sum(Δγ) / length(Δγ), "oversample" => gt.oversample,
            "knots_per_period" => gt.knots_per_period),
        description = "γ/γ₀ read through the trajectory interpolants the radiation kernel integrates " *
            "(mean over the disk with the min–max band) against τ, and the histogram of each " *
            "electron's net energy change γ_f − γ₀ at the end of the physics span.")
    return png
end

# Emission-time profile: WHEN the ensemble radiates, in lab time. Each electron is sampled on a proper-time
# grid through the same interpolants the radiation kernel integrates (4-acceleration from `a_itp`), and its
# emitted-energy weight per step is binned by lab time t = x⁰/c in laser periods (0 = the pulse peak at the
# focus). Bins are a Dict so boosted (inverse) runs, whose lab time runs ≫ τ, need no preset range.
# Per electron it also records the END of its own emission (see emission_end) and where it is then;
# batches fold in electron order, so `acc.ends` is in sunflower order.
#   acc = emission_time_acc(τi, τf, dτ)                               # dτ ≤ 0 ⇒ off
#   fold_emission!(acc, emission_time(trajs, acc.τs, c, T))           # per batch
#   write_emission_time(outdir, run_tag, acc; T, window_periods, w0, Rdisc)

# Lab-frame radiated energy of one proper-time step dτ: the invariant Larmor power ∝ −𝔞μ𝔞^μ times the lab
# duration dt = γ dτ (u⁰ = γc, metric (+,−,−,−)). dτ is constant on the grid, so it drops out of the
# normalized profile; max(0, …) absorbs spline round-off where 𝔞 ≈ 0.
emission_weight(uμ, 𝔞μ, c, dτ) = max(0.0, -ElectronDynamicsModels.m_dot(𝔞μ, 𝔞μ)) * uμ[1] / c

# End of one electron's emission: the first sample by which 99 % of its own emitted energy is out (the
# chip's t99, per electron — an energy integral, so invariant to the τ grid at any γ). `nothing` if it
# emits nothing.
function emission_end(w)
    W = sum(w)
    W > 0 || return nothing
    return findfirst(>=(0.99W), cumsum(w))
end

# t: lab periods, x: 4-position (a.u.) at the end, ρ0: start radius, W: total weight (NaN t/x if none).
const EmissionEnd = NamedTuple{(:t, :x, :ρ0, :W), Tuple{Float64, NTuple{4, Float64}, Float64, Float64}}

function emission_time(trajs, τs, c, T)
    nch = max(1, min(Threads.nthreads(), length(trajs)))
    chunks = collect(Iterators.partition(eachindex(trajs), cld(length(trajs), nch)))
    dτ = length(τs) > 1 ? τs[2] - τs[1] : 0.0
    parts = [Dict{Int, Float64}() for _ in chunks]
    ends = Vector{EmissionEnd}(undef, length(trajs))
    tasks = map(enumerate(chunks)) do (ci, ch)
        Threads.@spawn begin
            d = parts[ci]
            w = zeros(length(τs))
            for e in ch
                ρ0 = NaN
                for (k, τk) in enumerate(τs)
                    xμ, uμ, 𝔞μ = ElectronDynamicsModels.state_with_acceleration(trajs[e], τk)
                    k == 1 && (ρ0 = hypot(xμ[2], xμ[3]))
                    b = floor(Int, xμ[1] / c / T)
                    w[k] = emission_weight(uμ, 𝔞μ, c, dτ)
                    d[b] = get(d, b, 0.0) + w[k]
                end
                ke = emission_end(w)
                ends[e] = if ke === nothing
                    (; t = NaN, x = (NaN, NaN, NaN, NaN), ρ0, W = 0.0)
                else
                    xμ = ElectronDynamicsModels.state_with_acceleration(trajs[e], τs[ke])[1]
                    (; t = xμ[1] / c / T, x = Tuple(xμ[1:4]), ρ0, W = sum(w))
                end
            end
        end
    end
    foreach(wait, tasks)
    return (; bins = reduce(mergewith!(+), parts), ends)
end

emission_time_acc(τi, τf, dτ) = (; τs = dτ > 0 ? collect(τi:dτ:τf) : Float64[], bins = Dict{Int, Float64}(), ends = EmissionEnd[])
fold_emission!(acc, d) = (mergewith!(+, acc.bins, d.bins); append!(acc.ends, d.ends); acc)

function write_emission_time(outdir, run_tag, acc; T, window_periods, w0 = nothing, Rdisc = nothing, note = nothing)
    ks = sort!(collect(keys(acc.bins)))
    t = Float64.(ks) .+ 0.5                                   # bin centres, periods
    w = [acc.bins[k] for k in ks]
    # per electron, sunflower order: own 99 % emission end (lab periods), 4-position then (N×4, a.u.),
    # start radius (a.u.) and total emitted weight (same units as w)
    t_end = [e.t for e in acc.ends]
    x_end = isempty(acc.ends) ? zeros(0, 4) : permutedims(reduce(hcat, [collect(e.x) for e in acc.ends]))
    et = (; t, w, T, window_periods, dτ = length(acc.τs) > 1 ? acc.τs[2] - acc.τs[1] : NaN, note,
        t_end, x_end, ρ0 = [e.ρ0 for e in acc.ends], W_end = [e.W for e in acc.ends], w0, Rdisc)
    file = joinpath(outdir, "emissiontime_$(run_tag).jls")
    serialize(file, et)
    write_emission_time_chip(outdir, run_tag, et)
    any(isfinite, t_end) && w0 !== nothing && Rdisc !== nothing && write_emission_end_chip(outdir, run_tag, et)
    return file
end

# Chip: where each electron is when its own emission ends (99 % of its energy out), colored by that lab
# time, with the start disc's edge dashed; and the end time against the starting radius.
function write_emission_end_chip(outdir, run_tag, et)
    em = isfinite.(et.t_end)
    interior = et.ρ0 .< et.Rdisc * (1 - 1e-9)                 # sunflower edge points start ON R_disc
    ρe = hypot.(et.x_end[:, 2], et.x_end[:, 3])
    sel = em .& interior
    frac_out = count(ρe[sel] .> et.Rdisc) / max(1, count(sel))
    lo, hi = extrema(et.t_end[em])
    ord = filter(i -> em[i], sortperm(et.t_end))              # latest on top
    fig = Figure(size = (1250, 540))
    ax1 = Axis(fig[1, 1]; title = "position when each electron's emission ends", xlabel = "x / w₀", ylabel = "y / w₀", aspect = 1)
    sc = scatter!(ax1, et.x_end[ord, 2] ./ et.w0, et.x_end[ord, 3] ./ et.w0; color = et.t_end[ord], colormap = :viridis, markersize = 4)
    φs = range(0, 2π, length = 361)
    lines!(ax1, et.Rdisc / et.w0 .* cos.(φs), et.Rdisc / et.w0 .* sin.(φs); color = :crimson, linestyle = :dash, linewidth = 1.5)
    Colorbar(fig[1, 2], sc; label = "emission end t (periods, 0 = pulse peak at the focus)")
    ax2 = Axis(fig[1, 3]; title = "emission end by starting radius", xlabel = "ρ₀ / w₀", ylabel = "emission end t (periods)")
    scatter!(ax2, et.ρ0[em] ./ et.w0, et.t_end[em]; markersize = 3, color = (:steelblue, 0.6))
    Label(fig[0, :], @sprintf("emission end — %s  (99 %% of each electron's energy out; %.1f %% outside R_disc then; t ∈ [%.0f, %.0f] periods)",
        first(run_tag, 8), 100frac_out, lo, hi); fontsize = 15, font = :bold)
    png = joinpath(outdir, "emitend_$(run_tag).png")
    save(png, fig)
    write_derived(outdir; kind = "emission_end", label = @sprintf("emission end: %.1f %% outside R_disc", 100frac_out), run_id = run_tag,
        plot = basename(png), source = "emissiontime_$(run_tag).jls",
        plot_params = merge(Dict{String, Any}("frac_outside_Rdisc" => frac_out, "t_end_min" => lo, "t_end_max" => hi,
            "t_end_median" => sort(et.t_end[em])[cld(count(em), 2)], "n_edge" => count(.!interior)),
            et.note === nothing ? Dict{String, Any}() : Dict{String, Any}("provenance_note" => et.note)),
        description = "Where the electrons are when they stop radiating: for each electron, the lab time by which " *
            "99 % of its own emitted energy is out (the emission-time chip's weight, per electron) and its transverse " *
            "position at that moment, colored by that time; the start disc's edge dashed. The fraction outside " *
            "R_disc counts interior electrons (the sunflower's edge points start on it). Right: the end time " *
            "against the starting radius.")
    return png
end

# Chip: normalized emission rate and its cumulative fraction vs lab time, with the 50 / 90 / 99 % times.
function write_emission_time_chip(outdir, run_tag, et)
    cs = cumsum(et.w) ./ sum(et.w)
    tp(p) = et.t[findfirst(>=(p), cs)]
    t50, t90, t99 = tp(0.5), tp(0.9), tp(0.99)
    lo, hi = tp(1e-4) - 5, tp(0.999) + 5
    sel = lo .<= et.t .<= hi
    fig = Figure(size = (1150, 420))
    ax1 = Axis(fig[1, 1]; title = "emission rate (normalized)", xlabel = "lab time t (periods, 0 = pulse peak at the focus)", ylabel = "fraction per period")
    lines!(ax1, et.t[sel], et.w[sel] ./ sum(et.w); color = :steelblue, linewidth = 1.5)
    ax2 = Axis(fig[1, 2]; title = "cumulative emission", xlabel = "lab time t (periods)", ylabel = "fraction radiated", limits = (nothing, (0, 1.02)))
    lines!(ax2, et.t[sel], cs[sel]; color = :steelblue, linewidth = 2)
    for (p, tq) in ((50, t50), (90, t90), (99, t99))
        vlines!(ax2, [tq]; color = (:gray40, 0.8), linewidth = 1, linestyle = :dot)
        text!(ax2, tq, 0.04; text = " $(p) %", fontsize = 11, color = :gray30)
    end
    Label(fig[0, :], @sprintf("emission time — %s  (50 / 90 / 99 %% radiated by t = %.1f / %.1f / %.1f periods)",
        first(run_tag, 8), t50, t90, t99); fontsize = 15, font = :bold)
    png = joinpath(outdir, "emissiontime_$(run_tag).png")
    save(png, fig)
    write_derived(outdir; kind = "emission_time", label = @sprintf("emission time: 99 %% by t = %.0f periods", t99), run_id = run_tag,
        plot = basename(png), source = "emissiontime_$(run_tag).jls",
        plot_params = merge(Dict{String, Any}("t50_periods" => t50, "t90_periods" => t90, "t99_periods" => t99,
            "window_periods" => et.window_periods, "dtau" => et.dτ),
            get(et, :note, nothing) === nothing ? Dict{String, Any}() : Dict{String, Any}("provenance_note" => et.note)),
        description = "When the electron ensemble radiates: the emitted-energy weight of every electron, sampled " *
            "through the trajectory interpolants the radiation kernel integrates, binned by LAB time (laser periods, " *
            "0 = pulse peak at the focus). Left: normalized rate; right: cumulative fraction with the times by " *
            "which 50 / 90 / 99 % is radiated. Lab time, not observer time: arrival at the screen is compressed " *
            "by (1 − β·n) for forward-moving electrons.")
    return png
end

# As-run initial conditions: cache + chip. The disk and its Δz offsets are deterministic
# from [config], but the cache pins the EXACT as-run xμ₀/u₀ (reconstruction drift becomes
# visible instead of silent) and lets the IC chip re-render anywhere without an EDM solve —
# the same publish-autonomy contract as the γ(τ) trace (both are enumerated in the .reduced
# marker at reduce time). `datafile` on the sidecar ships the cache to the archive store and
# puts a /data download URL on the chip. Cheap (N×6 floats ≈ 100 KB at N = 2000): always on.
#
# The chip is MODE-AWARE, because unbunched runs (nb = 0, most campaigns) have Δz ≡ 0 and a
# Δz-only chip degenerates to a flat disk over a flat line. Panel 1 colors the disk by the
# local LG amplitude |u_rel| (closed form, p = 2 |m| = 2 — the production mode) — who actually
# radiates — and panel 2 shows the arrival surface when bunched, or the sampled mode profile
# |u_rel|(ρ) when not. Other (p, m): panel 1 falls back to Δz coloring, no weight panel.
function write_ic_products(xμ0, u0, dz, outdir, run_tag; γ0, λ, w₀, nb, l, chirp, p = 2, m = -2)
    icfile = joinpath(outdir, "ic_$(run_tag).jls")
    X = permutedims(reduce(hcat, xμ0))       # N×4 as-run start 4-positions (bunch_dz included)
    x, y = X[:, 2] ./ λ, X[:, 3] ./ λ
    dzλ = collect(dz) ./ λ
    ρw₀ = sqrt.(x .^ 2 .+ y .^ 2) .* (λ / w₀)
    urel = (Int(p) == 2 && abs(Int(m)) == 2) ?
        (σ = ρw₀ .^ 2; abs.(√12 .* 2 .* σ .* (1 .- 4 .* σ ./ 3 .+ σ .^ 2 ./ 3) .* exp.(-σ))) :
        nothing
    neff = urel === nothing ? nothing : sum(urel)^2 / (length(urel) * sum(abs2, urel))
    serialize(icfile, (; xμ0 = X, u0 = collect(u0), dz = collect(dz), γ0, λ, w₀,
        p = Int(p), m = Int(m), u_rel = urel, bunch = (; nb, l, chirp)))
    fig = Figure(size = (1080, 480))
    ax1 = Axis(fig[1, 1]; title = urel === nothing ?
            "start disk, colored by bunching offset Δz" :
            "start disk, colored by the local drive amplitude",
        xlabel = "x  [λ]", ylabel = "y  [λ]", aspect = 1)
    sc = scatter!(ax1, x, y; color = urel === nothing ? dzλ : urel, markersize = 5,
        colormap = urel === nothing ? :viridis : :inferno)
    if urel !== nothing
        # LG mode intensity contours over the disk — the continuous field the points sample
        # (same closed form as the point coloring; axes are in λ, σ wants w₀).
        ext = 1.08 * max(maximum(abs, x), maximum(abs, y))
        gr = range(-ext, ext, length = 301)
        amp = (xx, yy) -> (σ = (xx^2 + yy^2) * λ^2 / w₀^2;
            abs(√12 * 2σ * (1 - 4σ / 3 + σ^2 / 3) * exp(-σ)))
        contour!(ax1, gr, gr, [amp(xx, yy) for xx in gr, yy in gr];
            levels = 6, color = (:gray40, 0.6), linewidth = 0.8)
    end
    Colorbar(fig[1, 2], sc; label = urel === nothing ? "Δz  [λ]" : "|u_rel|")
    if nb != 0
        ax2 = Axis(fig[1, 3]; title = "arrival surface: lens parabola + helix spread",
            xlabel = "(ρ/w₀)²", ylabel = "Δz  [λ]")
        scatter!(ax2, ρw₀ .^ 2, dzλ; color = atan.(y, x), colormap = :twilight, markersize = 4)
    elseif urel !== nothing
        ax2 = Axis(fig[1, 3]; title = @sprintf("mode sampling — N_eff/N = %.2f", neff),
            xlabel = "ρ / w₀", ylabel = "|u_rel|")
        scatter!(ax2, ρw₀, urel; color = (:crimson, 0.5), markersize = 4)
    end
    Label(fig[0, :], @sprintf("as-run initial conditions — N = %d, γ = %g, n_b = %d, ℓ = %d",
        length(x), γ0, nb, l), fontsize = 17)
    out = joinpath(outdir, "inverse_thomson_ic_$(run_tag).png")
    save(out, fig)
    pp = Dict{String, Any}("N" => length(x), "bunch_nb" => nb, "bunch_l" => l,
        "max |Δz| [λ]" => round(maximum(abs, dzλ); sigdigits = 3))
    neff === nothing || (pp["N_eff/N"] = round(neff; sigdigits = 3))
    write_derived(
        outdir; kind = "ic", label = "initial conditions (as run)",
        run_id = run_tag, plot = basename(out), datafile = basename(icfile),
        plot_params = pp,
        description = "The exact as-run start disk. Left: transverse sunflower positions " *
            "colored by the local LG amplitude |u_rel| (who actually radiates; the weight " *
            "behind the N_eff ceiling). Right: bunched runs show the arrival surface — Δz " *
            "against (ρ/w₀)², the lens parabola with the ℓθ helix as azimuthal spread — and " *
            "unbunched runs the sampled mode profile |u_rel|(ρ) with its rings and nodes. The " *
            "`ic_<id>.jls` cache stores xμ₀/u₀/Δz/|u_rel|, so this chip re-renders without an " *
            "EDM solve.",
    )
    println("saved → $(basename(out))")
    return icfile
end

# Per-electron Δγ/γ₀ over the start disk — WHERE the drain happens, the spatial complement of
# gamma_drain_product's γ(τ) view. Reads the pair's gammatau_ drains (trajectory order = disk
# order) and the LL run's ic_ cache. The right panel tests the drain against the LOCAL drive:
# the small-drain law 3.5e-7·a₀²γ evaluated at a₀|u_rel| per electron — points leaving the
# dashed curve are the law bending — with the classical drains as the numerical-residual
# control. 2 parents route the chip to the comparison card, next to the γ(τ) overlay.
function drain_disk_product(dir, cl, ll, γ, a0)
    fic = joinpath(dir, "ic_$(ll.id).jls")
    fcl = joinpath(dir, "gammatau_$(cl.id).jls")
    fll = joinpath(dir, "gammatau_$(ll.id).jls")
    all(isfile, (fic, fcl, fll)) ||
        return println("drain disk: missing ic/gammatau caches for the γ=$γ a₀=$a0 pair — skip")
    ic = deserialize(fic)
    d_cl, d_ll = deserialize(fcl).drain, deserialize(fll).drain
    N = size(ic.xμ0, 1)
    (length(d_ll) == N && length(d_cl) == N) ||
        return println("drain disk: trace/disk N mismatch for the γ=$γ a₀=$a0 pair — skip")
    x, y = ic.xμ0[:, 2] ./ ic.λ, ic.xμ0[:, 3] ./ ic.λ
    fig = Figure(size = (1080, 480))
    ax1 = Axis(fig[1, 1]; title = "Δγ/γ₀ over the start disk  (Landau–Lifshitz)",
        xlabel = "x  [λ]", ylabel = "y  [λ]", aspect = 1)
    sc = scatter!(ax1, x, y; color = d_ll, colormap = :viridis, markersize = 5)
    Colorbar(fig[1, 2], sc; label = "Δγ/γ₀")
    if ic.u_rel === nothing
        ρw₀ = sqrt.(x .^ 2 .+ y .^ 2) .* (ic.λ / ic.w₀)
        ax2 = Axis(fig[1, 3]; title = "drain against radius", xlabel = "ρ / w₀", ylabel = "Δγ/γ₀")
        scatter!(ax2, ρw₀, d_cl; color = (:seagreen, 0.4), markersize = 4, label = "classical")
        scatter!(ax2, ρw₀, d_ll; color = (:crimson, 0.5), markersize = 4, label = "Landau–Lifshitz")
    else
        ax2 = Axis(fig[1, 3]; title = "drain against the local drive",
            xlabel = "|u_rel|", ylabel = "Δγ/γ₀")
        us = range(0, maximum(ic.u_rel); length = 200)
        lines!(ax2, us, 3.5e-7 .* (a0 .* us) .^ 2 .* γ; color = :gray40, linestyle = :dash,
            label = "3.5×10⁻⁷ (a₀|u_rel|)² γ")
        scatter!(ax2, ic.u_rel, d_cl; color = (:seagreen, 0.4), markersize = 4, label = "classical")
        scatter!(ax2, ic.u_rel, d_ll; color = (:crimson, 0.5), markersize = 4,
            label = "Landau–Lifshitz")
    end
    axislegend(ax2; position = :lt)
    Label(fig[0, :], @sprintf(
        "γ=%g  a₀=%g — per-electron radiation-reaction drain over the disk  (N = %d)", γ, a0, N),
        fontsize = 17)
    out = joinpath(dir, @sprintf("inverse_thomson_drain_disk_%s-%s.png",
        first(ll.id, 8), first(cl.id, 8)))
    save(out, fig)
    write_derived(
        dir; kind = "drain_disk", label = "Δγ/γ₀ disk map — where the drain happens",
        run_id = [cl.id, ll.id], plot = basename(out), source = "gammatau_$(ll.id).jls",
        plot_params = Dict(
            "Δγ/γ (LL, disk mean)" => round(sum(d_ll) / N; sigdigits = 3),
            "Δγ/γ (LL, max)" => round(maximum(d_ll); sigdigits = 3),
            "classical residual (max |Δγ/γ|)" => round(maximum(abs, d_cl); sigdigits = 2),
            "linear law at peak drive" => round(3.5e-7 * a0^2 * γ; sigdigits = 3)),
        description = "Per-electron end-state drain Δγ/γ₀ = 1 − γ(τf)/γ₀ mapped onto the start " *
            "disk (left) and against the local drive amplitude |u_rel| (right), with the " *
            "small-drain law 3.5×10⁻⁷(a₀|u_rel|)²γ dashed — per-electron departure from the " *
            "curve is the law bending; the classical drains ride along as the numerical-" *
            "residual control. Same trajectory splines as the γ(τ) overlay.",
    )
    println("saved → $(basename(out))")
    return
end

# Incoherent angular energy (EDM_ANGULAR_ENERGY=1): the ensemble's dW/dΩ on a far-field direction grid,
# summed in intensity over the electrons (ElectronDynamicsModels.angular_energy), with the 1/e half-widths
# of Wei et al.'s divergence definition (half the full width at 1/e of the peak). θ in rad; W in
# energy per steradian (a.u.). Cache angenergy_<tag>.jls + chip + derived sidecar. `tilt` (rad): the grid is centred
# on the beam direction, rotated by it about ŷ (far_field_directions), so θx, θy and the half-widths are about the beam.
function write_angular_energy(outdir, run_tag, W, θx, θy; zsign, oversample, N, tilt = 0.0, note = nothing)
    wid = one_over_e_halfwidths(W, θx, θy)
    dΩ = (θx[end] - θx[1]) / (length(θx) - 1) * (θy[end] - θy[1]) / (length(θy) - 1)
    ae = (; θx = collect(Float64, θx), θy = collect(Float64, θy), W, zsign, oversample, N, tilt = Float64(tilt),
        halfwidth_x = wid.x, halfwidth_y = wid.y, halfwidth_req = wid.r_eq, W_grid = sum(W) * dΩ, note)
    aefile = joinpath(outdir, "angenergy_$(run_tag).jls")
    serialize(aefile, ae)
    write_angular_energy_chip(outdir, run_tag, ae)
    return aefile
end

function write_angular_energy_chip(outdir, run_tag, ae)
    mr = 1e3
    tx, ty = ae.θx .* mr, ae.θy .* mr
    peak, I = findmax(ae.W)
    fig = Figure(size = (1150, 460))
    ax = Axis(fig[1, 1]; title = "dW/dΩ (incoherent, normalized)", xlabel = "θx (mrad)", ylabel = "θy (mrad)", aspect = 1)
    hm = heatmap!(ax, tx, ty, ae.W ./ peak; colormap = :inferno)
    contour!(ax, tx, ty, ae.W ./ peak; levels = [exp(-1)], color = :white, linewidth = 1.5)
    Colorbar(fig[1, 2], hm)
    ax2 = Axis(fig[1, 3]; title = "cuts through the peak", xlabel = "θ (mrad)", ylabel = "dW/dΩ / peak")
    lines!(ax2, tx, ae.W[:, I[2]] ./ peak; label = "θx")
    lines!(ax2, ty, ae.W[I[1], :] ./ peak; label = "θy")
    hlines!(ax2, [exp(-1)]; color = :gray, linestyle = :dash)
    axislegend(ax2)
    Label(fig[0, :], @sprintf("angular energy — %s  (1/e half-width θx %.3g, θy %.3g, r_eq %.3g mrad; N = %d%s)",
        first(run_tag, 8), ae.halfwidth_x * mr, ae.halfwidth_y * mr, ae.halfwidth_req * mr, ae.N,
        iszero(ae.tilt) ? "" : @sprintf("; grid about the beam, tilt %.4g°", rad2deg(ae.tilt))); fontsize = 15, font = :bold)
    png = joinpath(outdir, "angenergy_$(run_tag).png")
    save(png, fig)
    write_derived(outdir; kind = "angular_energy", label = @sprintf("angular energy: 1/e half-width %.3g mrad", ae.halfwidth_req * mr),
        run_id = run_tag, plot = basename(png), source = "angenergy_$(run_tag).jls",
        plot_params = Dict("halfwidth_x_mrad" => ae.halfwidth_x * mr, "halfwidth_y_mrad" => ae.halfwidth_y * mr,
            "halfwidth_req_mrad" => ae.halfwidth_req * mr, "theta_max_mrad" => maximum(abs, ae.θx) * mr,
            "n_theta" => length(ae.θx), "zsign" => ae.zsign, "oversample" => ae.oversample, "N" => ae.N,
            "W_grid" => ae.W_grid, "peak" => peak, "tilt_deg" => rad2deg(ae.tilt)),
        description = "Radiated energy per solid angle summed INCOHERENTLY over the electrons (each electron's " *
            "far-field Jackson integral along its own worldline; no observer clock, no window), on a grid of " *
            "far-field directions on the screen side. Half-widths at 1/e of the peak: along θx and θy through " *
            "the peak, and r_eq = √(area/π) of the region above 1/e. NaN = the 1/e level lies outside the grid. " *
            "tilt_deg ≠ 0: the grid is centred on the tilted beam direction (angles are about the beam).")
    return png
end
