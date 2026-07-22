# Trajectory-side products shared between the live solver (inverse_thomson_scattering.jl)
# and the backfill path (gammatau_backfill.jl): the γ(τ)/γ₀ trace reduction and the as-run
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
                drain[e] = 1 - tr(τf)[2][1] / (γ0 * c)
            end
        end
    end
    foreach(wait, tasks)
    return reduce(+, sums) ./ length(trajs),
        reduce((a, b) -> min.(a, b), los), reduce((a, b) -> max.(a, b), his), drain
end

# As-run initial conditions: cache + chip. The disk and its Δz offsets are deterministic
# from [config], but the cache pins the EXACT as-run xμ₀/u₀ (reconstruction drift becomes
# visible instead of silent) and lets the IC chip re-render anywhere without an EDM solve —
# the same publish-autonomy contract as the γ(τ) trace (both are enumerated in the .reduced
# marker at reduce time). `datafile` on the sidecar ships the cache to the storagebox and
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
