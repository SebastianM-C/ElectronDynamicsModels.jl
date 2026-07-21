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
# puts a /data download URL on the chip. Cheap (N×4 floats ≈ 64 KB at N = 2000): always on.
function write_ic_products(xμ0, u0, dz, outdir, run_tag; γ0, λ, w₀, nb, l, chirp)
    icfile = joinpath(outdir, "ic_$(run_tag).jls")
    X = permutedims(reduce(hcat, xμ0))       # N×4 as-run start 4-positions (bunch_dz included)
    serialize(icfile, (; xμ0 = X, u0 = collect(u0), dz = collect(dz), γ0, λ, w₀,
        bunch = (; nb, l, chirp)))
    x, y = X[:, 2] ./ λ, X[:, 3] ./ λ
    dzλ = collect(dz) ./ λ
    fig = Figure(size = (1080, 480))
    ax1 = Axis(fig[1, 1]; title = "start disk, colored by bunching offset Δz",
        xlabel = "x  [λ]", ylabel = "y  [λ]", aspect = 1)
    sc = scatter!(ax1, x, y; color = dzλ, markersize = 5)
    Colorbar(fig[1, 2], sc; label = "Δz  [λ]")
    ax2 = Axis(fig[1, 3]; title = "arrival surface: lens parabola + helix spread",
        xlabel = "(ρ/w₀)²", ylabel = "Δz  [λ]")
    scatter!(ax2, (x .^ 2 .+ y .^ 2) .* (λ / w₀)^2, dzλ;
        color = atan.(y, x), colormap = :twilight, markersize = 4)
    Label(fig[0, :], @sprintf("as-run initial conditions — N = %d, γ = %g, n_b = %d, ℓ = %d",
        length(x), γ0, nb, l), fontsize = 17)
    out = joinpath(outdir, "inverse_thomson_ic_$(run_tag).png")
    save(out, fig)
    write_derived(
        outdir; kind = "ic", label = "initial conditions (as run)",
        run_id = run_tag, plot = basename(out), datafile = basename(icfile),
        plot_params = Dict("N" => length(x), "bunch_nb" => nb, "bunch_l" => l,
            "max |Δz| [λ]" => round(maximum(abs, dzλ); sigdigits = 3)),
        description = "The exact as-run start disk: transverse sunflower positions colored by " *
            "the longitudinal bunching offset Δz (the arrival-surface placement), and Δz against " *
            "(ρ/w₀)² — the lens parabola, with the ℓθ helix as the azimuthal spread. The " *
            "`ic_<id>.jls` cache stores xμ₀/u₀/Δz, so this chip re-renders without an EDM solve.",
    )
    println("saved → $(basename(out))")
    return icfile
end
