# Verify the Lorenz gauge condition  ∂_μ A^μ = 0  on a small patch of the screen.
#
# The Liénard-Wiechert 4-potential satisfies the Lorenz condition identically, so
# this is a self-consistency check on the *computed* field (retarded-time solve +
# observer-time binning):  if the pipeline produces a valid 4-potential, the four
# terms below must cancel to within finite-difference + interpolation error.
#
# With x⁰ = c t, the condition in the rfft-of-time domain (per frequency bin ν) is
#
#     ∂A⁰/∂x⁰ + ∂Aˣ/∂x + ∂Aʸ/∂y + ∂Aᶻ/∂z = 0
#     └ i·2π·freq[ν]/c · A⁰_ω ┘   └──── finite differences in x, y, z ────┘
#
# Scheme:
#   • time term  — exact spectral derivative ∂/∂x⁰ = (1/c)∂/∂t  ↔  i·2π·freq/c.
#   • x, y       — 2nd-order CENTERED differences on the production pixel grid
#                  (Δx = 50w₀/399 ≈ 9.4λ).  The 2ω rings turn ~0.4 rad/pixel, so a
#                  one-sided difference would carry ~20% error; centered drops it
#                  to ~(kΔ)²/6 ≈ 1% (fund) / few-% (2ω).
#   • z          — 2nd-order CENTERED difference from THREE screens at z and z±δz
#                  (δz = λ/50).  k_z·δz ≈ 0.13 → ~0.3% (fund) / ~1% (2ω) error,
#                  matched to the transverse accuracy.
#
# Only the box region (+1 pixel each side for the centered stencils) is recomputed,
# on the EXACT production observer-time grid (same x⁰_start, δt, N_samples), so the
# harmonic bins line up with the stored analysis.  Cost ≈ 1% of the production run.
# A cross-check against the cached production harmonic slices confirms the recompute
# reproduces the stored field.
#
#   julia --project=scripts scripts/verify_lorenz_gauge.jl

using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using OrdinaryDiffEqNonlinearSolve
using SciMLBase
using StaticArrays
using SymbolicIndexingInterface
using LinearAlgebra
using AcceleratedKernels
using AMDGPU
using FFTW
using Serialization
using Statistics
using Printf
using CairoMakie

const backend = AMDGPU.ROCBackend()
const c = 137.03599908330932

# ── Physics / laser (identical to production_400.jl) ──
const ω = 0.057
const τ = 150 / ω
const λ = 2π * c / ω
const w₀ = 75λ
const Rmax = 3.25w₀
const a₀ = 0.1

@named world = Worldline(:τ, :atomic)
@named laser = LaguerreGaussLaser(;
    wavelength = λ, a0 = a₀, beam_waist = w₀, radial_index = 2, azimuthal_index = -2,
    world, temporal_profile = :gaussian, temporal_width = τ, focus_position = 0.0,
    polarization = :circular,
)
@named elec = ClassicalElectron(; laser)
const sys = mtkcompile(elec)

const τi = -8τ
const τf = 8τ
const prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, [sys.x => [τi * c, 0.0, 0.0, 0.0], sys.u => [c, 0.0, 0.0, 0.0]], (τi, τf);
    u0_constructor = SVector{8}, fully_determined = true,
)
const set_x = setsym_oop(prob, [Initial(sys.x); Initial(sys.u)])

const ϕ = (1 + √5) / 2
radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)
function sunflower(n, α)
    pts = Vector{Vector{Float64}}()
    stride = 2π / ϕ^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = radius(k, n, b)
        push!(pts, [r * cos(k * stride), r * sin(k * stride)])
    end
    return pts
end
abserr(a₀) = (amp = log10(a₀); 10^(-amp^2 / 27 + 32amp / 27 - 220 / 27))

function build_trajs(N)
    xμ = [SVector{4, Float64}(τi * c, r[1], r[2], 0.0) for r in Rmax * sunflower(N, 2)]
    function prob_func(prob, ctx)
        u0_new, p = set_x(prob, SVector{8}(xμ[ctx.sim_id]..., c, 0.0, 0.0, 0.0))
        return remake(prob; u0 = u0_new, p)
    end
    ens = EnsembleProblem(prob; prob_func, safetycopy = false)
    sol = solve(ens, Vern9(), EnsembleThreads(); reltol = 1.0e-12, abstol = abserr(a₀), trajectories = N)
    return trajectory_interpolants(sol)
end

# ── Observer sampling (identical to production_400.jl) ──
const Z = 2.0e5λ
const samples_per_period = 16
const δt = 2π / ω / samples_per_period
const N_samples = 2000                       # quick run; 8000 matches production & enables cache cross-check
const x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)
const x⁰_samples = range(start = x⁰_start, step = c * δt, length = N_samples)

const Nx_full = 400
const xfull = LinRange(-25w₀, 25w₀, Nx_full)  # full production transverse grid
const Δx = step(xfull)                        # = 50w₀/399 ≈ 9.4λ
const Δy = Δx

# ── Verification patch (box matches scripts/lorenz_box_placement.jl) ──
const CX, CY = 126, 299                       # box centre pixel on the 400² grid
const NB = 10                                 # half-width → 21×21 evaluation pixels
const data_i = (CX - NB - 1):(CX + NB + 1)    # +1 each side for the centered stencils
const data_j = (CY - NB - 1):(CY + NB + 1)
const x_sub = xfull[data_i]   # LinRange slice (isbits — required by the GPU kernel)
const y_sub = xfull[data_j]

# ── Run knobs ──
const N = 2000                                # quick run; 10000 matches production
const δz = λ / 50                             # centered z-difference half-step (3 screens)
const n_substeps = 1                          # match production RK4
const harmonics = 1:6
const map_harmonics = (1, 2)                  # full term-decomposition maps for these

# interior eval-pixel coordinates (full-grid pixels CX-NB..CX+NB), in w₀ units
const xc_box = collect(xfull[(CX - NB):(CX + NB)]) ./ w₀
const yc_box = collect(xfull[(CY - NB):(CY + NB)]) ./ w₀

const TERM_LABELS = ["∂ₓ₀A⁰", "∂ₓAˣ", "∂_yAʸ", "∂_zAᶻ"]
const TERM_COLORS = [:black, :dodgerblue, :seagreen, :crimson]

# Full term-decomposition maps over the box for one harmonic: the four terms and
# their sum share a symmetric scale (so |Σ|≪|term| reads as a near-white panel);
# the 6th panel is the relative residual.
function plot_terms(res)
    mats = (res.TT, res.TX, res.TY, res.TZ, res.DIV)
    cr = maximum(M -> maximum(abs ∘ real, M), mats)
    fig = Figure()
    Label(fig[0, :], @sprintf("Lorenz terms — %gω₁   (Re; four terms+Σ share scale, Σ→0 ⇒ near white)", res.f),
        fontsize = 15, font = :bold)
    titles = [TERM_LABELS..., "Σ = ∂_μA^μ"]
    for (k, M) in enumerate(mats)
        r = cld(k, 3); col = (k - 1) % 3 + 1
        gl = fig[r, col] = GridLayout()
        ax = Axis(gl[1, 1], width = 220, height = 220, title = titles[k], xlabel = "x/w₀", ylabel = "y/w₀")
        hm = heatmap!(ax, xc_box, yc_box, real.(M), colorrange = (-cr, cr), colormap = :seismic)
        Colorbar(gl[1, 2], hm, width = 10, height = 220)
    end
    gl = fig[2, 3] = GridLayout()
    ax = Axis(gl[1, 1], width = 220, height = 220, title = "|Σ| / k‖A‖", xlabel = "x/w₀", ylabel = "y/w₀")
    hm = heatmap!(ax, xc_box, yc_box, res.rel, colorrange = (0, maximum(res.rel)), colormap = :thermal)
    Colorbar(gl[1, 2], hm, width = 10, height = 220)
    resize_to_layout!(fig)
    out = @sprintf("lorenz_gauge_terms_n%d.png", res.n)
    save(out, fig)
    println("saved → ", out)
end

# Summary across harmonics: term magnitudes + residual (left), relative residual (right).
function plot_summary(results)
    ns = [r.n for r in results]
    fig = Figure(size = (1000, 430))
    ax1 = Axis(fig[1, 1], yscale = log10, xlabel = "harmonic n", ylabel = "mean |term|  (box)",
        title = "term magnitudes vs harmonic", xticks = ns)
    for k in 1:4
        ys = [r.tmag[k] for r in results]
        lines!(ax1, ns, ys, color = TERM_COLORS[k], label = TERM_LABELS[k])
        scatter!(ax1, ns, ys, color = TERM_COLORS[k])
    end
    dy = [r.divmed for r in results]
    lines!(ax1, ns, dy, color = :orange, linestyle = :dash, label = "|Σ| median")
    scatter!(ax1, ns, dy, color = :orange, marker = :diamond)
    axislegend(ax1, position = :rt, labelsize = 10)

    ax2 = Axis(fig[1, 2], yscale = log10, xlabel = "harmonic n", ylabel = "|Σ| / (k‖A‖)",
        title = "relative Lorenz residual", xticks = ns)
    lines!(ax2, ns, [r.relmed for r in results], color = :purple, label = "median")
    scatter!(ax2, ns, [r.relmed for r in results], color = :purple)
    lines!(ax2, ns, [r.relmax for r in results], color = :purple, linestyle = :dash, label = "max")
    scatter!(ax2, ns, [r.relmax for r in results], color = :purple, marker = :utriangle)
    axislegend(ax2, position = :rb, labelsize = 10)
    save("lorenz_gauge_summary.png", fig)
    println("saved → lorenz_gauge_summary.png")
end

# Relative-residual spatial map for every harmonic, in one grid.
function plot_relmaps(results)
    fig = Figure()
    Label(fig[0, :], "Relative Lorenz residual  |∂_μA^μ| / (k‖A‖)  per harmonic", fontsize = 15, font = :bold)
    for (k, res) in enumerate(results)
        r = cld(k, 3); col = (k - 1) % 3 + 1
        gl = fig[r, col] = GridLayout()
        ax = Axis(gl[1, 1], width = 200, height = 200, xlabel = "x/w₀", ylabel = "y/w₀",
            title = @sprintf("%gω₁  (med %.1e)", res.f, res.relmed))
        hi = max(maximum(res.rel) * 1e-9, quantile(vec(res.rel), 0.99))
        hm = heatmap!(ax, xc_box, yc_box, res.rel, colorrange = (0, hi), colormap = :thermal)
        Colorbar(gl[1, 2], hm, width = 10, height = 200)
    end
    resize_to_layout!(fig)
    save("lorenz_gauge_relmap.png", fig)
    println("saved → lorenz_gauge_relmap.png")
end

# Compute the three time-domain screens (z, z±δz), or load them from the cache.
# Caching the GPU outputs lets the (cheap) downstream analysis be re-run instantly.
const screens_cache = "lorenz_screens_N$(N)_Ns$(N_samples)_box$(CX)-$(CY)-$(NB).jls"

function get_screens()
    if isfile(screens_cache)
        d = deserialize(screens_cache)
        if d.N == N && d.N_samples == N_samples && d.samples_per_period == samples_per_period &&
           d.CX == CX && d.CY == CY && d.NB == NB && d.δz == δz
            println("loaded cached screens ← $screens_cache")
            return d.A0, d.Am, d.Ap
        end
        println("cache params differ from current run — recomputing")
    end
    println("solving $N trajectories…")
    @time trajs = build_trajs(N)
    # Three screens at z and z±δz (same transverse sub-grid and observer-time grid).
    screen0 = ObserverScreen(x_sub, y_sub, Z, x⁰_samples)
    screenm = ObserverScreen(x_sub, y_sub, Z - δz, x⁰_samples)
    screenp = ObserverScreen(x_sub, y_sub, Z + δz, x⁰_samples)
    println("accumulating potential on 3 screens…")
    @time A0 = accumulate_potential(trajs, screen0, GPUKernelRK4(), backend; n_substeps)
    @time Am = accumulate_potential(trajs, screenm, GPUKernelRK4(), backend; n_substeps)
    @time Ap = accumulate_potential(trajs, screenp, GPUKernelRK4(), backend; n_substeps)
    serialize(screens_cache, (; N, N_samples, samples_per_period, CX, CY, NB, δz, A0, Am, Ap))
    println("cached screens → $screens_cache")
    return A0, Am, Ap
end

# ── Frequency-domain check (centered FD on the rfft-of-time field) ──
function freq_domain_check(A0, Am, Ap)
    A0ω = rfft(A0, 1)
    Amω = rfft(Am, 1)
    Apω = rfft(Ap, 1)
    freqs = rfftfreq(N_samples, 1 / δt)

    # Cross-check vs cached production harmonic slices — only matches at production size.
    cachefile = "A_rk4_400_N10000_Ns8000_spp16_hslices.jls"   # μ=3 (Aʸ)
    if N == 10_000 && N_samples == 8000 && isfile(cachefile)
        cache = deserialize(cachefile)
        for n in (1, 2)
            idx = argmin(abs.(freqs .- n * ω / 2π))
            rel = norm(A0ω[idx, 3, :, :] .- cache.fields[n, data_i, data_j]) / norm(cache.fields[n, data_i, data_j])
            @printf("  cross-check Aʸ %dω vs cache: rel diff = %.2e\n", n, rel)
        end
    else
        println("  (cross-check skipped — needs N=10000, N_samples=8000 to match the cache)")
    end

    ni, nj = size(A0, 3), size(A0, 4)
    interior_i = 2:(ni - 1)
    interior_j = 2:(nj - 1)
    Mi, Mj = length(interior_i), length(interior_j)
    results = NamedTuple[]
    println("\n[frequency domain, centered FD]   rel = |Σ| / (k_ν‖A_ω‖),  k_ν = 2π·f_ν/c")
    println("n   f/ω₁    |div| med      |div| max      rel med   rel max   ⟨|∂ₓ₀A⁰|⟩    ⟨|∂ₓAˣ|⟩    ⟨|∂_yAʸ|⟩    ⟨|∂_zAᶻ|⟩")
    for n in harmonics
        idx = argmin(abs.(freqs .- n * ω / 2π))
        kfac = im * 2π * freqs[idx] / c
        TT = Matrix{ComplexF64}(undef, Mi, Mj)
        TX = similar(TT); TY = similar(TT); TZ = similar(TT); DIV = similar(TT)
        ANORM = Matrix{Float64}(undef, Mi, Mj)
        for (jj, j) in enumerate(interior_j), (ii, i) in enumerate(interior_i)
            tt = kfac * A0ω[idx, 1, i, j]
            tx = (A0ω[idx, 2, i + 1, j] - A0ω[idx, 2, i - 1, j]) / (2Δx)
            ty = (A0ω[idx, 3, i, j + 1] - A0ω[idx, 3, i, j - 1]) / (2Δy)
            tz = (Apω[idx, 4, i, j] - Amω[idx, 4, i, j]) / (2δz)
            TT[ii, jj] = tt; TX[ii, jj] = tx; TY[ii, jj] = ty; TZ[ii, jj] = tz
            DIV[ii, jj] = tt + tx + ty + tz
            ANORM[ii, jj] = sqrt(abs2(A0ω[idx, 1, i, j]) + abs2(A0ω[idx, 2, i, j]) +
                                 abs2(A0ω[idx, 3, i, j]) + abs2(A0ω[idx, 4, i, j]))
        end
        rel = abs.(DIV) ./ (abs(kfac) .* ANORM)   # |Σ| / (k_ν‖A_ω‖)
        tmag = [mean(abs, TT), mean(abs, TX), mean(abs, TY), mean(abs, TZ)]
        f = freqs[idx] / (ω / 2π)
        push!(results, (; n, f, TT, TX, TY, TZ, DIV, rel, tmag,
            divmed = median(abs.(DIV)), relmed = median(rel), relmax = maximum(rel)))
        @printf("%-3d %5.3f  %.3e   %.3e   %.2e  %.2e   %.3e   %.3e   %.3e   %.3e\n",
            n, f, median(abs.(DIV)), maximum(abs.(DIV)), median(rel), maximum(rel), tmag...)
    end
    plot_summary(results)
    plot_relmaps(results)
    for res in results
        res.n in map_harmonics && plot_terms(res)
    end
    return results
end

# ── Time-domain check (forward FD on the raw, pre-FFT field) ──
# Lorenz holds pointwise in spacetime, so ∂_μA^μ(t,x,y,z) ≈ 0 at every sample.
# Forward differences as requested; the time term uses Δx⁰ = c·δt (the x⁰=ct sample
# step) — NOT δt — so ∂/∂x⁰ is consistent with the spatial terms.  Compared against
# the field-derivative scale  k·‖A‖ = (ω/c)·√(A⁰²+Aˣ²+Aʸ²+Aᶻ²)  and, as in the
# spectral check, against max|individual term|.
function plot_time_trace(TT, TX, TY, TZ, DIV, SCALE, ATRANS, ts)
    cc = (size(DIV, 2) + 1) ÷ 2
    cd = (size(DIV, 3) + 1) ÷ 2
    xax = (x⁰_samples[ts] .- x⁰_start) ./ λ
    # floor from the term/residual scale (NOT k‖A‖, which is ~8 orders larger) so the
    # term lines are visible; the wide gap up to the dashed k‖A‖ line is the point.
    flr = maximum(a -> maximum(abs, @view a[:, cc, cd]), (TT, TX, TY, TZ, DIV)) * 1e-4
    fig = Figure(size = (1050, 460))
    ax1 = Axis(fig[1, 1], yscale = log10, xlabel = "x⁰ − x⁰_start  [λ]", ylabel = "magnitude",
        title = "term magnitudes, residual & k‖A‖ at box centre")
    lines!(ax1, xax, max.(abs.(TT[:, cc, cd]), flr), color = TERM_COLORS[1], label = TERM_LABELS[1])
    lines!(ax1, xax, max.(abs.(TX[:, cc, cd]), flr), color = TERM_COLORS[2], label = TERM_LABELS[2])
    lines!(ax1, xax, max.(abs.(TY[:, cc, cd]), flr), color = TERM_COLORS[3], label = TERM_LABELS[3])
    lines!(ax1, xax, max.(abs.(TZ[:, cc, cd]), flr), color = TERM_COLORS[4], label = TERM_LABELS[4])
    lines!(ax1, xax, max.(abs.(DIV[:, cc, cd]), flr), color = :orange, linewidth = 2, label = "|Σ|")
    lines!(ax1, xax, max.(SCALE[:, cc, cd], flr), color = :purple, linestyle = :dash, label = "k‖A‖")
    axislegend(ax1, position = :rt, labelsize = 9)

    # zoom: signed terms summing to ~0 over a few periods near the transverse peak
    tpk = argmax(@view ATRANS[:, cc, cd])
    w = max(1, tpk - 3samples_per_period):min(size(DIV, 1), tpk + 3samples_per_period)
    ax2 = Axis(fig[1, 2], xlabel = "x⁰ − x⁰_start  [λ]", ylabel = "term (signed)",
        title = "signed terms cancel (zoom near pulse peak)")
    for (k, M) in enumerate((TT, TX, TY, TZ))
        lines!(ax2, xax[w], M[w, cc, cd], color = TERM_COLORS[k], label = TERM_LABELS[k])
    end
    lines!(ax2, xax[w], DIV[w, cc, cd], color = :orange, linewidth = 2, label = "Σ = ∂_μA^μ")
    axislegend(ax2, position = :rt, labelsize = 9)
    save("lorenz_time_trace.png", fig)
    println("saved → lorenz_time_trace.png")
end

function plot_time_map(DIV, SCALE, rel_scale, ATRANS, ts)
    energy = [sum(@view ATRANS[t, :, :]) for t in axes(ATRANS, 1)]
    tpk = argmax(energy)
    xc, yc = xc_box, yc_box
    fig = Figure()
    Label(fig[0, :], @sprintf("time-domain Lorenz @ peak sample  (x⁰−start = %.1fλ)",
            (x⁰_samples[ts[tpk]] - x⁰_start) / λ), fontsize = 15, font = :bold)
    crd = maximum(abs, @view DIV[tpk, :, :])
    panels = ((real.(DIV[tpk, :, :]), "Σ = ∂_μA^μ", (-crd, crd), :seismic),
        (SCALE[tpk, :, :], "k‖A‖", (0, maximum(@view SCALE[tpk, :, :])), :viridis),
        (rel_scale[tpk, :, :], "|Σ| / k‖A‖", (0, maximum(@view rel_scale[tpk, :, :])), :thermal))
    for (k, (M, t, cr, cm)) in enumerate(panels)
        gl = fig[1, k] = GridLayout()
        ax = Axis(gl[1, 1], width = 230, height = 230, title = t, xlabel = "x/w₀", ylabel = "y/w₀")
        hm = heatmap!(ax, xc, yc, M, colorrange = cr, colormap = cm)
        Colorbar(gl[1, 2], hm, width = 10, height = 230)
    end
    resize_to_layout!(fig)
    save("lorenz_time_map.png", fig)
    println("saved → lorenz_time_map.png")
end

function time_domain_check(A0, Am, Ap)
    Δx⁰ = c * δt                        # x⁰=ct sample step  (= step(x⁰_samples))
    kf = ω / c                          # fundamental wavenumber for the k‖A‖ scale
    nt, _, ni, nj = size(A0)
    interior_i = 2:(ni - 1)
    interior_j = 2:(nj - 1)
    ts = 1:(nt - 1)                     # forward time difference needs t+1
    Mt, Mi, Mj = length(ts), length(interior_i), length(interior_j)
    TT = Array{Float64, 3}(undef, Mt, Mi, Mj)
    TX = similar(TT); TY = similar(TT); TZ = similar(TT)
    SCALE = similar(TT); ATRANS = similar(TT)
    @inbounds for (jj, j) in enumerate(interior_j), (ii, i) in enumerate(interior_i), (tk, t) in enumerate(ts)
        a0 = A0[t, 1, i, j]; ax = A0[t, 2, i, j]; ay = A0[t, 3, i, j]; az = A0[t, 4, i, j]
        TT[tk, ii, jj] = (A0[t + 1, 1, i, j] - a0) / Δx⁰
        TX[tk, ii, jj] = (A0[t, 2, i + 1, j] - ax) / Δx
        TY[tk, ii, jj] = (A0[t, 3, i, j + 1] - ay) / Δy
        TZ[tk, ii, jj] = (Ap[t, 4, i, j] - az) / δz
        SCALE[tk, ii, jj] = kf * sqrt(a0^2 + ax^2 + ay^2 + az^2)
        ATRANS[tk, ii, jj] = hypot(ax, ay)
    end
    DIV = TT .+ TX .+ TY .+ TZ
    maxterm = max.(abs.(TT), abs.(TX), abs.(TY), abs.(TZ))
    rel_scale = abs.(DIV) ./ SCALE
    rel_term = abs.(DIV) ./ maxterm
    mask = ATRANS .> 0.1 * maximum(ATRANS)   # focus on the pulse (transverse field active)

    println("\n[time domain, forward FD — pre-FFT, pointwise]")
    @printf("  active samples (|A_⊥| > 10%% of peak): %d of %d\n", count(mask), length(mask))
    @printf("  |div|          median=%.3e   max=%.3e   (active)\n", median(abs.(DIV)[mask]), maximum(abs.(DIV)[mask]))
    @printf("  |div| / k‖A‖   median=%.3e   max=%.3e   (active)\n", median(rel_scale[mask]), maximum(rel_scale[mask]))
    @printf("  |div| / max|t| median=%.3e   max=%.3e   (active)\n", median(rel_term[mask]), maximum(rel_term[mask]))
    @printf("  ⟨|∂ₓ₀A⁰|⟩=%.3e ⟨|∂ₓAˣ|⟩=%.3e ⟨|∂_yAʸ|⟩=%.3e ⟨|∂_zAᶻ|⟩=%.3e   ⟨k‖A‖⟩=%.3e (active)\n",
        mean(abs.(TT)[mask]), mean(abs.(TX)[mask]), mean(abs.(TY)[mask]), mean(abs.(TZ)[mask]), mean(SCALE[mask]))

    plot_time_trace(TT, TX, TY, TZ, DIV, SCALE, ATRANS, ts)
    plot_time_map(DIV, SCALE, rel_scale, ATRANS, ts)
    return (; t0 = TT, sdiv = TX .+ TY .+ TZ, div = DIV, atrans = ATRANS, ts)
end

# The two halves of ∂_μA^μ=0 as heatmaps: the 0th-component term ∂ₓ₀A⁰ must equal
# MINUS the spatial divergence ∂ₓAˣ+∂_yAʸ+∂_zAᶻ, so the first two maps should look
# identical and the residual Σ near-white.  Top row = frequency domain (fundamental,
# Re); bottom row = time domain at the pulse-peak sample.  Each row shares one scale.
function plot_balance(freq_results, td)
    res = freq_results[1]
    f_t0 = real.(res.TT)
    f_ndiv = real.(-(res.TX .+ res.TY .+ res.TZ))   # −∇·A⃗  (should match f_t0)
    f_sum = real.(res.DIV)

    energy = [sum(@view td.atrans[t, :, :]) for t in axes(td.atrans, 1)]
    tpk = argmax(energy)
    t_t0 = td.t0[tpk, :, :]
    t_ndiv = -td.sdiv[tpk, :, :]
    t_sum = td.div[tpk, :, :]

    tpk_x0 = (x⁰_samples[td.ts[tpk]] - x⁰_start) / λ
    titles = ("∂ₓ₀A⁰  (0th-component term)", "−(∂ₓAˣ+∂_yAʸ+∂_zAᶻ)  (spatial divergence)", "Σ = ∂_μA^μ  (residual)")
    tags = (@sprintf("%gω₁ (Re)", res.f), @sprintf("time, peak (x⁰−start=%.0fλ)", tpk_x0))
    rows = ((f_t0, f_ndiv, f_sum), (t_t0, t_ndiv, t_sum))

    fig = Figure()
    Label(fig[0, :], "Lorenz balance:  0th-component term  vs  −(spatial divergence)  ⇒  identical maps, Σ→0",
        fontsize = 15, font = :bold)
    for (ri, (m1, m2, m3)) in enumerate(rows)
        cr = maximum(M -> maximum(abs, M), (m1, m2))   # shared scale from the two compared maps
        for (ci, M) in enumerate((m1, m2, m3))
            gl = fig[ri, ci] = GridLayout()
            ax = Axis(gl[1, 1], width = 215, height = 215, xlabel = "x/w₀", ylabel = "y/w₀",
                title = @sprintf("%s\n[%s]", titles[ci], tags[ri]))
            hm = heatmap!(ax, xc_box, yc_box, M, colorrange = (-cr, cr), colormap = :seismic)
            Colorbar(gl[1, 2], hm, width = 10, height = 215)
        end
    end
    resize_to_layout!(fig)
    save("lorenz_balance.png", fig)
    println("saved → lorenz_balance.png")
end

function main()
    @printf("Lorenz-gauge check — box (%d,%d) ±%d px,  δz=λ/%g,  N=%d,  N_samples=%d\n",
        CX, CY, NB, λ / δz, N, N_samples)
    @printf("box centre x=%.2f w₀  y=%.2f w₀;  Δx=Δy=%.3fλ\n", xfull[CX] / w₀, xfull[CY] / w₀, Δx / λ)

    A0, Am, Ap = get_screens()
    freq_results = freq_domain_check(A0, Am, Ap)
    td = time_domain_check(A0, Am, Ap)
    plot_balance(freq_results, td)

    println("\nrel ≪ 1 ⇒ Lorenz holds (limited by FD truncation + interpolation/binning noise).")
    return nothing
end

main()
