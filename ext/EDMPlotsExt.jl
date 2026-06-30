module EDMPlotsExt

# CairoMakie implementations of the plotting stubs declared in src/plotting.jl. Loaded
# automatically when both ElectronDynamicsModels and CairoMakie are in the session.

using ElectronDynamicsModels
using CairoMakie

# Printf-free scientific-notation formatter (Printf isn't a dep of this extension). "2.65e-20".
function _sci(x; digits = 2)
    x == 0 && return "0"
    e = floor(Int, log10(abs(x)))
    return string(round(x / 10.0^e; digits = digits)) * "e" * string(e)
end

function ElectronDynamicsModels.plot_harmonic_grid(
        maps::AbstractArray{<:Number, 3}, x_grid, y_grid;
        w₀ = 1, labels, title = "",
        colormap = :jet, colorrange = harmonic_colorrange, transform = real,
        colorbar_offset = true, ncols = 3, panelsize = 300, outfile = nothing,
    )
    ncomp = size(maps, 1)
    length(labels) == ncomp ||
        throw(ArgumentError("labels ($(length(labels))) must match component count ($ncomp)"))
    xs = collect(x_grid) ./ w₀
    ys = collect(y_grid) ./ w₀
    xlab, ylab = w₀ == 1 ? ("x", "y") : ("x / w₀", "y / w₀")
    # Scope the LaTeX theme to this figure (no global set_theme! side effect).
    fig = with_theme(theme_latexfonts()) do
        f = Figure()
        isempty(title) || Label(f[0, :], title; fontsize = 16, font = :bold)
        for c in 1:ncomp
            cmap_c = @view maps[c, :, :]
            data = transform.(cmap_c)
            cr = colorrange(data)
            row, col = (c - 1) ÷ ncols + 1, (c - 1) % ncols + 1
            gl = f[row, col] = GridLayout()
            ax = Axis(
                gl[1, 1]; width = panelsize, height = panelsize, xlabel = xlab, ylabel = ylab,
                title = "$(labels[c])  (peak $(_sci(maximum(abs, cmap_c))))",
            )
            hm = heatmap!(ax, xs, ys, data; colormap, colorrange = cr)
            if colorbar_offset
                # Factor the shared power-of-ten into the Colorbar's native `label` so the ticks carry
                # only short mantissas (the exponent rides the Colorbar's own label and stays attached).
                # The ±peak ticks would otherwise sit exactly on the colorrange boundary, and Makie
                # drops boundary-coincident ticks at layout time on some panels — leaving a bare bar
                # showing only "0". Nudging the two end ticks a hair *inside* the range (the labels still
                # read the true extremes) makes them render reliably on every panel.
                m = max(abs(cr[1]), abs(cr[2]))
                e = m > 0 ? floor(Int, log10(m)) : 0
                sc = 10.0^e
                pad = 0.0015 * (cr[2] - cr[1])
                Colorbar(
                    gl[1, 2], hm; width = 10, height = panelsize, ticklabelsize = 10, labelsize = 11,
                    ticks = ([cr[1] + pad, 0.0, cr[2] - pad],
                        [string(round(cr[1] / sc; digits = 2)), "0", string(round(cr[2] / sc; digits = 2))]),
                    label = L"\times 10^{%$e}", labelrotation = 0,
                )
            else
                Colorbar(gl[1, 2], hm; width = 10, height = panelsize)
            end
        end
        # Breathing room between columns so each colorbar's tick labels clear the neighbouring panel.
        min(ncols, ncomp) > 1 && colgap!(f.layout, 26)
        resize_to_layout!(f)
        f
    end
    outfile === nothing || save(outfile, fig)
    return fig
end

function ElectronDynamicsModels.plot_power_spectrum(
        freqs, power_spec; ω, labels,
        colors = [:black, :dodgerblue, :seagreen, :crimson, :darkorange, :purple],
        linestyles = nothing, title = "", outfile = nothing,
    )
    xh = collect(freqs) ./ (ω / 2π)            # frequency in units of the fundamental
    yfloor = maximum(power_spec) * 1.0e-30     # log axis needs a positive floor
    fig = with_theme(theme_latexfonts()) do
        f = Figure(size = (1150, 560))
        ax = Axis(f[1, 1]; xlabel = "frequency / ω₁", ylabel = "Σ_pixels |Â_μ|²", yscale = log10, title)
        vlines!(ax, 1:floor(Int, last(xh)); color = (:gray, 0.35), linestyle = :dash)
        for c in axes(power_spec, 2)
            lines!(ax, xh, max.(power_spec[:, c], yfloor); label = labels[c],
                color = colors[(c - 1) % length(colors) + 1],
                linestyle = linestyles === nothing ? :solid : linestyles[c], linewidth = 1.8)
        end
        xlims!(ax, 0, last(xh))
        axislegend(ax; position = :rt)
        f
    end
    outfile === nothing || save(outfile, fig)
    return fig
end

function ElectronDynamicsModels.plot_phase_grid(
        maps::AbstractArray{<:Number, 3}, x_grid, y_grid;
        w₀ = 1, labels, title = "", ncols = 3, panelsize = 300, outfile = nothing,
    )
    return ElectronDynamicsModels.plot_harmonic_grid(
        maps, x_grid, y_grid; w₀, labels, title, ncols, panelsize, outfile,
        transform = angle, colormap = :phase, colorrange = _ -> (-Float64(π), Float64(π)),
        colorbar_offset = false,   # phase is O(1) over (-π, π) — no ×10ⁿ offset, keep auto ticks
    )
end

function ElectronDynamicsModels.plot_phase_with_rings(
        maps::AbstractArray{<:Number, 3}, x_grid, y_grid;
        w₀ = 1, labels, radii, tol, title = "", panelsize = 300, outfile = nothing,
    )
    ncomp = size(maps, 1)
    length(labels) == ncomp ||
        throw(ArgumentError("labels ($(length(labels))) must match component count ($ncomp)"))
    xs = collect(x_grid) ./ w₀
    ys = collect(y_grid) ./ w₀
    xlab, ylab = w₀ == 1 ? ("x", "y") : ("x / w₀", "y / w₀")
    # One colour per test radius — shared by its dashed annulus (left) and scatter series (right).
    palette = (:dodgerblue, :seagreen, :crimson, :darkorange, :purple)
    ringcolor(i) = palette[(i - 1) % length(palette) + 1]
    # Ring pixels (shared across components — same grid), sorted by azimuth, via the package helper.
    rings = map(enumerate(radii)) do (i, R)
        idxs, az = ElectronDynamicsModels.ring_pixels(x_grid, y_grid, R; tol)
        (R = R, color = ringcolor(i), idxs = idxs, az = az)
    end
    ntr = ncomp                          # every component gets the unwrapped fit ∠F ≈ slope·φ + b (Eᶻ/Bᶻ wind too)
    fits = Dict{Int, NamedTuple}()       # component => (; slope, b) vectors over radii (NaN for empty rings)
    θ = range(-Float64(π), Float64(π), 200)
    # Fit line on the wrapped ±π scatter axis: re-wrap slope·φ+b and break the polyline at 2π jumps.
    function wrapped_fit(slope, b)
        φ = collect(range(-Float64(π), Float64(π), 400))
        y = @. mod(slope * φ + b + π, 2π) - π
        for k in 2:length(y)
            abs(y[k] - y[k - 1]) > π && (y[k] = NaN)
        end
        return φ, y
    end
    fig = with_theme(theme_latexfonts()) do
        f = Figure()
        isempty(title) || Label(f[0, :], title; fontsize = 16, font = :bold)
        for c in 1:ncomp
            cmap_c = @view maps[c, :, :]
            ph = angle.(cmap_c)
            # left: phase heatmap with the test annuli as dashed R±tol circles (in w₀ units).
            axh = Axis(
                f[c, 1]; width = panelsize, height = panelsize, xlabel = xlab, ylabel = ylab,
                title = "$(labels[c])  (peak $(round(maximum(abs, cmap_c); sigdigits = 3)))",
            )
            hm = heatmap!(axh, xs, ys, ph; colormap = :phase, colorrange = (-Float64(π), Float64(π)))
            Colorbar(f[c, 2], hm; width = 10, height = panelsize, label = "∠F [rad]")
            # R±tol annuli: a white underlay for contrast on the cyclic :phase map (which already
            # cycles through the series colours), with the series-matched dashed line on top.
            for r in rings, Rb in (r.R - tol, r.R + tol)
                cx, cy = (Rb / w₀) .* cos.(θ), (Rb / w₀) .* sin.(θ)
                lines!(axh, cx, cy; color = :white, linewidth = 2.2)
                lines!(axh, cx, cy; color = r.color, linestyle = :dash, linewidth = 1.0)
            end
            # right: ∠F vs azimuth on each ring, colour-matched to its annulus; each component
            # also gets the unwrapped linear-winding fit overlaid (slope ≈ ℓ).
            axr = Axis(
                f[c, 3]; width = panelsize, height = panelsize,
                xlabel = "azimuth φ", ylabel = "∠F", limits = (-π, π, -π, π),
            )
            sl = Float64[]; bb = Float64[]
            for r in rings
                if isempty(r.idxs)
                    c ≤ ntr && (push!(sl, NaN); push!(bb, NaN))
                    continue
                end
                scatter!(axr, r.az, ph[r.idxs]; color = r.color, markersize = 5,
                    label = "R/w₀=$(round(r.R / w₀; sigdigits = 2))")
                if c ≤ ntr
                    fit = ElectronDynamicsModels.phase_winding_fit(
                        r.az, ph[r.idxs]; weights = abs.(cmap_c[r.idxs]),
                    )
                    push!(sl, fit.slope); push!(bb, fit.intercept)
                    φf, yf = wrapped_fit(fit.slope, fit.intercept)
                    lines!(axr, φf, yf; color = r.color, linewidth = 1.6)
                end
            end
            c ≤ ntr && (fits[c] = (slope = sl, b = bb))
            c == 1 && axislegend(axr; labelsize = 8, position = :lt)
        end
        resize_to_layout!(f)
        f
    end
    outfile === nothing || save(outfile, fig)
    return (; fig, fits)
end

function ElectronDynamicsModels.plot_phase_polar(
        maps::AbstractArray{<:Number, 3}, x_grid, y_grid;
        w₀ = 1, labels, radii, tol, title = "", panelsize = 300, outfile = nothing,
    )
    ncomp = size(maps, 1)
    length(labels) == ncomp ||
        throw(ArgumentError("labels ($(length(labels))) must match component count ($ncomp)"))
    palette = (:dodgerblue, :seagreen, :crimson, :darkorange, :purple)
    ringcolor(i) = palette[(i - 1) % length(palette) + 1]
    rings = map(enumerate(radii)) do (i, R)
        idxs, az = ElectronDynamicsModels.ring_pixels(x_grid, y_grid, R; tol)
        (R = R, color = ringcolor(i), idxs = idxs, az = az)
    end
    fig = with_theme(theme_latexfonts()) do
        f = Figure()
        isempty(title) || Label(f[0, :], title; fontsize = 16, font = :bold)
        handles = Any[]; leglabels = String[]
        for c in 1:ncomp
            ph = angle.(@view maps[c, :, :])
            # angular = azimuth φ; radial = ∠F shifted to [0, 2π) so it stays non-negative.
            ax = PolarAxis(f[1, c]; width = panelsize, height = panelsize, title = string(labels[c]))
            for r in rings
                isempty(r.idxs) && continue
                sc = scatter!(ax, r.az, ph[r.idxs] .+ Float64(π); color = r.color, markersize = 5)
                c == 1 && (push!(handles, sc); push!(leglabels, "R/w₀=$(round(r.R / w₀; sigdigits = 2))"))
            end
        end
        # PolarAxis has no `axislegend`; build one from the first panel's ring handles.
        isempty(handles) || Legend(f[1, ncomp + 1], handles, leglabels; labelsize = 8)
        resize_to_layout!(f)
        f
    end
    outfile === nothing || save(outfile, fig)
    return fig
end

end # module EDMPlotsExt
