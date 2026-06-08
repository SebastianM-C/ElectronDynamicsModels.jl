module EDMPlotsExt

# CairoMakie implementations of the plotting stubs declared in src/plotting.jl. Loaded
# automatically when both ElectronDynamicsModels and CairoMakie are in the session.

using ElectronDynamicsModels
using CairoMakie

function ElectronDynamicsModels.plot_harmonic_grid(
        maps::AbstractArray{<:Number, 3}, x_grid, y_grid;
        w₀ = 1, labels, title = "",
        colormap = :jet, colorrange = harmonic_colorrange, transform = real,
        ncols = 3, panelsize = 300, outfile = nothing,
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
            row, col = (c - 1) ÷ ncols + 1, (c - 1) % ncols + 1
            gl = f[row, col] = GridLayout()
            ax = Axis(
                gl[1, 1]; width = panelsize, height = panelsize, xlabel = xlab, ylabel = ylab,
                title = "$(labels[c])  (peak $(round(maximum(abs, cmap_c); sigdigits = 3)))",
            )
            hm = heatmap!(ax, xs, ys, data; colormap, colorrange = colorrange(data))
            Colorbar(gl[1, 2], hm; width = 10, height = panelsize)
        end
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
    )
end

end # module EDMPlotsExt
