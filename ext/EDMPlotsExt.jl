module EDMPlotsExt

# CairoMakie implementations of the plotting stubs declared in src/plotting.jl. Loaded
# automatically when both ElectronDynamicsModels and CairoMakie are in the session.

using ElectronDynamicsModels
using CairoMakie

function ElectronDynamicsModels.plot_harmonic_grid(
        maps::AbstractArray{<:Number, 3}, x_grid, y_grid;
        w₀ = 1, labels, title = "",
        colormap = :jet, colorrange = harmonic_colorrange,
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
            data = real.(maps[c, :, :])
            row, col = (c - 1) ÷ ncols + 1, (c - 1) % ncols + 1
            gl = f[row, col] = GridLayout()
            ax = Axis(
                gl[1, 1]; width = panelsize, height = panelsize, xlabel = xlab, ylabel = ylab,
                title = "$(labels[c])  (peak $(round(maximum(abs, data); sigdigits = 3)))",
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

end # module EDMPlotsExt
