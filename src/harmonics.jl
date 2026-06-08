# Harmonic-map reduction — the time→frequency collapse that the solver and plot scripts
# each previously inlined (locating harmonic bins + per-component rfft). Pure numerics
# (FFTW only); the CairoMakie rendering of these maps lives in the plotting extension.

"""
    harmonic_bins(N_samples, δt, ω, harmonics) -> Vector{Int}

The `rfft` bin indices closest to the harmonics `n·ω` of the fundamental, for a time series
of `N_samples` samples spaced `δt` apart. `harmonics` is any iterable of integers (e.g.
`(1, 2)`). Deduplicates the locator copied across the solver/plot scripts.
"""
function harmonic_bins(N_samples::Integer, δt::Real, ω::Real, harmonics)
    freqs = rfftfreq(N_samples, 1 / δt)
    return [findmin(f -> abs(f - n * ω / 2π), freqs)[2] for n in harmonics]
end

"""
    harmonic_maps(field::NamedTuple, bins) -> Array{ComplexF64,4}   # primary: (;E,B) → 6 components
    harmonic_maps(cube::AbstractArray{<:Number,4}, bins) -> Array{ComplexF64,4}   # generic cube

Spatial maps of a field/potential at the harmonic `bins` (see [`harmonic_bins`](@ref)):
`rfft` along the time (first) axis and slice out the rows in `bins`.

The **field method** takes `(; E, B)` (each `(N_samples, 3, Nx, Ny)`) and returns
`(length(bins), 6, Nx, Ny)` — components `Eˣ Eʸ Eᶻ Bˣ Bʸ Bᶻ` (E in `1:3`, B in `4:6`). This
is the primary entry point for the field runs.

The **cube method** is the generic core: any `(N_samples, n_components, Nx, Ny)` array →
`(length(bins), n_components, Nx, Ny)`; it also serves the 4-component 4-potential `A`. Both
deduplicate the per-component `rfft` reduction copied across 5 scripts — the transform is
done one component at a time (these cubes are tens of GB at full resolution).
"""
function harmonic_maps(cube::AbstractArray{<:Number, 4}, bins::AbstractVector{<:Integer})
    out = Array{ComplexF64, 4}(undef, length(bins), size(cube, 2), size(cube, 3), size(cube, 4))
    for j in axes(cube, 2)
        Fω = rfft(cube[:, j, :, :], 1)        # one component at a time: peak ≈ cube + 1 comp, not 2×cube
        for (k, b) in enumerate(bins)
            out[k, j, :, :] = Fω[b, :, :]
        end
        Fω = nothing
        GC.gc()
    end
    return out
end

harmonic_maps(field::NamedTuple, bins::AbstractVector{<:Integer}) =
    cat(harmonic_maps(field.E, bins), harmonic_maps(field.B, bins); dims = 2)

"""
    power_spectrum(cube) -> Matrix{Float64}

Per-component frequency power spectrum of a `(N_samples, n_components, Nx, Ny)` cube, summed
over the screen: `out[ν, c] = Σ_pixels |rfft(cube[:,c,:,:], 1)[ν]|²`. Returns `(N÷2+1, n_components)`;
pair with `rfftfreq(N_samples, 1/δt)` for the frequency axis. One component at a time (memory).
"""
function power_spectrum(cube::AbstractArray{<:Number, 4})
    out = zeros(Float64, size(cube, 1) ÷ 2 + 1, size(cube, 2))
    for c in axes(cube, 2)
        Fω = rfft(cube[:, c, :, :], 1)
        out[:, c] = dropdims(sum(abs2, Fω; dims = (2, 3)); dims = (2, 3))
        Fω = nothing
        GC.gc()
    end
    return out
end

# Field method: E in columns 1:3, B in 4:6 (matches harmonic_maps' component order).
power_spectrum(field::NamedTuple) = hcat(power_spectrum(field.E), power_spectrum(field.B))
