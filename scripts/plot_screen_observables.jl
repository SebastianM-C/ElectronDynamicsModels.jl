# Screen-observable maps for a field-accumulation run (the field .jls from
# thomson_scattering.jl): energy fluence (∫Sᶻ dt) and angular-momentum flux
# (∫dLz/dA dt) on the screen, from `screen_observables`. Two rows — the total
# field (E,B) and the radiation-only field (E_rad,B_rad) — so you can see the
# near-field's (small, far-field) contribution to the totals. Prints the scalar
# totals (energy, Lz, Lz/U).
#
# Loads the (; E, B, E_rad, B_rad) field .jls and reconstructs the Thomson screen;
# samples_per_period comes from the parent run TOML (via manifest.jl). Emits an
# `observables` derived sidecar for the dashboard.
#
#   julia --project=scripts scripts/plot_screen_observables.jl [field.jls]

using ElectronDynamicsModels
using Serialization
using Printf
using CairoMakie

# Thomson screen constants (identical to scripts/thomson_scattering.jl).
const c = 137.03599908330932
const ω = 0.057
const λ = 2π * c / ω
const w₀ = 75λ
const Rmax = 3.25w₀
const Z = 2.0e5λ
const τ = 150 / ω
const τi = -8τ
const ε₀ = 1 / (4π)          # atomic units (4πε₀ = 1); map structure is ε₀-independent

const datafile = length(ARGS) ≥ 1 ? ARGS[1] : error("pass the field .jls as ARG 1")
const stem = replace(datafile, r"\.jls$" => "")

include(joinpath(@__DIR__, "manifest.jl"))
const dir = dirname(abspath(datafile))
const parent = find_parent_manifest(dir, basename(datafile))
parent === nothing && error("no run_*.toml in $dir binds $(basename(datafile)) — needed for samples_per_period")
const spp = spp_from_manifest(parent[2])

function thomson_screen(Nx, N_samples)
    δt = 2π / ω / spp
    x⁰_start = c * τi + hypot(Z, 25w₀ + Rmax)
    x⁰ = range(start = x⁰_start, step = c * δt, length = N_samples)
    return ObserverScreen(LinRange(-25w₀, 25w₀, Nx), LinRange(-25w₀, 25w₀, Nx), Z, x⁰; c)
end

println("loading $datafile …")
fld = deserialize(datafile)
N_samples, _, Nx, Ny = size(fld.E)
screen = thomson_screen(Nx, N_samples)
dt = step(screen.x⁰_samples) / c

println("computing screen_observables (total + radiation) …")
obs_tot = screen_observables(fld, screen; ε₀)
obs_rad = screen_observables((; E = fld.E_rad, B = fld.B_rad), screen; ε₀)

# Per-pixel time-integrated maps: energy fluence ∫Sᶻ dt, angular-momentum flux ∫dLz dt.
energy_fluence(o) = dropdims(sum(@view(o.S[:, 3, :, :]); dims = 1); dims = 1) .* dt
lz_fluence(o) = dropdims(sum(o.Lz_density; dims = 1); dims = 1) .* dt

@printf("\n%-12s %14s %14s %14s\n", "", "energy_total", "Lz_total", "Lz/U")
for (name, o) in (("total", obs_tot), ("radiation", obs_rad))
    @printf("%-12s %14.4e %14.4e %14.4e\n", name, o.energy_total, o.Lz_total,
        o.energy_total == 0 ? NaN : o.Lz_total / o.energy_total)
end

# ── Figure: rows = total / radiation, cols = energy fluence / Lz fluence ──
fig = Figure()
Label(fig[0, :], "Screen observables — $(basename(stem))", fontsize = 14, font = :bold)
rows = (("total field", obs_tot), ("radiation field", obs_rad))
cols = (("energy fluence  ∫Sᶻ dt", energy_fluence, :viridis),
    ("L_z flux  ∫(dLz/dA) dt", lz_fluence, :seismic))
for (r, (rname, o)) in enumerate(rows), (cc, (cname, mapf, cmap)) in enumerate(cols)
    data = mapf(o)
    gl = fig[r, cc] = GridLayout()
    ax = Axis(gl[1, 1]; width = 320, height = 320, xlabel = "x [a.u.]", ylabel = "y [a.u.]",
        title = "$rname — $cname")
    hm = if cmap == :viridis
        heatmap!(ax, collect(screen.x_grid), collect(screen.y_grid), data; colormap = cmap)
    else
        cr = maximum(abs, data)
        heatmap!(ax, collect(screen.x_grid), collect(screen.y_grid), data;
            colormap = cmap, colorrange = cr > 0 ? (-cr, cr) : (-1.0, 1.0))
    end
    Colorbar(gl[1, 2], hm, width = 12, height = 320)
end
resize_to_layout!(fig)
out = stem * "_observables.png"
save(out, fig)
println("saved → $out")

# ── Derived sidecar for the dashboard ──
let pid = parent[1]
    if pid === nothing
        @warn "parent run manifest for $(basename(datafile)) has no run_id; skipping sidecar"
    else
        write_derived(dir; kind = "observables",
            label = @sprintf("screen observables (U=%.2e, Lz=%.2e)", obs_tot.energy_total, obs_tot.Lz_total),
            run_id = pid, plot = basename(out), source = basename(datafile))
        println("derived sidecar → observables (parent run $pid)")
    end
end
