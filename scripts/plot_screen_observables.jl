# Screen-observable maps for a field-accumulation run (the field .jls from
# thomson_scattering.jl), via `screen_observables`. Total field (E,B) and the
# radiation-only field (E_rad,B_rad) get *separate* figures, each with:
#   - energy fluence   ∫Sᶻ dt        (time-integrated — "what got radiated where")
#   - L_z flux         ∫(dLz/dA) dt  (time-integrated angular-momentum pattern)
#   - |S| at peak slot               (instantaneous Poynting magnitude)
#   - energy density u at peak slot  (instantaneous field energy)
# The "peak slot" is argmax of screen energy over time — the instant the pulse is
# on the screen (the *last* time sample is after the pulse has passed, ≈0). Scalar
# totals (energy, Lz, Lz/U) are printed and annotated. Emits an `observables`
# derived sidecar per field (total/radiation → dashboard secondary picker).
#
#   julia --project=scripts scripts/plot_screen_observables.jl [field.jls]

using ElectronDynamicsModels
using Serialization
using Printf
using CairoMakie
include(joinpath(@__DIR__, "plot_theme.jl"))   # LaTeX (Computer Modern) fonts

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
xg, yg = collect(screen.x_grid), collect(screen.y_grid)
dt = step(screen.x⁰_samples) / c

println("computing screen_observables (total + radiation) …")
obs_tot = screen_observables(fld, screen; ε₀)
obs_rad = screen_observables((; E = fld.E_rad, B = fld.B_rad), screen; ε₀)

# Peak emission slot from the total field's screen-integrated energy density.
slot_energy = dropdims(sum(obs_tot.energy_density; dims = (2, 3)); dims = (2, 3))
const k_peak = argmax(slot_energy)
@printf("peak emission slot k = %d of %d  (x⁰ = %.4g)\n", k_peak, N_samples, screen.x⁰_samples[k_peak])

@printf("\n%-12s %14s %14s %14s\n", "", "energy_total", "Lz_total", "Lz/U")
for (name, o) in (("total", obs_tot), ("radiation", obs_rad))
    @printf("%-12s %14.4e %14.4e %14.4e\n", name, o.energy_total, o.Lz_total,
        o.energy_total == 0 ? NaN : o.Lz_total / o.energy_total)
end

# Per-pixel maps.
energy_fluence(o) = dropdims(sum(@view(o.S[:, 3, :, :]); dims = 1); dims = 1) .* dt
lz_fluence(o) = dropdims(sum(o.Lz_density; dims = 1); dims = 1) .* dt
Smag_peak(o) = [hypot(o.S[k_peak, 1, i, j], o.S[k_peak, 2, i, j], o.S[k_peak, 3, i, j]) for i in 1:Nx, j in 1:Ny]
u_peak(o) = o.energy_density[k_peak, :, :]

# Float32-safe colorrange: Makie casts data + range to Float32 and errors if the
# range collapses (cmin==cmax) — which happens when the values are so tiny they
# underflow Float32 (e.g. the transverse Sφ ~1e-50 at low resolution). Fall back
# to a unit range in that case so the (effectively-zero) panel still renders.
function _crange(data; diverging)
    if diverging
        m = maximum(abs, data)
        return (isfinite(m) && Float32(m) > 0.0f0) ? (-m, m) : (-1.0, 1.0)
    else
        lo, hi = extrema(data)
        return (isfinite(hi) && Float32(hi - lo) > 0.0f0) ? (Float64(lo), Float64(hi)) : (0.0, 1.0)
    end
end

function panel!(fig, pos, data, title, cmap)
    gl = fig[pos...] = GridLayout()
    ax = Axis(gl[1, 1]; width = 300, height = 300, xlabel = "x [a.u.]", ylabel = "y [a.u.]", title = title)
    hm = heatmap!(ax, xg, yg, data; colormap = cmap, colorrange = _crange(data; diverging = cmap == :seismic))
    Colorbar(gl[1, 2], hm, width = 12, height = 300)
    return ax
end

function field_figure(o, fieldname, outfile)
    fig = Figure()
    Label(fig[0, :], @sprintf("Screen observables — %s field — %s\nU=%.3e  Lz=%.3e  Lz/U=%.3e",
            fieldname, basename(stem), o.energy_total, o.Lz_total,
            o.energy_total == 0 ? NaN : o.Lz_total / o.energy_total),
        fontsize = 13, font = :bold)
    panel!(fig, (1, 1), energy_fluence(o), L"\int S^z\,\mathrm{d}t", :viridis)
    panel!(fig, (1, 2), lz_fluence(o), L"\int (\mathrm{d}L_z/\mathrm{d}A)\,\mathrm{d}t", :seismic)
    panel!(fig, (2, 1), Smag_peak(o), "|S|   (peak slot k=$k_peak)", :viridis)
    panel!(fig, (2, 2), u_peak(o), "energy density u   (peak slot k=$k_peak)", :viridis)
    resize_to_layout!(fig)
    save(outfile, fig)
    println("saved → $outfile")
    return outfile
end

out_tot = field_figure(obs_tot, "total", stem * "_observables_total.png")
out_rad = field_figure(obs_rad, "radiation", stem * "_observables_radiation.png")

# ── Vortex figure: azimuthal Poynting Sφ + (Sˣ,Sʸ) circulation at the peak slot ──
# Sφ = (x Sʸ − y Sˣ)/r is the azimuthal energy flux (its r-moment integrates to L_z);
# its sign is the OAM handedness, and the (Sˣ,Sʸ) arrows show the swirl directly.
function vortex_panel!(fig, pos, o, fieldname)
    Sx = o.S[k_peak, 1, :, :]
    Sy = o.S[k_peak, 2, :, :]
    Sphi = [(xg[i] * Sy[i, j] - yg[j] * Sx[i, j]) / max(hypot(xg[i], yg[j]), eps()) for i in 1:Nx, j in 1:Ny]
    gl = fig[pos...] = GridLayout()
    ax = Axis(gl[1, 1]; width = 320, height = 320, xlabel = "x [a.u.]", ylabel = "y [a.u.]",
        title = "$fieldname:   Sφ + (Sˣ,Sʸ)   @ k=$k_peak")
    hm = heatmap!(ax, xg, yg, Sphi; colormap = :seismic, colorrange = _crange(Sphi; diverging = true))
    Colorbar(gl[1, 2], hm, width = 12, height = 320)
    # subsampled, direction-normalized arrows (length ∝ subsample spacing, not |S⊥|)
    st = max(1, Nx ÷ 18)
    L = 0.8 * st * (xg[2] - xg[1])
    px = Float64[]; py = Float64[]; du = Float64[]; dv = Float64[]
    for i in 1:st:Nx, j in 1:st:Ny
        m = hypot(Sx[i, j], Sy[i, j])
        m == 0 && continue
        push!(px, xg[i]); push!(py, yg[j]); push!(du, L * Sx[i, j] / m); push!(dv, L * Sy[i, j] / m)
    end
    arrows!(ax, px, py, du, dv; lengthscale = 1, arrowsize = 7, color = (:black, 0.6), linewidth = 1)
    return ax
end

function vortex_figure(outfile)
    fig = Figure()
    Label(fig[0, :], "Transverse-Poynting circulation @ peak slot — $(basename(stem))", fontsize = 13, font = :bold)
    vortex_panel!(fig, (1, 1), obs_tot, "total")
    vortex_panel!(fig, (1, 2), obs_rad, "radiation")
    resize_to_layout!(fig)
    save(outfile, fig)
    println("saved → $outfile")
    return outfile
end

# ── Temporal figure: radiated power P(t) and AM emission rate dLz/dt per slot ──
function temporal_figure(outfile)
    dA = step(screen.x_grid) * step(screen.y_grid)
    Pt(o) = dropdims(sum(@view(o.S[:, 3, :, :]); dims = (2, 3)); dims = (2, 3)) .* dA
    Lzt(o) = dropdims(sum(o.Lz_density; dims = (2, 3)); dims = (2, 3)) .* dA
    ks = 1:N_samples
    fig = Figure()
    Label(fig[0, :], "Temporal profiles — $(basename(stem))", fontsize = 13, font = :bold)
    ax1 = Axis(fig[1, 1]; width = 460, height = 220, xlabel = "observer-time slot",
        ylabel = L"P = \int S^z\,\mathrm{d}A", title = "radiated power")
    lines!(ax1, ks, Pt(obs_tot); label = "total")
    lines!(ax1, ks, Pt(obs_rad); label = "radiation", linestyle = :dash)
    vlines!(ax1, [k_peak]; color = (:gray, 0.7), linestyle = :dot)
    axislegend(ax1; labelsize = 9)
    ax2 = Axis(fig[2, 1]; width = 460, height = 220, xlabel = "observer-time slot",
        ylabel = L"\mathrm{d}L_z/\mathrm{d}t = \int (\mathrm{d}L_z/\mathrm{d}A)\,\mathrm{d}A",
        title = "angular-momentum emission rate")
    lines!(ax2, ks, Lzt(obs_tot); label = "total")
    lines!(ax2, ks, Lzt(obs_rad); label = "radiation", linestyle = :dash)
    vlines!(ax2, [k_peak]; color = (:gray, 0.7), linestyle = :dot)
    axislegend(ax2; labelsize = 9)
    resize_to_layout!(fig)
    save(outfile, fig)
    println("saved → $outfile")
    return outfile
end

out_vortex = vortex_figure(stem * "_observables_vortex.png")
out_temporal = temporal_figure(stem * "_observables_temporal.png")

# ── Derived sidecars: maps (per field), vortex, temporal ──
let pid = parent[1]
    if pid === nothing
        @warn "parent run manifest for $(basename(datafile)) has no run_id; skipping sidecar"
    else
        for (fieldname, o, out) in (("total", obs_tot, out_tot), ("radiation", obs_rad, out_rad))
            write_derived(dir; kind = "observables",
                label = @sprintf("screen observables (%s): U=%.2e, Lz=%.2e", fieldname, o.energy_total, o.Lz_total),
                run_id = pid, plot = basename(out), source = basename(datafile),
                setup = Dict("field" => fieldname),
                description = "Screen observables (" * fieldname * " field). " *
                    raw"Top: time-integrated energy fluence $\int S^z\,dt$ and angular-momentum flux " *
                    raw"$\int \frac{dL_z}{dA}\,dt$. Bottom: instantaneous Poynting magnitude $|\mathbf{S}|$ and " *
                    raw"energy density $u=\tfrac{1}{2}\varepsilon_0(E^2+c^2B^2)$ at the peak emission slot. " *
                    raw"Radiated angular momentum per energy: $L_z/U$ (×$\hbar\omega$ ⇒ per photon).")
        end
        write_derived(dir; kind = "obs_vortex", label = "transverse-Poynting circulation @ peak",
            run_id = pid, plot = basename(out_vortex), source = basename(datafile),
            description = raw"Azimuthal Poynting $S_\varphi=(x\,S^y-y\,S^x)/r$ (colour; sign is the OAM handedness) " *
                raw"with transverse $(S^x,S^y)$ arrows at the peak slot. Its radial moment integrates to " *
                raw"$L_z=\int(x\,T^{zy}-y\,T^{zx})\,dA$, so the swirl is a direct picture of the radiated angular momentum.")
        write_derived(dir; kind = "obs_temporal", label = "radiated power + L_z rate vs time",
            run_id = pid, plot = basename(out_temporal), source = basename(datafile),
            description = raw"Radiated power $P(t)=\int S^z\,dA$ and angular-momentum emission rate " *
                raw"$\mathrm{d}L_z/\mathrm{d}t=\int\frac{dL_z}{dA}\,dA$ per observer-time slot (total vs radiation field). " *
                raw"The dotted line marks the peak-emission slot used for the snapshots above.")
        println("derived sidecars → observables (total/radiation) + obs_vortex + obs_temporal (parent run $pid)")
    end
end
