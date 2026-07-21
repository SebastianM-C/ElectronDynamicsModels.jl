# Screen-observable maps for a field-accumulation run (the field .jls from
# thomson_scattering.jl), via `screen_observables`. Total field (E,B) and the
# far-only field (E_far,B_far) get *separate* figures, each with:
#   - energy fluence   ∫Sᶻ dt        (time-integrated — "what got radiated where")
#   - L_z flux         ∫(dLz/dA) dt  (time-integrated angular-momentum pattern)
#   - |S| at peak slot               (instantaneous Poynting magnitude)
#   - energy density u at peak slot  (instantaneous field energy)
# The "peak slot" is argmax of screen energy over time — the instant the pulse is
# on the screen (the *last* time sample is after the pulse has passed, ≈0). Scalar
# totals (energy, Lz, Lz/U) are printed and annotated. Emits an `observables`
# derived sidecar per field (total/far → dashboard secondary picker).
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
# Screen geometry from the run's [setup] when recorded (EDM_SCREEN_HW / windowed runs, e.g.
# inverse_thomson_scattering.jl); legacy fallbacks reproduce the historical ±25w₀ full window.
const setup_sec = get(parent[2], "setup", Dict{String, Any}())
const screen_hw = get(setup_sec, "screen_hw", 25w₀)
const x⁰_start_rec = get(setup_sec, "x0_start", nothing)

function thomson_screen(Nx, N_samples)
    δt = 2π / ω / spp
    x⁰_start = x⁰_start_rec === nothing ? c * τi + hypot(Z, screen_hw + Rmax) : x⁰_start_rec
    x⁰ = range(start = x⁰_start, step = c * δt, length = N_samples)
    return ObserverScreen(LinRange(-screen_hw, screen_hw, Nx), LinRange(-screen_hw, screen_hw, Nx), Z, x⁰; c)
end

# ── Reduced products + cache ──────────────────────────────────────────────────
# The figures only ever use *reductions* of screen_observables (∫…dt maps, peak-slot
# snapshots, length-N time series). The full output is ~38 GB/field (S alone is
# (N,3,Nx,Ny)), so we cache just the reductions (~16 MB) in <stem>_obscache.jls and
# skip BOTH the 86 GB .jls reload and the (minutes-long) screen_observables recompute
# when re-plotting. Bump OBS_CACHE_VERSION whenever the reduced schema/math changes,
# so an older cache is detected as stale and recomputed.
const OBS_CACHE_VERSION = 3   # v3: screen geometry (screen_hw / x0_start) now read from the manifest
const cachefile = stem * "_obscache.jls"
const recompute = ("--recompute" in ARGS) || get(ENV, "EDM_OBS_RECOMPUTE", "0") == "1"

# Collapse one screen_observables result to the plot-ready quantities (per field).
reduce_field(o, kp, dt, dA) = (;
    energy_fluence = dropdims(sum(@view(o.S[:, 3, :, :]); dims = 1); dims = 1) .* dt,
    lz_fluence = dropdims(sum(o.Lz_density; dims = 1); dims = 1) .* dt,
    Smag_peak = [hypot(o.S[kp, 1, i, j], o.S[kp, 2, i, j], o.S[kp, 3, i, j]) for i in axes(o.S, 3), j in axes(o.S, 4)],
    u_peak = o.energy_density[kp, :, :],
    Sx_peak = o.S[kp, 1, :, :],
    Sy_peak = o.S[kp, 2, :, :],
    Pt = dropdims(sum(@view(o.S[:, 3, :, :]); dims = (2, 3)); dims = (2, 3)) .* dA,
    Lzt = dropdims(sum(o.Lz_density; dims = (2, 3)); dims = (2, 3)) .* dA,
    energy_total = o.energy_total,
    Lz_total = o.Lz_total,
)

# The heavy path: load the 86 GB field, run screen_observables, reduce to ~16 MB.
# EDM_DIRECT_READ=1 streams via O_DIRECT to keep page cache off the container cgroup
# (same rationale + failure mode as harmonic_products._read_cube).
function build_cache()
    println("loading $datafile …")
    fld = get(ENV, "EDM_DIRECT_READ", "0") == "1" ?
        open(deserialize, `dd if=$datafile bs=64M iflag=direct status=none`) : deserialize(datafile)
    Ns, _, nx, ny = size(fld.E)
    scr = thomson_screen(nx, Ns)
    dt = step(scr.x⁰_samples) / c
    dA = step(scr.x_grid) * step(scr.y_grid)
    # `:total` runs (accumulate_field mode=Val(:total)) save only (E, B); the far
    # field (E_far, B_far) exists only for `:split`. At the screen distance the near field
    # is ~1e-9 of the far field, so the total observables ≈ the far ones anyway.
    has_far = hasproperty(fld, :E_far) && hasproperty(fld, :B_far)
    println("computing screen_observables (total$(has_far ? " + far" : "; :total run — no separate far field")) …")
    ot = screen_observables(fld, scr; ε₀)
    ofar = has_far ? screen_observables((; E = fld.E_far, B = fld.B_far), scr; ε₀) : nothing
    se = dropdims(sum(ot.energy_density; dims = (2, 3)); dims = (2, 3))
    kp = argmax(se)
    return (;
        version = OBS_CACHE_VERSION,
        total = reduce_field(ot, kp, dt, dA),
        far = has_far ? reduce_field(ofar, kp, dt, dA) : nothing,
        k_peak = kp, slot_energy = se, Nx = nx, Ny = ny, N_samples = Ns,
    )
end

function build_and_save()
    cc = build_cache()
    serialize(cachefile, cc)
    println(@sprintf("cached reductions → %s  (%.1f MB)", cachefile, filesize(cachefile) / 1.0e6))
    return cc
end

# Cache-invalidation policy: decide whether to reuse <stem>_obscache.jls or recompute.
function load_or_build()
    return if recompute || !(isfile(cachefile)) || (mtime(datafile) > mtime(cachefile))
        cc = build_and_save()
    else
        cc = deserialize(cachefile)
        if haskey(cc, :version) && cc.version == OBS_CACHE_VERSION
            return cc
        else
            cc = build_and_save()
        end
    end
end

const cache = load_or_build()
const Nx, Ny, N_samples, k_peak = cache.Nx, cache.Ny, cache.N_samples, cache.k_peak
const red_tot, red_far = cache.total, cache.far
const has_far = red_far !== nothing   # :split saves the radiation field; :total does not
const screen = thomson_screen(Nx, N_samples)
const xg, yg = collect(screen.x_grid), collect(screen.y_grid)
@printf("peak emission slot k = %d of %d  (x⁰ = %.4g)\n", k_peak, N_samples, screen.x⁰_samples[k_peak])

@printf("\n%-12s %14s %14s %14s\n", "", "energy_total", "Lz_total", "Lz/U")
for (name, r) in (has_far ? (("total", red_tot), ("far", red_far)) : (("total", red_tot),))
    @printf(
        "%-12s %14.4e %14.4e %14.4e\n", name, r.energy_total, r.Lz_total,
        r.energy_total == 0 ? NaN : r.Lz_total / r.energy_total
    )
end

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

function field_figure(r, fieldname, outfile)
    fig = Figure()
    Label(
        fig[0, :], @sprintf(
            "Screen observables — %s field — %s\nU=%.3e  Lz=%.3e  Lz/U=%.3e",
            fieldname, basename(stem), r.energy_total, r.Lz_total,
            r.energy_total == 0 ? NaN : r.Lz_total / r.energy_total
        ),
        fontsize = 13, font = :bold
    )
    panel!(fig, (1, 1), r.energy_fluence, L"\int S^z\,\mathrm{d}t", :viridis)
    panel!(fig, (1, 2), r.lz_fluence, L"\int (\mathrm{d}L_z/\mathrm{d}A)\,\mathrm{d}t", :seismic)
    panel!(fig, (2, 1), r.Smag_peak, "|S|   (peak slot k=$k_peak)", :viridis)
    panel!(fig, (2, 2), r.u_peak, "energy density u   (peak slot k=$k_peak)", :viridis)
    resize_to_layout!(fig)
    save(outfile, fig)
    println("saved → $outfile")
    return outfile
end

out_tot = field_figure(red_tot, "total", stem * "_observables_total.png")
out_far = has_far ? field_figure(red_far, "far", stem * "_observables_far.png") : nothing

# ── Vortex figure: azimuthal Poynting Sφ + (Sˣ,Sʸ) circulation at the peak slot ──
# Sφ = (x Sʸ − y Sˣ)/r is the azimuthal energy flux (its r-moment integrates to L_z);
# its sign is the OAM handedness, and the (Sˣ,Sʸ) arrows show the swirl directly.
function vortex_panel!(fig, pos, r, fieldname)
    Sx = r.Sx_peak
    Sy = r.Sy_peak
    Sphi = [(xg[i] * Sy[i, j] - yg[j] * Sx[i, j]) / max(hypot(xg[i], yg[j]), eps()) for i in 1:Nx, j in 1:Ny]
    gl = fig[pos...] = GridLayout()
    ax = Axis(
        gl[1, 1]; width = 320, height = 320, xlabel = "x [a.u.]", ylabel = "y [a.u.]",
        title = "$fieldname:   Sφ + (Sˣ,Sʸ)   @ k=$k_peak"
    )
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
    vortex_panel!(fig, (1, 1), red_tot, "total")
    has_far && vortex_panel!(fig, (1, 2), red_far, "far")
    resize_to_layout!(fig)
    save(outfile, fig)
    println("saved → $outfile")
    return outfile
end

# ── Temporal figure: radiated power P(t) and AM emission rate dLz/dt per slot ──
function temporal_figure(outfile)
    ks = 1:N_samples
    fig = Figure()
    Label(fig[0, :], "Temporal profiles — $(basename(stem))", fontsize = 13, font = :bold)
    ax1 = Axis(
        fig[1, 1]; width = 460, height = 220, xlabel = "observer-time slot",
        ylabel = L"P = \int S^z\,\mathrm{d}A", title = "radiated power"
    )
    lines!(ax1, ks, red_tot.Pt; label = "total")
    has_far && lines!(ax1, ks, red_far.Pt; label = "far", linestyle = :dash)
    vlines!(ax1, [k_peak]; color = (:gray, 0.7), linestyle = :dot)
    axislegend(ax1; labelsize = 9)
    ax2 = Axis(
        fig[2, 1]; width = 460, height = 220, xlabel = "observer-time slot",
        ylabel = L"\mathrm{d}L_z/\mathrm{d}t = \int (\mathrm{d}L_z/\mathrm{d}A)\,\mathrm{d}A",
        title = "angular-momentum emission rate"
    )
    lines!(ax2, ks, red_tot.Lzt; label = "total")
    has_far && lines!(ax2, ks, red_far.Lzt; label = "far", linestyle = :dash)
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
        for (fieldname, r, out) in (has_far ? (("total", red_tot, out_tot), ("far", red_far, out_far)) : (("total", red_tot, out_tot),))
            write_derived(
                dir; kind = "observables",
                label = @sprintf("screen observables (%s): U=%.2e, Lz=%.2e", fieldname, r.energy_total, r.Lz_total),
                run_id = pid, plot = basename(out), source = basename(datafile),
                setup = Dict("field" => fieldname),
                description = "Screen observables (" * fieldname * " field). " *
                    raw"Top: time-integrated energy fluence $\int S^z\,dt$ and angular-momentum flux " *
                    raw"$\int \frac{dL_z}{dA}\,dt$. Bottom: instantaneous Poynting magnitude $|\mathbf{S}|$ and " *
                    raw"energy density $u=\tfrac{1}{2}\varepsilon_0(E^2+c^2B^2)$ at the peak emission slot. " *
                    raw"Radiated angular momentum per energy: $L_z/U$ (×$\hbar\omega$ ⇒ per photon)."
            )
        end
        write_derived(
            dir; kind = "obs_vortex", label = "transverse-Poynting circulation @ peak",
            run_id = pid, plot = basename(out_vortex), source = basename(datafile),
            description = raw"Azimuthal Poynting $S_\varphi=(x\,S^y-y\,S^x)/r$ (colour; sign is the OAM handedness) " *
                raw"with transverse $(S^x,S^y)$ arrows at the peak slot. Its radial moment integrates to " *
                raw"$L_z=\int(x\,T^{zy}-y\,T^{zx})\,dA$, so the swirl is a direct picture of the radiated angular momentum."
        )
        write_derived(
            dir; kind = "obs_temporal", label = "radiated power + L_z rate vs time",
            run_id = pid, plot = basename(out_temporal), source = basename(datafile),
            description = raw"Radiated power $P(t)=\int S^z\,dA$ and angular-momentum emission rate " *
                raw"$\mathrm{d}L_z/\mathrm{d}t=\int\frac{dL_z}{dA}\,dA$ per observer-time slot (total vs far field). " *
                raw"The dotted line marks the peak-emission slot used for the snapshots above."
        )
        println("derived sidecars → observables (total/far) + obs_vortex + obs_temporal (parent run $pid)")
        # Drain-path only (EDM_REDUCTION_MARKER): enumerate the obs cache in <uuid>.reduced (cf. harmonic_products).
        get(ENV, "EDM_REDUCTION_MARKER", "0") == "1" && record_reduction!(dir, pid, cachefile)
    end
end
