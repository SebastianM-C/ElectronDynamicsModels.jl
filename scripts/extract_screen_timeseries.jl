# Sample screen pixels and extract the time-domain E(t), B(t) per component from a field cube.
# The cube (field_*.jls) holds fld.E/fld.B as (N_samples, 3, Nx, Ny) TOTAL fields; this pulls a
# handful of pixels' full, un-windowed waveforms into a small timeseries_<run_id>.jls and records
# it in the run's .reduced marker, so it publishes with the run (storagebox, RunManifests key
# :timeseries) and survives the cube.
#
#   EDM_TS_PIXELS="x1,y1;x2,y2;…" julia --project=scripts extract_screen_timeseries.jl run_<id>.toml […]
# EDM_TS_PIXELS: pixel centres in w₀ units (nearest pixel), same syntax as EDM_POSITIONS. Unset ⇒
# the radii × azimuths default below, clipped to the screen.
using TOML, Serialization, RunManifests

const RADII_W0 = (4.0, 8.0, 12.0)
const AZ = (0.0, π / 2, π, 3π / 2)        # 0°, 90°, 180°, 270°

requested_pixels() = map(split(get(ENV, "EDM_TS_PIXELS", ""), ';'; keepempty = false)) do p
    xy = parse.(Float64, strip.(split(p, ',')))
    length(xy) == 2 || error("EDM_TS_PIXELS: expected `x,y` pairs separated by `;`, got \"$p\"")
    (xy[1], xy[2])
end

function extract(toml)
    m = TOML.parsefile(toml)
    dir = dirname(abspath(toml))
    cfg, las, setup = m["config"], m["laser"], get(m, "setup", Dict())
    run_tag = m["provenance"]["run_id"]
    cube = joinpath(dir, m["outputs"]["datafile"])
    isfile(cube) || (@warn "cube absent — skipping" run_tag cube; return)
    Nx, Ny, w0, spp = cfg["Nx"], cfg["Ny"], las["w0"], cfg["samples_per_period"]
    hw = get(setup, "screen_hw", 25 * w0)            # the run's screen half-width (legacy forward runs: 25 w₀)
    x_grid = LinRange(-hw, hw, Nx)
    y_grid = LinRange(-hw, hw, Ny)
    req = requested_pixels()
    targets = isempty(req) ? [(R * cos(φ), R * sin(φ)) for R in RADII_W0 for φ in AZ if R * w0 <= hw] : req
    for (x, y) in targets   # nearest-pixel would silently clamp to the edge
        max(abs(x), abs(y)) * w0 <= hw || error("EDM_TS_PIXELS: ($x, $y) w₀ is off the ±$(hw / w0) w₀ screen")
    end
    println("loading $(basename(cube)) …")
    fld = deserialize(cube)                          # (; E, B[, E_far, B_far]); E,B are (Ns,3,Nx,Ny) totals
    N_samples = size(fld.E, 1)
    pixels = map(targets) do (x, y)
        ix = argmin(abs.(x_grid .- x * w0))
        iy = argmin(abs.(y_grid .- y * w0))
        (; x_req_w0 = x, y_req_w0 = y, ix, iy, x_over_w0 = x_grid[ix] / w0, y_over_w0 = y_grid[iy] / w0,
            E = Array(fld.E[:, :, ix, iy]),          # (N_samples, 3): Eˣ Eʸ Eᶻ vs sample, un-windowed
            B = Array(fld.B[:, :, ix, iy]))
    end
    # Native time axis: observer time x⁰ = x0_start + (k−1)·c·δt, δt = T_laser/spp (a.u.).
    ω = 2π * 137.035999177 / las["wavelength"]       # c in a.u.
    δt = 2π / ω / spp
    out = joinpath(dir, "timeseries_$(run_tag).jls")
    serialize(out, (; run_tag, a0 = las["a0"], spp, N_samples, w0, δt, x0_start = get(setup, "x0_start", nothing),
        screen_hw = hw, pixels))
    final = joinpath(dir, "$(run_tag).reduced")
    if !isfile(final)
        # an uncommitted .partial without a final marker ⇒ a deferred reduce is still writing it
        isfile(final * ".partial") && error("$(run_tag): reduce in flight (.reduced.partial, no .reduced) — rerun after it commits")
        # sync-postprocess runs write no marker: seed it with the builder's convention set (every
        # uuid-tagged .jls but the cube) so the marker stays complete once it exists
        for f in readdir(dir)
            endswith(f, ".jls") && occursin(run_tag, f) && f ∉ (basename(cube), basename(out)) &&
                record_reduction!(dir, run_tag, f)
        end
    end
    partial = record_reduction!(dir, run_tag, out)
    mv(partial, final; force = true)   # commit atomically, as run_cell.sh does
    println("wrote $(basename(out)) — $(length(pixels)) pixels × $N_samples samples")
end

isempty(ARGS) && error("usage: extract_screen_timeseries.jl run_<id>.toml [run_<id>.toml ...]")
foreach(extract, ARGS)
