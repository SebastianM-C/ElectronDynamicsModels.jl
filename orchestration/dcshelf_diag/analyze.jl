# dcshelf_diag analysis: per-pixel Eᶻ edge forensics + peak h2/h1 (E, B[, E_far, B_far]) under rect and Hann.
#   julia +1.12 --project=scripts orchestration/dcshelf_diag/analyze.jl <campaign_dir> [june_timeseries_dir]
# The June comparison reads the φ₀-ladder timeseries extracts (f5233fc era, pre window-end coverage).
using ElectronDynamicsModels, Serialization, TOML, Printf, Statistics

const C_AU = 137.035999177

# Eᶻ at pixels (0,0), (R,0) R = 4, 8, 12 w₀: start/end levels, exact zeros in the last 2 %, biggest step.
function edge_forensics(E, x_grid, y_grid, w0)
    n = size(E, 1); k = max(1, n ÷ 50)
    for R in (0.0, 4.0, 8.0, 12.0)
        ix = argmin(abs.(x_grid .- R * w0)); iy = argmin(abs.(y_grid))
        Ez = E[:, 3, ix, iy]; d = abs.(diff(Ez)); j = argmax(d)
        @printf "    R=%4.1f w₀  Ez head=% .4e tail=% .4e  zeros(last %d)=%d  max step=%.2e at k=%d/%d  osc rms=%.2e\n" R mean(Ez[1:k]) mean(Ez[end-k+1:end]) k count(==(0.0), Ez[end-k+1:end]) d[j] j n std(Ez)
    end
end

# Peak-over-screen |h| per field for harmonics 1 and 2; ratio h2/h1.
function h21(cube, bins; window)
    h = harmonic_maps(cube, bins; window)                        # (2, 3, Nx, Ny)
    pk(i) = maximum(sqrt.(dropdims(sum(abs2, h[i, :, :, :]; dims = 1); dims = 1)))
    pk(1), pk(2), pk(2) / pk(1)
end

function analyze_run(toml)
    m = TOML.parsefile(toml); dir = dirname(abspath(toml))
    cfg, las, setup = m["config"], m["laser"], m["setup"]
    w0, spp = las["w0"], cfg["samples_per_period"]
    ω = 2π * C_AU / las["wavelength"]; δt = 2π / ω / spp
    fld = deserialize(joinpath(dir, m["outputs"]["datafile"]))
    Ns = size(fld.E, 1); hw = setup["screen_hw"]
    xg = LinRange(-hw, hw, size(fld.E, 3)); yg = LinRange(-hw, hw, size(fld.E, 4))
    # Static field of N charges (q = −1 a.u.) at distance Z, on axis: Eᶻ ≈ −N/Z² × sign(ẑ)
    Z = get(setup, "Z", NaN)
    @printf "\n%s  a0=%.0e  mode=%s  Ns=%d  N=%d   static Coulomb on axis ≈ %.4e (N/Z²)\n" m["provenance"]["run_id"][1:8] las["a0"] cfg["mode"] Ns cfg["N"] cfg["N"] / Z^2
    edge_forensics(fld.E, xg, yg, w0)
    bins = harmonic_bins(Ns, δt, ω, (1, 2))
    fields = [("E", fld.E), ("B", fld.B)]
    hasproperty(fld, :E_far) && append!(fields, [("E_far", fld.E_far), ("B_far", fld.B_far)])
    for (name, cube) in fields, (wn, w) in (("rect", nothing), ("hann", hann))
        h1, h2, r = h21(cube, bins; window = w)
        @printf "    %-5s %-4s |h1|=%.3e |h2|=%.3e  h2/h1=%.3e\n" name wn h1 h2 r
    end
end

function june(dir)
    println("\n== June φ₀ ladder (f5233fc, pre window-end coverage): Eᶻ edge forensics from timeseries extracts")
    for f in sort(filter(f -> startswith(basename(f), "timeseries_"), readdir(dir; join = true)))
        t = deserialize(f); @printf "  %s a0=%.0e\n" basename(f)[12:19] t.a0
        for p in t.pixels[1:4:end]
            Ez = p.E[:, 3]; n = length(Ez); k = n ÷ 50; d = abs.(diff(Ez)); j = argmax(d)
            @printf "    R=%4.1f w₀  Ez head=% .4e tail=% .4e  zeros(last %d)=%d  max step=%.2e at k=%d/%d\n" p.R_over_w0 mean(Ez[1:k]) mean(Ez[end-k+1:end]) k count(==(0.0), Ez[end-k+1:end]) d[j] j n
        end
    end
end

camp = ARGS[1]
foreach(analyze_run, sort(filter(f -> startswith(basename(f), "run_") && endswith(f, ".toml"), readdir(camp; join = true))))
length(ARGS) > 1 && june(ARGS[2])
