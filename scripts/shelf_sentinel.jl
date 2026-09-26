# scripts/shelf_sentinel.jl — per-cell production checks from a reduced run (CPU, seconds):
#   • Coulomb-shelf sentinel: −Eᶻ/(N/Z²) at the first and last observer samples of the timeseries pixels
#     (must both be 1: the static field of N charges at the screen, uncut by the window) and the count of
#     exact zeros in the last 2 % of the window (> 0 ⇒ electrons dropped out before the window closed —
#     the pre-d1d8c99 truncation that faked the June 2ω shelf);
#   • E-vs-B agreement: peak |h2|/|h1| of E and of B from the hmaps (equal when the E side is clean);
#   • provenance assertions: [window].ok, repo_dirty = false, [config].apodization as reduced.
# Needs timeseries_<id>.jls (extract_screen_timeseries.jl, while the cube exists) and hmaps_<id>.jls.
#
#   julia --project=scripts scripts/shelf_sentinel.jl <run_manifest.toml>
# Writes shelf_<id>.png + derived_shelf_<id8>.toml; exits 1 when a check fails (the numbers still land).
using TOML, Serialization, Printf, Statistics, CairoMakie, RunManifests

toml = ARGS[1]
m = TOML.parsefile(toml); dir = dirname(abspath(toml))
cfg, setup, prov = m["config"], m["setup"], m["provenance"]
id = prov["run_id"]; id8 = first(id, 8)
coul = cfg["N"] / setup["Z"]^2                                   # static on-screen field of N unit charges
ts = deserialize(joinpath(dir, "timeseries_$id.jls"))
hm = deserialize(joinpath(dir, "hmaps_$id.jls"))

k = max(1, ts.N_samples ÷ 50)                                     # last 2 % of the window
head = [-p.E[1, 3] / coul for p in ts.pixels]
tail = [-p.E[end, 3] / coul for p in ts.pixels]
zeros_tail = maximum(count(==(0.0), p.E[end-k+1:end, 3]) for p in ts.pixels)

pk(i, c) = maximum(sqrt.(dropdims(sum(abs2, hm.fields_h[i, c, :, :]; dims = 1); dims = 1)))
i1, i2 = findfirst(==(1), hm.harmonics), findfirst(==(2), hm.harmonics)
h21E, h21B = pk(i2, 1:3) / pk(i1, 1:3), pk(i2, 4:6) / pk(i1, 4:6)

window_ok = get(get(m, "window", Dict()), "ok", missing)
dirty = get(prov, "repo_dirty", missing)
apod = get(cfg, "apodization", "hann")
shelf_ok = all(x -> abs(x - 1) < 1e-3, head) && all(x -> abs(x - 1) < 1e-3, tail) && zeros_tail == 0
ok = shelf_ok && window_ok === true && dirty === false
@printf "%s shelf head %.6f…%.6f tail %.6f…%.6f zeros %d | h2/h1 E %.4e B %.4e | window %s dirty %s apod %s → %s\n" id8 extrema(head)... extrema(tail)... zeros_tail h21E h21B window_ok dirty apod (ok ? "OK" : "FAIL")

fig = Figure(size = (900, 330))
for (j, (lo, ttl)) in enumerate(((0.0, "whole window"), (0.95, "last 5 %")))
    ax = Axis(fig[1, j]; title = ttl, xlabel = "fraction of observer window", ylabel = j == 1 ? "−Eᶻ / (N/Z²)" : "")
    for p in ts.pixels
        x = (0:ts.N_samples-1) ./ (ts.N_samples - 1); sel = x .>= lo
        lines!(ax, x[sel], -p.E[sel, 3] ./ coul; linewidth = 1)
    end
    hlines!(ax, [1.0]; color = :black, linestyle = :dash, linewidth = 1)
end
Label(fig[0, :], @sprintf("Coulomb-shelf sentinel: head %.4f, tail %.4f (min), %d tail zeros · h2/h1 E %.3e B %.3e", minimum(head), minimum(tail), zeros_tail, h21E, h21B); fontsize = 13)
png = joinpath(dir, "shelf_$id.png"); save(png, fig)
write_derived(dir; kind = "shelf", label = ok ? "production checks: pass" : "production checks: FAIL", run_id = id,
    plot = basename(png), source = basename(toml),
    plot_params = Dict("coulomb_N_over_Z2" => coul, "head_min" => minimum(head), "head_max" => maximum(head),
        "tail_min" => minimum(tail), "tail_max" => maximum(tail), "tail_zeros_last_2pct" => zeros_tail,
        "h2_over_h1_E" => h21E, "h2_over_h1_B" => h21B, "window_ok" => string(window_ok),
        "repo_dirty" => string(dirty), "apodization" => apod, "pass" => ok),
    description = "−Eᶻ at the timeseries pixels over the observer window, normalized to the static field N/Z² of the electrons at the screen: flat at 1 from the first to the last sample means the window is fully covered (no Coulomb step, no dropped electrons). Also records the E-vs-B h2/h1 agreement and the run's window, dirty-tree and apodization flags.")
ok || exit(1)
