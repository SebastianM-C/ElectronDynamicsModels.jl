# scripts/shelf_sentinel.jl — per-cell production checks from a reduced run (CPU, seconds):
#   • Coulomb-shelf sentinel: −Eᶻ/(N/Z²) at the first and last observer samples of the timeseries pixels
#     (must both be 1: the static field of N charges at the screen, uncut by the window) and the count of
#     exact zeros in the last 2 % of the window (> 0 ⇒ electrons dropped out before the window closed —
#     the pre-d1d8c99 truncation that faked the June 2ω shelf);
#   • E-vs-B agreement: peak |h2|/|h1| of E and of B from the hmaps (equal when the E side is clean);
#   • provenance assertions: [window].ok (or negligible clipping: slot_fill ≥ 1 − 1e-6 with a flat window end),
#     repo_dirty = false, [config].apodization as reduced.
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

# Window-end flatness: E range over the last 30 periods ÷ the pixel's peak |E − E_end| over all components, max over
# pixels. A tail off N/Z² with a flat window end is the post-pulse charge state (electrons displaced or drifting at
# a₀ ≳ 2), not truncation: the radiated burst has ended inside the window.
const FLAT_MAX = 1e-6
w30 = max(1, ts.N_samples - 30 * cfg["samples_per_period"] + 1):ts.N_samples
flatness = maximum(maximum(maximum(p.E[w30, c]) - minimum(p.E[w30, c]) for c in 1:3) /
                   maximum(abs, p.E .- p.E[end:end, :]) for p in ts.pixels)

win = get(m, "window", Dict())
window_ok = get(win, "ok", missing)
slot_fill = get(win, "slot_fill", missing)
dirty = get(prov, "repo_dirty", missing)
apod = get(cfg, "apodization", "hann")
shelf_state = all(x -> abs(x - 1) < 1e-3, tail) ? "rest" : flatness <= FLAT_MAX ? "drifted" : "truncated"
shelf_ok = all(x -> abs(x - 1) < 1e-3, head) && zeros_tail == 0 && shelf_state != "truncated"
# A few worst electrons a handful of samples short of the window end (strong push at a₀ ≳ 10) drop < 1e-6 of the slots;
# with a flat window end those slots carry no radiated signal.
window_state = window_ok === true ? "ok" :
    slot_fill isa Real && slot_fill >= 1 - FLAT_MAX && flatness <= FLAT_MAX ? "negligible clipping" : "clipped"
ok = shelf_ok && window_state != "clipped" && dirty === false
@printf "%s shelf head %.6f…%.6f tail %.6f…%.6f (%s, flatness %.1e) zeros %d | h2/h1 E %.4e B %.4e | window %s dirty %s apod %s → %s\n" id8 extrema(head)... extrema(tail)... shelf_state flatness zeros_tail h21E h21B window_state dirty apod (ok ? "OK" : "FAIL")

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
notes = filter(!isempty, [shelf_state == "drifted" ? "drifted charges" : "", window_state == "negligible clipping" ? "negligible clipping" : ""])
write_derived(dir; kind = "shelf", label = !ok ? "production checks: FAIL" :
        isempty(notes) ? "production checks: pass" : "production checks: pass ($(join(notes, ", ")), window flat)", run_id = id,
    plot = basename(png), source = basename(toml),
    plot_params = Dict("coulomb_N_over_Z2" => coul, "head_min" => minimum(head), "head_max" => maximum(head),
        "tail_min" => minimum(tail), "tail_max" => maximum(tail), "tail_zeros_last_2pct" => zeros_tail,
        "h2_over_h1_E" => h21E, "h2_over_h1_B" => h21B, "window_ok" => string(window_ok),
        "repo_dirty" => string(dirty), "apodization" => apod, "pass" => ok,
        "shelf_state" => shelf_state, "window_state" => window_state, "slot_fill" => string(slot_fill),
        "window_end_flatness" => flatness, "flatness_max" => FLAT_MAX),
    description = "−Eᶻ at the timeseries pixels over the observer window, normalized to the static field N/Z² of the electrons at the screen: flat at 1 from the first to the last sample means the window is fully covered (no Coulomb step, no dropped electrons). At a₀ ≳ 2 the post-pulse charges are displaced (LPWA, tail > 1) or drifting (focused beam, tail < 1), so the tail leaves 1 physically; the shelf then passes as \"drifted\" when the field is flat at the window end (range over the last 30 periods ≤ 1e-6 of the pixel's peak excursion), i.e. the radiated burst ended inside the window. Also records the E-vs-B h2/h1 agreement and the run's window, dirty-tree and apodization flags.")
# Spectral falloff (aliasing headroom): screen-integrated E power in the harmonic bands n = 2…8 relative to n = 1,
# from the un-windowed powspec cache. Power above the SPP Nyquist folds back identically at every rate, so the
# direct aliasing measure is how many decades the spectrum has fallen before n_Nyquist = SPP/2 (in units of n₀ω).
ps = deserialize(joinpath(dir, "powspec_$id.jls"))
spp = cfg["samples_per_period"]
nf = ps.freqs ./ (ps.freqs[end] / (spp / 2)) ./ ps.n0      # frequency axis in harmonic orders
Epow = vec(sum(ps.ps[:, 1:3]; dims = 2))
band(n) = sum(Epow[abs.(nf .- n) .< 0.5])
nmax = min(8, floor(Int, spp / 2 / ps.n0))
rel = [band(n) / band(1) for n in 2:nmax]
fig2 = Figure(size = (520, 330))
ax = Axis(fig2[1, 1]; yscale = log10, xlabel = "harmonic order n", ylabel = "P_n / P_1 (E, screen-integrated)",
    title = @sprintf("spectral falloff to the SPP %d Nyquist edge (n = %g)", spp, spp / 2 / ps.n0))
scatterlines!(ax, 2:nmax, max.(rel, 1e-30))
png2 = joinpath(dir, "falloff_$id.png"); save(png2, fig2)
write_derived(dir; kind = "falloff", label = @sprintf("spectral falloff: P_%d/P_1 = %.1e", nmax, rel[end]), run_id = id,
    plot = basename(png2), source = "powspec_$id.jls",
    plot_params = merge(Dict("spp" => spp, "n_nyquist" => spp / 2 / ps.n0),
        Dict("P$(n)_over_P1" => r for (n, r) in zip(2:nmax, rel))),
    description = "Screen-integrated E power in the harmonic bands n = 2…8 relative to the fundamental (un-windowed " *
        "power spectrum). Power above the Nyquist order folds back the same way at every sampling rate, so the " *
        "number of decades the spectrum has already fallen by n = SPP/2 is the direct aliasing headroom.")
@printf "%s falloff P_n/P_1 n=2..%d: %s\n" id8 nmax join([@sprintf("%.1e", r) for r in rel], " ")

ok || exit(1)
