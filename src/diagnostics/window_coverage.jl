# Host-side observer-window coverage check — the executed-slot count of the GPU field kernels
# without touching them.
#
# Each GPU thread (pixel) marches only the observer samples inside its pixel's arrival window
# `k_start:k_end` (kernel_newton.jl / kernel_rk4.jl) and silently returns when the window is
# empty. The solver scripts anchor the window START at x⁰_start = c·τi + hypot(Z, hw + Rmax)
# with hw the screen HALF-WIDTH — the edge midpoint, not the corner of a square screen (which
# sits √2·hw out), so the extreme corner pixels see the farthest electrons only from sample
# ≈ (√2−1)·hw·(hw+Rmax)/(Z·δx⁰) on: a negative `lead_margin` here (≈ −150 samples at the
# production framing). Those are pre-pulse static-field samples, harmless for the radiation
# analysis but real missing contributions. The END is not anchored at all: an observer window
# that outlives a pixel's retarded image of τf drops that electron's tail contribution from
# the cube — a source-side truncation nobody wants. This check is therefore a validity chip as
# much as the exact "slots executed" input of the FLOP accounting.
#
# Exact verdict from five pixels per electron: for fixed τ the arrival offset
# t_px = ψ(τ) + ρ²/(R + d³) (see `_window_edge`) is an increasing function of the transverse
# distance ρ² between the pixel and the electron's position at τ, and ρ² is separable and
# convex in the pixel coordinates — so over a rectangular grid t_px is maximal at one of the
# four corners and minimal at the grid node nearest to the electron. Ten spline evaluations
# per electron give k_start_max (corners at τi) and k_end_min (nearest node at τf) exactly;
# only electrons that turn out clipped are recounted over every pixel.

# Nearest index of a sorted grid to `x` (separable ρ² ⇒ per-axis nearest node is the argmin).
function _nearest_index(grid, x)
    n = length(grid)
    i = searchsortedfirst(grid, x)
    i <= 1 && return 1
    i > n && return n
    return abs(grid[i] - x) < abs(x - grid[i - 1]) ? i : i - 1
end

# Arrival offsets of one electron at proper time τ over the four corner pixels (max) and the
# nearest pixel (min) of the screen.
function _window_edge_extremes(gpu_traj, x_grid, y_grid, z_screen, τ)
    Nx, Ny = length(x_grid), length(y_grid)
    v = gpu_traj.itp(τ)
    xe = v[gpu_traj.x_idxs[2]]
    ye = v[gpu_traj.x_idxs[3]]
    ix = _nearest_index(x_grid, xe)
    iy = _nearest_index(y_grid, ye)
    t_min, _ = _window_edge(gpu_traj, SVector{3}(x_grid[ix], y_grid[iy], z_screen), τ)
    t_max = -Inf
    for cx in (1, Nx), cy in (1, Ny)
        t, _ = _window_edge(gpu_traj, SVector{3}(x_grid[cx], y_grid[cy], z_screen), τ)
        t_max = max(t_max, t)
    end
    return t_min, t_max
end

# The kernels' slot-range formulas (strict-interior slots, matching Tsit5 with
# save_start/save_end = false), unclamped so margins can be read off.
_k_start_raw(t_px, t_first, inv_δ) = floor(Int, (t_px - t_first) * inv_δ) + 2
_k_end_raw(t_px, t_first, inv_δ) = ceil(Int, (t_px - t_first) * inv_δ)

# Exact executed-slot count of one electron over every pixel (only needed for clipped electrons).
function _executed_slots_exact(gpu_traj, x_grid, y_grid, z_screen, t_first, inv_δ, N_samples, τi, τf)
    total = 0
    for iy in eachindex(y_grid), ix in eachindex(x_grid)
        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)
        t_i, _ = _window_edge(gpu_traj, r_obs, τi)
        t_f, _ = _window_edge(gpu_traj, r_obs, τf)
        k_start = max(1, _k_start_raw(t_i, t_first, inv_δ))
        k_end = min(N_samples, _k_end_raw(t_f, t_first, inv_δ))
        total += max(0, k_end - k_start + 1)
    end
    return total
end

"""
    window_coverage(trajs, screen; recount = true) -> NamedTuple

Check on the host that the observer window `screen.x⁰_samples` lies inside every pixel's
arrival window for every electron — the condition under which the GPU field kernels
(`accumulate_field` with `GPUKernelNewton` / `GPUKernelRK4`) execute all `N_samples` slots per
(electron, pixel), the cube receives every electron's full history, and the nominal work
`N·N_samples·Nx·Ny` is exact. Uses the kernels' own window arithmetic (`_window_edge`, same
`floor`/`ceil` slot formulas) on the host spline, evaluated at the four corner pixels and the
pixel nearest to the electron, which bound the arrival offsets exactly (see the file header);
clipped electrons are recounted over every pixel when `recount = true`.

Fields of the result:
- `ok::Bool` — every electron fully covered.
- `slots_nominal`, `slots_executed`, `slot_fill = slots_executed / slots_nominal`
  (`slots_executed` is `missing` for a clipped run with `recount = false`).
- `electrons_clipped`, `slots_dropped = slots_nominal - slots_executed`.
- `lead_margin_samples` — samples the window could start earlier before the first pixel is
  clipped at the start (negative = already clipped by that many samples at the worst pixel);
  `tail_margin_samples` — same for the end of the window (a negative value means some
  trajectory ends before the window does, at some pixel).
- `worst_electron` — index with the smallest tail margin.
- `per_electron::Vector` of `(; k_start_max, k_end_min, lead_margin, tail_margin,
  slots_executed)`.

Cost: ten spline evaluations per electron (milliseconds for 10⁴ electrons), plus
`2·Nx·Ny` per clipped electron. Rounding note: `floor`/`ceil` at an exact sample boundary can
differ by one slot between this host evaluation and a device launch; that cannot flip `ok`
except at a measure-zero boundary.
"""
function window_coverage(trajs::AbstractVector{<:TrajectoryInterpolant}, screen::ObserverScreen;
        recount::Bool = true)
    x_grid, y_grid, z_screen = screen.x_grid, screen.y_grid, screen.z
    issorted(x_grid) && issorted(y_grid) ||
        throw(ArgumentError("window_coverage: screen grids must be sorted ascending"))
    Nx, Ny = length(x_grid), length(y_grid)
    N_samples = length(screen.x⁰_samples)
    N_samples >= 1 || throw(ArgumentError("window_coverage: empty observer window"))
    δx⁰ = N_samples > 1 ? step(screen.x⁰_samples) : one(first(screen.x⁰_samples))
    δx⁰ > 0 || throw(ArgumentError("window_coverage: x⁰_samples must be strictly increasing"))
    t_first = first(screen.x⁰_samples) - z_screen   # light-front offset, as in the Newton kernel
    inv_δ = inv(δx⁰)
    N = length(trajs)
    slots_px = N_samples * Nx * Ny

    per = Vector{NamedTuple{(:k_start_max, :k_end_min, :lead_margin, :tail_margin, :slots_executed),
        NTuple{5, Int}}}(undef, N)
    Threads.@threads for e in 1:N
        traj = trajs[e]
        gpu_traj = to_gpu(traj)   # host-array GPUCubicSpline: the kernels' own arithmetic
        τi, τf = first(traj.itp.t), last(traj.itp.t)
        _, t_i_max = _window_edge_extremes(gpu_traj, x_grid, y_grid, z_screen, τi)
        t_f_min, _ = _window_edge_extremes(gpu_traj, x_grid, y_grid, z_screen, τf)
        ks_raw = _k_start_raw(t_i_max, t_first, inv_δ)
        ke_raw = _k_end_raw(t_f_min, t_first, inv_δ)
        k_start_max = max(1, ks_raw)
        k_end_min = min(N_samples, ke_raw)
        covered = k_start_max == 1 && k_end_min == N_samples
        executed = if covered
            slots_px
        elseif recount
            _executed_slots_exact(gpu_traj, x_grid, y_grid, z_screen, t_first, inv_δ, N_samples, τi, τf)
        else
            -1
        end
        per[e] = (; k_start_max, k_end_min, lead_margin = 1 - ks_raw,
            tail_margin = ke_raw - N_samples, slots_executed = executed)
    end

    slots_nominal = N * slots_px
    clipped = count(p -> p.slots_executed != slots_px, per)
    known = all(p -> p.slots_executed >= 0, per)
    slots_executed = known ? sum(p -> p.slots_executed, per; init = 0) : missing
    worst = isempty(per) ? 0 : argmin(p.tail_margin for p in per)
    return (;
        ok = clipped == 0,
        slots_nominal,
        slots_executed,
        slot_fill = known ? slots_executed / slots_nominal : missing,
        electrons_clipped = clipped,
        slots_dropped = known ? slots_nominal - slots_executed : missing,
        lead_margin_samples = isempty(per) ? 0 : minimum(p.lead_margin for p in per),
        tail_margin_samples = isempty(per) ? 0 : minimum(p.tail_margin for p in per),
        worst_electron = worst,
        N_samples,
        per_electron = per,
    )
end
