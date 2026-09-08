# Algorithmic FLOP profile of the GPU field kernels, measured on the host.
#
# The production kernels are deterministic in work per (electron, pixel, observer sample)
# "slot": a fixed Newton trip count (`_bracketed_slot_solve`) or a fixed number of RK4
# sub-steps, branch-free Liénard–Wiechert algebra, an integer-only spline search. So the
# arithmetic of a run is
#
#     FLOPs = flop_per_slot · slots_executed + flop_per_pixel_launch · N · Nx · Ny
#
# and the two constants can be read off the kernel body itself: the SAME per-electron kernel
# functions run on the KernelAbstractions CPU backend with `CountedFloats.Counted{Float64}`
# buffers, a tiny synthetic electron and a 3 × 3 screen, at three observer-window lengths.
# Differencing two lengths isolates the per-slot cost exactly; the third length asserts the
# linearity (a future data-dependent branch fails loudly here instead of mis-costing runs).
# Nothing touches the GPU: the profile costs milliseconds of CPU and is written into the run
# manifest's [flops] section by the solver scripts.
#
# What the count means: operations AS WRITTEN at the Julia level — after inlining and
# promotion, before LLVM's CSE / dead-code elimination / FMA contraction. That is the usual
# "algorithmic FLOP" figure to put against a device's peak; hardware instruction counts
# (FP64 division and sqrt expand to FMA sequences, CSE removes the antisymmetric duplicates
# of the Faraday tensor) differ by an O(1) factor that only hardware counters can pin down.

const FLOP_CONVENTION = "algorithmic FP64 count of the per-electron field kernel as written \
(CountedFloats on the CPU backend): add/sub/mul/div/sqrt = 1, fma = 2, pow = 1, \
transcendental = 1; comparisons, negation, abs and rounding are not FLOPs"

# The synthetic electron: the analytic transverse-wiggle worldline of the test suite
# (future-directed timelike for g > sqrt(vz² + (A·Ω)²)), 64 knots — the knot count only
# changes the integer spline search, never the FLOP count.
function _profile_trajectory(; g = 1.2, A = 0.25, Ω = 2.0, vz = 0.0, τspan = (0.0, 20.0),
        N = 64, K = 1.0, x0 = 0.0, y0 = 0.0)
    ts = collect(range(τspan[1], τspan[2]; length = N))
    us = [SVector{8}(g * τ, x0 + A * sin(Ω * τ), y0, vz * τ, g, A * Ω * cos(Ω * τ), 0.0, vz)
          for τ in ts]
    itp = CubicSpline(us, ts; extrapolation = ExtrapolationType.Extension)
    as = [SVector{4}(0.0, -A * Ω^2 * sin(Ω * τ), 0.0, 0.0) for τ in ts]
    a_itp = CubicSpline(as, ts; extrapolation = ExtrapolationType.Extension)
    return TrajectoryInterpolant(itp, a_itp, SVector{4, Int}(1, 2, 3, 4),
        SVector{4, Int}(5, 6, 7, 8), K)
end

# Screen of the profile: 3 × 3 pixels at z = 50 on ±8, observer step 0.1 starting at 52 —
# later than every pixel's first arrival (R ∈ [50, 51.3] at τi = 0 ⇒ k_start = 1 and, for
# RK4, bridge_dt > 0) and, for N_samples ≤ 220, ending before every pixel's last arrival
# (x⁰ = 24 + R at τf = 20 ⇒ k_end = N_samples). `window_coverage` re-asserts this.
const _PROFILE_Z = 50.0
const _PROFILE_HALFW = 8.0
const _PROFILE_NPIX = 3
const _PROFILE_δ = 0.1
const _PROFILE_X0_FIRST = 52.0
const _PROFILE_NSAMPLES = (40, 80, 120)

_counted(a::AbstractArray) = Counted{Float64}.(a)
function _counted(s::GPUCubicSpline{D}) where {D}
    t = _counted(s.t)
    z = _counted(s.z)
    return GPUCubicSpline{D, typeof(t), typeof(z)}(t, _counted(s.h), z, _counted(s.c1), _counted(s.c2))
end
_counted(traj::TrajectoryInterpolant) = TrajectoryInterpolant(
    _counted(traj.itp), _counted(traj.a_itp), traj.x_idxs, traj.u_idxs, Counted{Float64}(traj.K))

# One counted launch of the per-electron field kernel on the CPU backend.
function _profile_launch(alg, mode::Val, n::Int, N_samples::Int, traj)
    Nx = Ny = _PROFILE_NPIX
    grid = collect(LinRange(-_PROFILE_HALFW, _PROFILE_HALFW, Nx))
    x_grid = _counted(grid)
    y_grid = _counted(grid)
    gpu_traj = _counted(to_gpu(traj; with_acceleration = true))
    τi, τf = Counted{Float64}(first(traj.itp.t)), Counted{Float64}(last(traj.itp.t))
    z = Counted{Float64}(_PROFILE_Z)
    δ = Counted{Float64}(_PROFILE_δ)
    x⁰_first = Counted{Float64}(_PROFILE_X0_FIRST)
    c = Counted{Float64}(1.0)
    E1 = zeros(Counted{Float64}, Nx, Ny, 3, N_samples)
    B1 = zeros(Counted{Float64}, Nx, Ny, 3, N_samples)
    E2 = mode === Val(:split) ? zeros(Counted{Float64}, Nx, Ny, 3, N_samples) : E1
    B2 = mode === Val(:split) ? zeros(Counted{Float64}, Nx, Ny, 3, N_samples) : B1
    pixel_iter = zeros(Int8, Nx, Ny)
    backend = KernelAbstractions.CPU()
    t_first = x⁰_first - z   # outside the counted block: the kernel entry computes it once per run
    return @count begin
        if alg isa GPUKernelNewton
            _gpu_newton_field_one_electron!(
                mode, E1, B1, E2, B2, gpu_traj, c, x_grid, y_grid, z,
                t_first, δ, N_samples, Nx, Ny, τi, τf, pixel_iter, backend, n)
        else
            _gpu_unified_field_one_electron!(
                mode, E1, B1, E2, B2, gpu_traj, c, x_grid, y_grid, z,
                x⁰_first, δ, N_samples, Nx, Ny, τi, τf, pixel_iter, backend, n)
        end
    end
end

_flop_cats(c::Counts) = NamedTuple{CountedFloats.FLOP_CATEGORIES}(
    ntuple(i -> c[CountedFloats.FLOP_CATEGORIES[i]], length(CountedFloats.FLOP_CATEGORIES)))

# Per-category division: exact for FLOP categories (throws otherwise), rounded for cmp/other,
# whose counts can be data-dependent (`clamp` short-circuits).
function _per(c::Counts, n::Int)
    cats = CountedFloats.CATEGORIES
    vals = map(cats) do cat
        v = c[cat]
        if cat in CountedFloats.FLOP_CATEGORIES
            rem(v, n) == 0 || error("flop_profile: category $cat count $v is not a multiple of $n — \
                the kernel's per-slot arithmetic is no longer uniform")
            div(v, n)
        else
            round(Int, v / n)
        end
    end
    return NamedTuple{cats}(vals)
end

# Per-(electron, pixel) launch cost: what the first launch did beyond its slots, per pixel.
function _per_launch(c1::NamedTuple, per_slot::NamedTuple, S1::Int, npix::Int)
    cats = CountedFloats.CATEGORIES
    vals = map(cats) do cat
        v = c1[cat] - per_slot[cat] * S1
        if cat in CountedFloats.FLOP_CATEGORIES
            rem(v, npix) == 0 || error("flop_profile: per-launch $cat count $v is not a multiple \
                of the pixel count $npix — the kernel's per-pixel setup is no longer uniform")
            div(v, npix)
        else
            round(Int, v / npix)
        end
    end
    return NamedTuple{cats}(vals)
end

_flops(nt::NamedTuple) = nt.add + nt.mul + nt.div + nt.sqrt + 2 * nt.fma + nt.pow + nt.trans

_mode_val(mode::Val) = mode
_mode_val(mode::Symbol) = Val(mode)

"""
    flop_profile(alg::GPUKernelNewton; mode = Val(:split), n_iters = 2) -> NamedTuple
    flop_profile(alg::GPUKernelRK4;    mode = Val(:split), n_substeps = 1) -> NamedTuple

Algorithmic FLOP cost of the `accumulate_field` kernel for `alg` with the given accuracy
knob and `mode` (`Val(:split)` / `Val(:total)`, or the symbol), measured on the host by
running the kernel body on `CountedFloats.Counted{Float64}` (see the file header). Returns

- `flop_per_slot` — FLOPs per (electron, pixel, observer sample), and `per_slot`, its
  per-category breakdown (`add, mul, div, sqrt, fma, pow, trans` exact; `cmp, other` are
  not FLOPs and are rounded means);
- `flop_per_pixel_launch` / `per_pixel_launch` — the per-(electron, pixel) setup cost
  (window edges, RK4 bridge; for RK4 minus the one inter-slot advance the last slot skips);
- `bytes_per_slot` — device-buffer read-modify-write traffic per slot (12 or 6 doubles),
  `arithmetic_intensity = flop_per_slot / bytes_per_slot`;
- `convention`, `alg`, `mode`, `n` (the accuracy knob), `n_name`.

A run's total is `flop_per_slot · slots_executed + flop_per_pixel_launch · N·Nx·Ny`
(`slots_executed` from [`window_coverage`](@ref); `N·N_samples·Nx·Ny` when the window is
fully covered). Costs milliseconds; the GPU is never touched.
"""
function flop_profile(alg::GPUKernelNewton; mode = Val(:split), n_iters::Int = 2)
    n_iters >= 1 || throw(ArgumentError("n_iters must be ≥ 1"))
    return _flop_profile(alg, _mode_val(mode), n_iters, :n_iters)
end
function flop_profile(alg::GPUKernelRK4; mode = Val(:split), n_substeps::Int = 1)
    n_substeps >= 1 || throw(ArgumentError("n_substeps must be ≥ 1"))
    return _flop_profile(alg, _mode_val(mode), n_substeps, :n_substeps)
end

function _flop_profile(alg, mode::Val, n::Int, n_name::Symbol)
    mode === Val(:split) || mode === Val(:total) ||
        throw(ArgumentError("mode must be Val(:split) or Val(:total), got $mode"))
    traj = _profile_trajectory()
    npix = _PROFILE_NPIX^2
    # The Float64 twin of the synthetic problem must be fully covered at every length used.
    grid = LinRange(-_PROFILE_HALFW, _PROFILE_HALFW, _PROFILE_NPIX)
    for N_samples in _PROFILE_NSAMPLES
        screen = ObserverScreen(grid, grid, _PROFILE_Z,
            range(_PROFILE_X0_FIRST; step = _PROFILE_δ, length = N_samples); c = 1.0)
        cov = window_coverage([traj], screen; recount = false)
        cov.ok || error("flop_profile: synthetic window not fully covered at N_samples = $N_samples")
    end
    C = map(N -> _profile_launch(alg, mode, n, N, traj), _PROFILE_NSAMPLES)
    S = map(N -> npix * N, _PROFILE_NSAMPLES)
    Δ1 = C[2] - C[1]
    Δ2 = C[3] - C[2]
    _flop_cats(Δ1) == _flop_cats(Δ2) || error(
        "flop_profile: per-slot FLOPs are not linear in the window length ($(_flop_cats(Δ1)) vs \
        $(_flop_cats(Δ2))) — the kernel has acquired a data-dependent branch; update the profile")
    per_slot = _per(Δ1, S[2] - S[1])
    per_pixel = _per_launch(NamedTuple(C[1]), per_slot, S[1], npix)
    bytes_per_slot = 16 * (mode === Val(:split) ? 12 : 6)   # read + write of 8-byte doubles
    fps = _flops(per_slot)
    return (;
        alg = string(nameof(typeof(alg))), mode = mode === Val(:split) ? :split : :total,
        n, n_name, convention = FLOP_CONVENTION,
        per_slot, per_pixel_launch = per_pixel,
        flop_per_slot = fps, flop_per_pixel_launch = _flops(per_pixel),
        bytes_per_slot, arithmetic_intensity = fps / bytes_per_slot,
    )
end
