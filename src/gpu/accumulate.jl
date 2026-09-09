"""
    accumulate_potential(trajs, screen, alg, backend::Backend; solve_kwargs...)

Compute the Liénard-Wiechert 4-potential using CPU retarded-time solve
and GPU-accelerated accumulation via AcceleratedKernels.

`backend` is a KernelAbstractions backend (e.g., `CUDA.CUDABackend()`).
Uses the original `trajs` (CubicSpline-based) for the CPU retarded-time solve,
and converts to `GPUCubicSpline` internally for the GPU accumulation phase.
"""
function accumulate_potential(
        trajs::Vector{<:TrajectoryInterpolant},
        screen::ObserverScreen, alg, backend::Backend;
        solve_kwargs...
    )
    x⁰_samples = screen.x⁰_samples
    N_samples = length(x⁰_samples)
    N_x⁰ = N_samples
    x⁰_first = first(x⁰_samples)
    δx⁰_step = (last(x⁰_samples) - x⁰_first) / (N_samples - 1)
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)

    A = zeros(N_samples, 4, Nx, Ny)

    # Pre-allocate GPU buffers (reused across electrons)
    τ_buf = Adapt.adapt(backend, fill(NaN, N_samples, Nx, Ny))
    A_buf = Adapt.adapt(backend, zeros(N_samples, 4, Nx, Ny))

    τ_all = fill(NaN, N_samples, Nx, Ny)

    # Create typed integrator pool once (reused across all electrons)
    traj0 = first(trajs)
    τi0 = first(traj0.itp.t)
    τf0 = last(traj0.itp.t)
    r_obs_0 = SVector{3}(screen.x_grid[1], screen.y_grid[1], screen.z)
    proto_prob = ODEProblem{false, SciMLBase.FullSpecialize}(
        retarded_time_rhs, τi0,
        (advanced_time(traj0, τi0, r_obs_0), advanced_time(traj0, τf0, r_obs_0)),
        (traj0, r_obs_0)
    )
    proto_integ = init(proto_prob, alg; saveat = x⁰_samples, save_start = false, save_end = false, solve_kwargs...)
    nworkers = Threads.nthreads()
    integ_pool = Channel{typeof(proto_integ)}(nworkers)
    put!(integ_pool, proto_integ)
    for _ in 2:nworkers
        put!(integ_pool, init(proto_prob, alg; saveat = x⁰_samples, save_start = false, save_end = false, solve_kwargs...))
    end

    # Per-electron streaming: upload one trajectory's spline at a time and
    # free it after the kernel completes.  Memory footprint is bounded by
    # `A_buf + τ_buf + one trajectory + small`, independent of N_macro.
    for traj in trajs
        τi = first(traj.itp.t)
        τf = last(traj.itp.t)

        # ── Phase 1: CPU retarded-time solve ──
        fill!(τ_all, NaN)

        Threads.@threads for ix in Base.OneTo(Nx)
            integ = take!(integ_pool)
            for iy in Base.OneTo(Ny)
                r_obs = SVector{3}(screen.x_grid[ix], screen.y_grid[iy], screen.z)
                x⁰_i = advanced_time(traj, τi, r_obs)
                x⁰_f = advanced_time(traj, τf, r_obs)

                integ.p = (traj, r_obs)
                reinit!(integ, τi; t0 = x⁰_i, tf = x⁰_f)
                solve!(integ)

                # Map saveat values back to τ_all's first-axis index. When
                # the per-pixel tspan is much narrower than x⁰_samples, the
                # integrator's sol.u contains only the saveat points within
                # [x⁰_i, x⁰_f] — a subset, possibly indexed in the middle of
                # x⁰_samples. Sequential k=1,2,… would misroute the data.
                sol_u = integ.sol.u::Vector{Float64}
                sol_t = integ.sol.t::Vector{Float64}
                for k in eachindex(sol_u)
                    idx = round(Int, (sol_t[k] - x⁰_first) / δx⁰_step) + 1
                    1 ≤ idx ≤ N_x⁰ || continue
                    @inbounds τ_all[idx, ix, iy] = sol_u[k]
                end
            end
            put!(integ_pool, integ)
        end

        # ── Phase 2: upload this trajectory + GPU accumulation ──
        gpu_traj = Adapt.adapt(backend, to_gpu(traj))
        _gpu_accumulate_kernel!(gpu_traj, screen, τ_all, τ_buf, A_buf, backend)

        # Stream-aware async free: CUDA.jl queues the free behind the kernel
        # that still reads these buffers, so no explicit sync is needed —
        # syncing here would serialize CPU Phase-1(N+1) against GPU kernel(N)
        # and kill the pipeline.  The trailing `Array(A_buf)` syncs on exit.
        finalize(gpu_traj.itp.t)
        finalize(gpu_traj.itp.h)
        finalize(gpu_traj.itp.z)
        finalize(gpu_traj.itp.c1)
        finalize(gpu_traj.itp.c2)
    end

    # Single D2H transfer at the end
    copyto!(A, Array(A_buf))
    return A
end

function _gpu_accumulate_kernel!(gpu_traj, screen, τ_all_cpu, τ_buf, A_buf, backend)
    # Copy retarded times to pre-allocated GPU buffer (H2D)
    copyto!(τ_buf, τ_all_cpu)

    # Capture screen grids (isbits LinRange — no adaptation needed)
    x_grid = screen.x_grid
    y_grid = screen.y_grid
    z_screen = screen.z

    # Capture trajectory components
    spline = gpu_traj.itp
    K = gpu_traj.K

    # Pre-compute CartesianIndices outside closure
    CI = CartesianIndices(τ_buf)

    # Kernel accumulates into A_buf (no zeroing — accumulates across electrons)
    AK.foreachindex(τ_buf, backend) do i
        k, ix, iy = Tuple(CI[i])

        τ = τ_buf[k, ix, iy]
        isnan(τ) && return

        v = spline(τ)
        xμ = SVector{4}(v[1], v[2], v[3], v[4])   # canonical state order (see to_gpu)
        uμ = SVector{4}(v[5], v[6], v[7], v[8])

        r_obs = SVector{3}(x_grid[ix], y_grid[iy], z_screen)
        disp = r_obs - xμ[SA[2, 3, 4]]
        xr = SVector{4}(norm(disp), disp[1], disp[2], disp[3])

        coeff = K / m_dot(xr, uμ)
        for j in Base.OneTo(4)
            @inbounds A_buf[k, j, ix, iy] += coeff * uμ[j]
        end
    end
    return
end

# ── Launch shape: pixels × observer-sample chunks ──────────────────────────────────────────
# One thread per (pixel, chunk): the thread walks the chunk's slice of the pixel's executed
# slot range. `n_chunks = 1` is one thread per pixel walking every slot (the original grid).
# Chunk-major decomposition keeps adjacent threads on adjacent pixels of the same chunk, so
# the accumulator writes stay coalesced; chunks own disjoint slots, so no atomics are needed.
# Threads of chunks > 1 start cold: their first slot is solved with N_COLD_ITERS safeguarded
# Newton corrections (see `_bracketed_slot_solve`) instead of the one-slot warm start.
const N_COLD_ITERS = 24

@inline function _chunk_pixel(i_lin, Nx, Ny, n_chunks)
    npx = Nx * Ny
    p = (i_lin - 1) % npx
    chunk = (i_lin - 1) ÷ npx + 1
    ix = p % Nx + 1
    iy = p ÷ Nx + 1
    return ix, iy, chunk
end

# The chunk's slice of the pixel's executed slots `k_start:k_end` (near-even contiguous split;
# the first `rem` chunks get one extra slot; empty when the range has fewer slots than chunks).
@inline function _chunk_slots(k_start, k_end, chunk, n_chunks)
    len = k_end - k_start + 1
    base, rem = divrem(len, n_chunks)
    c_start = k_start + (chunk - 1) * base + min(chunk - 1, rem)
    c_end = c_start + base + (chunk <= rem ? 1 : 0) - 1
    return c_start, c_end
end

"""
    _download_permuted(buf; backend = nothing, dev = 0, workers = 1) -> Array{T,4}

Download an accumulation buffer laid out `[ix, iy, μ, k]` (pixel-fastest for coalesced
device writes; observer slot `k` slowest) into the cube layout `[k, μ, ix, iy]` without a
full-size intermediate. The old `permutedims(Array(buf), (4, 3, 1, 2))` held cube + copy on
the host — a 2× peak that capped carrier-resolved cube designs at ~half the node's RAM (the
permute lives on the host because the >2³¹-element GPU `permutedims` overflows its 32-bit
linear indices, cudaError 700). Downloading contiguous `k`-chunks through the linear
`copyto!` DMA path and `permutedims!`-ing each into a view of the preallocated cube keeps
the host peak at ~(1 + 1/16)× cube; device memory and kernels are untouched, and every
transfer stays far below 2³¹ elements. Works unchanged on the CPU backend (`buf::Array`).

`workers > 1` spreads the 16 chunks over that many tasks, each with its own staging slab
(host peak ~(1 + workers/16)× cube): the download DMA and the single-threaded
`permutedims!` of the chunks then overlap. The tasks pin themselves to `dev` through
`gpu_device!(backend, dev)`, so `backend` and `dev` are required for that path.
"""
function _download_permuted(buf::AbstractArray{T, 4}; backend = nothing, dev::Integer = 0,
        workers::Integer = 1) where {T}
    Nx, Ny, M, K = size(buf)
    out = Array{T, 4}(undef, K, M, Nx, Ny)
    chunk = max(1, cld(K, 16))
    bufv = vec(buf)
    slab = Nx * Ny * M
    starts = collect(1:chunk:K)
    permute_chunk!(stage, k0) = begin
        nk = min(chunk, K - k0 + 1)
        copyto!(stage, 1, bufv, (k0 - 1) * slab + 1, nk * slab)
        permutedims!(
            view(out, k0:(k0 + nk - 1), :, :, :),
            reshape(view(stage, 1:(nk * slab)), Nx, Ny, M, nk), (4, 3, 1, 2),
        )
    end
    if workers <= 1 || length(starts) == 1 || backend === nothing
        stage = Array{T}(undef, slab * chunk)
        foreach(k0 -> permute_chunk!(stage, k0), starts)
        return out
    end
    queue = Channel{Int}(length(starts))
    foreach(k0 -> put!(queue, k0), starts)
    close(queue)
    @sync for _ in 1:min(workers, length(starts))
        Threads.@spawn begin
            gpu_device!(backend, dev)
            # `local`: the serial branch above binds a function-level `stage`; without it every
            # worker would assign that one shared (boxed) variable and race on a single slab.
            local mystage = Array{T}(undef, slab * chunk)
            for k0 in queue
                permute_chunk!(mystage, k0)
            end
        end
    end
    return out
end

"""
    _download_permuted_add!(out, buf) -> out

In-place sibling of [`_download_permuted`](@ref): download `buf` (`[ix, iy, μ, k]`) chunk by
chunk and ADD it into the preallocated host cube `out` (`[k, μ, ix, iy]`). Two staging
buffers of 1/16 cube (the linear download slab and its permuted copy) are the only
transient, so folding a device partial into a running host sum costs ~(1/8)× cube of
extra host memory — the primitive behind the streamed multi-device reduce, which keeps ONE
host cube resident instead of one per device.
"""
function _download_permuted_add!(out::Array{T, 4}, buf::AbstractArray{T, 4}) where {T}
    Nx, Ny, M, K = size(buf)
    size(out) == (K, M, Nx, Ny) || throw(DimensionMismatch(
        "_download_permuted_add!: out $(size(out)) does not match buf $(size(buf)) permuted"))
    chunk = max(1, cld(K, 16))
    stage = Array{T}(undef, Nx * Ny * M * chunk)
    pstage = Array{T, 4}(undef, chunk, M, Nx, Ny)
    bufv = vec(buf)
    slab = Nx * Ny * M
    for k0 in 1:chunk:K
        nk = min(chunk, K - k0 + 1)
        copyto!(stage, 1, bufv, (k0 - 1) * slab + 1, nk * slab)
        pv = view(pstage, 1:nk, :, :, :)
        permutedims!(pv, reshape(view(stage, 1:(nk * slab)), Nx, Ny, M, nk), (4, 3, 1, 2))
        view(out, k0:(k0 + nk - 1), :, :, :) .+= pv
    end
    return out
end

# ── Field-cube collection: the shared tail of the RK4/Newton `accumulate_field` kernels ──
# `_collect_fields` is the default (download every buffer to a fresh host NamedTuple);
# `_add_fields!` folds the device buffers into an existing host NamedTuple of the same
# shape. `accumulate_field`'s `sink` kwarg picks between them: `nothing` ⇒ collect and
# return; a callable ⇒ it receives `(E1, B1, E2, B2, mode)` device buffers while they are
# still alive and its return value is what `accumulate_field` returns (the sharded driver
# passes a sink that locks and `_add_fields!`s, so partials never coexist on the host).
function _collect_fields(E1_buf, B1_buf, E2_buf, B2_buf, mode::Val; backend = nothing, dev::Integer = 0,
        workers::Integer = 1)
    dl(buf) = _download_permuted(buf; backend, dev, workers)
    if mode == Val(:split)
        E_far = dl(E1_buf)
        B_far = dl(B1_buf)
        E_near = dl(E2_buf)
        B_near = dl(B2_buf)
        E = E_far .+ E_near
        B = B_far .+ B_near
        return (; E, B, E_far, B_far)
    else
        E = dl(E1_buf)
        B = dl(B1_buf)
        return (; E, B)
    end
end

function _add_fields!(acc::NamedTuple, E1_buf, B1_buf, E2_buf, B2_buf, mode::Val)
    if mode == Val(:split)
        haskey(acc, :E_far) || throw(ArgumentError("_add_fields!: split-mode buffers need an accumulator with E_far/B_far"))
        _download_permuted_add!(acc.E_far, E1_buf)
        _download_permuted_add!(acc.B_far, B1_buf)
        _download_permuted_add!(acc.E, E1_buf)
        _download_permuted_add!(acc.E, E2_buf)
        _download_permuted_add!(acc.B, B1_buf)
        _download_permuted_add!(acc.B, B2_buf)
    else
        _download_permuted_add!(acc.E, E1_buf)
        _download_permuted_add!(acc.B, B1_buf)
    end
    return acc
end

_finish_fields(sink::Nothing, E1_buf, B1_buf, E2_buf, B2_buf, mode::Val) =
    _collect_fields(E1_buf, B1_buf, E2_buf, B2_buf, mode)
_finish_fields(sink, E1_buf, B1_buf, E2_buf, B2_buf, mode::Val) =
    sink(E1_buf, B1_buf, E2_buf, B2_buf, mode)

# ── Persistent device accumulation buffers: electron batching ──────────────────────────────
# The host cost of a field run is dominated by the trajectory splines: at 16 knots per proper-time
# period a production electron carries ~7.4 MB of state + acceleration coefficients, so solving
# every electron up front and keeping its splines for the whole field phase costs ~7.4 MB × N — at
# N = 16 000 that is ~118 GB, on top of one cube copy at the download. Nothing in the accumulation
# needs them all at once: the electron loop uploads ONE trajectory at a time and the device buffers
# are the only state that must live across electrons. `FieldAccumulator` makes those buffers
# outlive a single `accumulate_field` call, so a driver can solve a batch, accumulate it, drop its
# splines, and continue — with the download still happening exactly once, at the end.

"""
    FieldAccumulator(screen, backend; mode = Val(:split))

The device-resident accumulation buffers of [`accumulate_field`](@ref), held as an object so that
they survive one call and several calls can sum into the same buffers.

By default (`buffers = nothing`, `finish = true`) `accumulate_field` allocates a set per call and
downloads it at the end — the single-call API, unchanged. With `finish = false` it instead returns
the `FieldAccumulator` holding that call's electrons, and passing it back as `buffers` on the next
call continues the accumulation in place; the cube is downloaded once, by the call that runs with
`finish = true` (or by [`finish_field`](@ref)).

This is the memory model behind the solver scripts' `EDM_ELECTRON_BATCH` knob: the host then holds
one batch of trajectory splines instead of every electron's for the whole field phase, while the
device keeps the same single buffer set as an unbatched run. The kernels, the per-electron uploads
and the launch order are untouched, so on one device a batched run adds the same per-electron
contributions in the same order as the single call (bit-identical); across devices the shard
composition changes with the batching, which moves the sum by the last bits.

Fields: `E1`, `B1` (far field in `:split`, total in `:total`), `E2`, `B2` (near field; aliases of
`E1`, `B1` in `:total`, where the kernel sums far + near before the write and never touches them),
the `mode`, the `dev`ice the buffers live on, and the running `n_electrons` count.

```julia
acc = accumulate_field(trajs[1:500], screen, alg, backend; finish = false)
acc = accumulate_field(trajs[501:1000], screen, alg, backend; buffers = acc, finish = false)
fld = accumulate_field(trajs[1001:1500], screen, alg, backend; buffers = acc)   # downloads once
```
"""
mutable struct FieldAccumulator{B, A, M}
    backend::B
    dev::Int
    E1::A
    B1::A
    E2::A
    B2::A
    mode::M
    n_electrons::Int
end

# The device the buffers were allocated on, when the backend can say. A backend with no vendor
# extension loaded (`gpu_device` errors) is still a perfectly good accumulation target for the
# single-device path, so this must not be the thing that makes it fail: 0 = "unknown", used only
# by the sharded driver, which needs the vendor API anyway.
_accumulator_device(backend) = try
    Int(gpu_device(backend))
catch
    0
end

function FieldAccumulator(screen::ObserverScreen, backend::Backend; mode::Val = Val(:split))
    mode == Val(:split) || mode == Val(:total) ||
        throw(ArgumentError("FieldAccumulator: mode must be Val(:split) or Val(:total), got $mode"))
    Nx, Ny = length(screen.x_grid), length(screen.y_grid)
    N_samples = length(screen.x⁰_samples)
    # Pixel-fastest accumulators for coalesced writes; `:total` collapses far+near in the kernel
    # into a single (E, B) pair (2 buffers instead of 4), halving device memory.
    E1 = Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples))
    B1 = Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples))
    E2 = mode == Val(:split) ? Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples)) : E1
    B2 = mode == Val(:split) ? Adapt.adapt(backend, zeros(Nx, Ny, 3, N_samples)) : B1
    return FieldAccumulator(backend, _accumulator_device(backend), E1, B1, E2, B2, mode, 0)
end

# `buffers` resolution shared by the `accumulate_field` methods: nothing ⇒ a fresh set (the
# single-call path), an accumulator ⇒ reuse it after checking it matches this screen and mode.
_field_buffers(::Nothing, screen::ObserverScreen, backend::Backend, mode::Val) =
    FieldAccumulator(screen, backend; mode)

function _field_buffers(acc::FieldAccumulator, screen::ObserverScreen, backend::Backend, mode::Val)
    acc.mode === mode || throw(ArgumentError(
        "accumulate_field: buffers were allocated with mode = $(acc.mode), called with mode = $mode"))
    want = (length(screen.x_grid), length(screen.y_grid), 3, length(screen.x⁰_samples))
    size(acc.E1) == want || throw(DimensionMismatch(
        "accumulate_field: buffers are $(size(acc.E1)), this screen needs $want"))
    return acc
end

_field_buffers(x, ::ObserverScreen, ::Backend, ::Val) = throw(ArgumentError(
    "accumulate_field: `buffers` must be a FieldAccumulator or nothing, got $(typeof(x))"))

"""
    finish_field(acc::FieldAccumulator; sink = nothing, workers = 1) -> (; E, B[, E_far, B_far])

Download the accumulated cube from a [`FieldAccumulator`](@ref) without adding more electrons —
the explicit form of `accumulate_field(no_more_electrons, …; buffers = acc)`. `workers > 1`
spreads the download's permute over that many tasks (see `_download_permuted`). `sink` receives
the device buffers instead, exactly as in `accumulate_field`.

The accumulator keeps its buffers (and its sum) afterwards; call it again, or keep accumulating
into it, if that is what the driver wants.
"""
function finish_field(acc::FieldAccumulator; sink = nothing, workers::Integer = 1)
    sink === nothing || return sink(acc.E1, acc.B1, acc.E2, acc.B2, acc.mode)
    # The threaded permute pins its tasks with `gpu_device!`, so it needs a known device.
    w = acc.dev == 0 ? 1 : workers
    return _collect_fields(acc.E1, acc.B1, acc.E2, acc.B2, acc.mode;
        backend = acc.backend, dev = acc.dev, workers = w)
end
