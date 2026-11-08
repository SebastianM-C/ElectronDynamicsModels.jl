# Precompute the radiated far-field cube for the animation: the exact
# Liénard–Wiechert far field of all electrons (mode = :split isolates it — the
# near/Coulomb term would visually swamp the emission this close to the disk),
# sampled on the SAME box and frame times as the pulse rendering. A 3D volume is
# just a stack of ObserverScreen z-slices, so the GPU kernel is reused unchanged
# in a loop over slices.
#
# Electrons live inside this volume, so pixels near a trajectory carry 1/R
# spikes — clip at a high percentile at render time, don't fix it here.
#
# ENV knobs:
#   EDM_GPU_BACKEND   cuda (default, local RTX 4080S) | rocm (cloud MI300X) | cpu (tests)
#   EDM_RAD_NT        transverse pixels per side        (default 96)
#   EDM_RAD_NSLICES   slices along propagation          (default 128)
#   EDM_ACCUM_ALG     rk4 | newton: retarded-time solver (default rk4). Newton solves the
#                     light-cone condition per sample, so its cost per sample is n_iters + 1
#                     spline evaluations against 4·n_substeps + 1 for the RK4 march
#   EDM_NEWTON_ITERS  Newton corrections per sample     (default 2; newton only)
#   EDM_NSUBSTEPS     retarded-solve substeps           (default recommended_n_substeps; rk4 only)
#   EDM_SAMPLE_CHUNKS threads per pixel, each walking a slice of the frame samples (default 1).
#                     One slice is only NT² pixels (16k at NT=128) — far fewer threads than a
#                     datacenter GPU holds — so chunks > 1 fill the device (each chunk starts
#                     from a cold light-cone solve).
#   EDM_RAD_FRAME_STRIDE  subsample the frame clock (test strips only, default 1)
#   EDM_BENCH         "1" → time the worst-case mid slice twice (2nd run excludes
#                     kernel compile), print the full-cube estimate, exit
#   EDM_ANIM_GEOMETRY direct (default, setup.jl) | inverse (setup_inverse.jl, whose own
#                     EDM_ANIM_* knobs pick γ, a0, mode, frames)
#   EDM_RAD_BATCHES   "a:b" — compute only batches a…b of the EDM_RAD_BATCH layout (one
#                     process per GPU, e.g. CUDA_VISIBLE_DEVICES=k with disjoint ranges into
#                     the same EDM_OUTDIR/EDM_RUN_TAG). Sealed batch files are skipped, so a
#                     re-run (after a crash or an eviction) resumes where it stopped.
#
# Output: animation/radiation_cube.jls with a NamedTuple
#   (; rad, slice_zs, txs, tys, frame_times)
#   rad :: Float32 (n_frames, NSLICES, NT, NT) = |E_far|² in scene order
#   (X = physics z slices, Y = physics x, Z = physics y) — a frame is rad[f, :, :, :].
# A checkpoint is written every 16 slices (same file + .partial suffix).
#
# Run manifest: with RunManifests + GPUDiagnostics in the env (--project=scripts, as
# run_cell.sh runs it), every run (bench, batched, monolithic) also writes run_<uuid>.toml:
# [config] (kind = "radiation_cube" + every knob), [laser], [setup], [timing], [gpu]
# (sampled telemetry + kernel launch times), [host], [units], [outputs]; see
# radiation_manifest.jl. EDM_RAD_PART=k (set by radiation_multigpu.jl) writes a per-GPU part
# instead, which the wrapper merges into the run's single manifest.
#
# Run: julia +release --startup=no --project=animation animation/precompute_radiation.jl


const GEOMETRY = lowercase(get(ENV, "EDM_ANIM_GEOMETRY", "direct"))
GEOMETRY in ("direct", "inverse") ||
    error("EDM_ANIM_GEOMETRY must be direct or inverse, got $(repr(GEOMETRY))")
const T_SETUP = @elapsed include(joinpath(@__DIR__, GEOMETRY == "inverse" ? "setup_inverse.jl" : "setup.jl"))

using Serialization
using Printf

const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "cuda"))
if GPU_BACKEND == "cuda"
    using CUDA
    const gpu_backend = CUDA.CUDABackend()
elseif GPU_BACKEND == "rocm"
    using AMDGPU
    const gpu_backend = AMDGPU.ROCBackend()
elseif GPU_BACKEND == "cpu"   # tests only: the same kernel on KernelAbstractions' CPU backend
    const gpu_backend = ElectronDynamicsModels.KA.CPU()
else
    error("EDM_GPU_BACKEND must be \"cuda\", \"rocm\" or \"cpu\" (tests), got $(repr(GPU_BACKEND))")
end
include(joinpath(@__DIR__, "radiation_manifest.jl"))

NT = parse(Int, get(ENV, "EDM_RAD_NT", "96"))
NSLICES = parse(Int, get(ENV, "EDM_RAD_NSLICES", "128"))
BENCH = get(ENV, "EDM_BENCH", "0") == "1"
# Production convention (thomson_scattering.jl): async per-electron launches.
# The kernel-level default is true; at animation-scale screens (~16k pixels)
# the device is occupancy-starved per electron, so overlapping the 800
# launches matters even more here than at production size.
SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))

txs = LinRange(first(xs), last(xs), NT)   # the setup's box (±2 w₀ unless EDM_ANIM_HALF_W)
tys = LinRange(first(ys), last(ys), NT)
# Slice extent along propagation, in λ units (defaults = the pulse box). The
# animation's detector plane sits at +16λ; extending ZMAX there makes the cube
# meet the screen with no gap.
ZMIN = parse(Float64, get(ENV, "EDM_RAD_ZMIN_LAMBDA", string(first(zs) / λ))) * λ
ZMAX = parse(Float64, get(ENV, "EDM_RAD_ZMAX_LAMBDA", string(last(zs) / λ))) * λ
slice_zs = LinRange(ZMIN, ZMAX, NSLICES)
# Frame subsetting for cheap controlled tests (slice-density/interpolation
# studies): stride > 1 subsamples the animation frame clock. NOT for production
# cubes — the renderer indexes frames by the full frame_times.
FRAME_STRIDE = parse(Int, get(ENV, "EDM_RAD_FRAME_STRIDE", "1"))
if FRAME_STRIDE > 1
    frame_times = frame_times[1:FRAME_STRIDE:end]
    n_frames = length(frame_times)
end
x⁰_samples = c .* frame_times

slice_screen(z) = ObserverScreen(txs, tys, z, x⁰_samples; c)

NSUB = let s = get(ENV, "EDM_NSUBSTEPS", "")
    isempty(s) ? recommended_n_substeps(slice_screen(first(slice_zs))) : parse(Int, s)
end
SAMPLE_CHUNKS = parse(Int, get(ENV, "EDM_SAMPLE_CHUNKS", "1"))
const ACCUM_ALG = lowercase(get(ENV, "EDM_ACCUM_ALG", "rk4"))
ACCUM_ALG in ("rk4", "newton") || error("EDM_ACCUM_ALG must be rk4 or newton, got $(repr(ACCUM_ALG))")
NEWTON_ITERS = parse(Int, get(ENV, "EDM_NEWTON_ITERS", "2"))
const SOLVER_ALG = ACCUM_ALG == "newton" ? GPUKernelNewton() : GPUKernelRK4()
const SOLVER_KW = ACCUM_ALG == "newton" ? (; n_iters = NEWTON_ITERS) : (; n_substeps = NSUB)
@info "radiation precompute" GPU_BACKEND GEOMETRY NT NSLICES n_frames ACCUM_ALG NSUB NEWTON_ITERS SAMPLE_CHUNKS n_electrons = length(trajs)

# EDM_RAD_BATCH > 0: stream the cube out as per-batch slice files instead of
# one monolith — each completed batch is serialized (atomic .tmp → rename) and
# its buffer reused, so (a) a local puller can download batches WHILE later
# slices compute, (b) the VM never holds the whole cube in RAM: cube size is
# capped by budget, not memory. Assemble locally with
# animation/assemble_radiation_cube.jl. 0 = legacy single-file output.
BATCH = parse(Int, get(ENV, "EDM_RAD_BATCH", "0"))
# Orchestration-aware output: run_cell.sh sets EDM_OUTDIR (campaign dir) and
# EDM_RUN_TAG (uuid); a direct local run keeps the plain name in animation/.
OUTDIR = get(ENV, "EDM_OUTDIR", @__DIR__)
RUN_TAG = get(ENV, "EDM_RUN_TAG", "")
PART = let s = get(ENV, "EDM_RAD_PART", ""); isempty(s) ? nothing : parse(Int, s) end
outfile = joinpath(OUTDIR, isempty(RUN_TAG) ? "radiation_cube.jls" : "radiation_cube_$(RUN_TAG).jls")
mkpath(OUTDIR)
nb = BATCH > 0 ? cld(NSLICES, BATCH) : 0
mine = let s = get(ENV, "EDM_RAD_BATCHES", "")
    BATCH == 0 && !isempty(s) && @warn "EDM_RAD_BATCHES=$s is ignored without EDM_RAD_BATCH > 0"
    (BATCH == 0 || isempty(s)) ? (1:nb) : (let (a, b) = parse.(Int, split(s, ':')); max(a, 1):min(b, nb) end)
end
# Atomic replace: mv(; force = true) deletes the destination before renaming, so a concurrent
# reader could find it missing for a moment.
atomic_mv(src, dst) = isdefined(Base.Filesystem, :rename) ?
    Base.Filesystem.rename(src, dst) : mv(src, dst; force = true)
tagged(prefix, ext) = joinpath(OUTDIR, string(prefix, "_", isempty(RUN_TAG) ? "local" : RUN_TAG,
    PART === nothing ? "" : "_gpu$(PART)", ext))
gputracefile = tagged("gputrace", ".tsv")

# |E_far|² AND the signed polarization component Ex_far per (t, x, y), Float32.
# The intensity drives the emission envelope; the signed component carries the
# wavefront phase (± isosurfaces = phase-striped radiation shells, matching the
# pulse's ribbons). Both come from the same kernel output — storing Ex is free.
const launch_timer = new_launch_timer()
function far_fields(screen; timer = launch_timer)
    fld = accumulate_field(
        trajs, screen, SOLVER_ALG, gpu_backend;
        SOLVER_KW..., sample_chunks = SAMPLE_CHUNKS, mode = Val(:split),
        sync_per_electron = SYNC, timer,
    )
    intens = Float32.(dropdims(sum(abs2, fld.E_far; dims = 2); dims = 2))
    signed = Float32.(fld.E_far[:, 1, :, :])
    return intens, signed
end

# ── Manifest sections (plain TOML values only) ──
const IS_INVERSE = GEOMETRY == "inverse"
# Keys follow the solver manifests' vocabulary wherever the meaning is the same, so the
# dashboard's canonical params (PARAM_SPEC) pick them up unchanged: Nx/Ny = screen pixels per
# side (NT), N_samples = time samples per pixel (the frames), N = electrons, scattering =
# "inverse" and gamma only on the inverse geometry (rest-electron manifests carry neither),
# [laser] m/p/profile/temporal_width/pol, [setup] Rmax. The cube-only knobs keep their own names.
function manifest_config()
    d = Dict{String, Any}(
        "kind" => "radiation_cube", "bench" => BENCH, "geometry" => GEOMETRY,
        "mode" => ANIM_MODE, "a0" => a₀, "N" => length(trajs),
        "Nx" => NT, "Ny" => NT, "N_samples" => n_frames, "n_slices" => NSLICES,
        "frame_stride" => FRAME_STRIDE,
        "t_start_T0" => first(frame_times) / T0, "t_end_T0" => last(frame_times) / T0,
        "z_min_lambda" => ZMIN / λ, "z_max_lambda" => ZMAX / λ,
        "transverse_halfwidth_w0" => 2.0, "sample_chunks" => SAMPLE_CHUNKS, "sync_per_electron" => SYNC,
        "field_mode" => "split", "accumulation_alg" => string(nameof(typeof(SOLVER_ALG))),
        "n_substeps" => NSUB, "newton_iters" => NEWTON_ITERS,   # both always, as the solver manifests do
        "batch" => BATCH, "n_batches" => nb, "batches" => BATCH > 0 ? "$(first(mine)):$(last(mine))" : "",
    )
    if IS_INVERSE
        d["scattering"] = "inverse"
        d["gamma"] = GAMMA
    end
    return d
end
manifest_laser() = Dict{String, Any}(
    "wavelength" => λ, "omega" => ω, "w0" => w₀, "a0" => a₀,
    "p" => p_radial, "m" => m_azimuthal, "pol" => string(pol),
    "profile" => "gaussian", "temporal_width" => τ,
    "initial_phase" => ϕ₀, "focus_position" => 0.0,
    "k_direction" => IS_INVERSE ? "[0, 0, -1]" : "[0, 0, 1]",
)
manifest_setup() = Dict{String, Any}(
    "Rmax" => Rmax, "electron_disk_rmax_w0" => Rmax / w₀, "electron_layout" => "sunflower",
    "tau_span" => [τi, τf], "backscatter_n0" => IS_INVERSE ? DOPPLER : 1.0,
    "saveat_knots_per_period" => IS_INVERSE ? SAVEAT_KNOTS : 16.0,
)
# Everything a cube's numbers depend on (the physics config, laser and setup, without the
# bookkeeping keys). A resume only reuses sealed batches written under the same fingerprint.
function cube_fingerprint()
    c = manifest_config()
    foreach(k -> delete!(c, k), ("bench", "batches", "sync_per_electron"))
    for (sec, d) in (("laser", manifest_laser()), ("setup", manifest_setup())), (k, v) in d
        c["$sec.$k"] = v
    end
    return repr(sort!(collect(c); by = first))
end
function finish_manifest(timing, outputs, telem)
    isfile(gputracefile) && (outputs["gpu_trace"] = basename(gputracefile))
    p = try
        write_radiation_manifest(; outdir = OUTDIR, run_tag = RUN_TAG, part = PART,
            config = manifest_config(), laser = manifest_laser(), setup = manifest_setup(),
            timing, outputs, backend = gpu_backend, backend_name = GPU_BACKEND,
            n_threads = NT * NT, telem, timer = launch_timer,
            units = units_section(ω, λ, w₀; n0 = IS_INVERSE ? DOPPLER : 1.0))
    catch err
        @warn "run manifest failed — the cube products are unaffected" exception = (err, catch_backtrace())
        nothing
    end
    p === nothing || @info "manifest → $p"
    return p
end

if BENCH
    # Mid slice = worst case (contains the electron disk). First call compiles
    # the kernel; the second is the honest per-slice cost. Only the second runs under the
    # launch timer, so [timing].kernel (and the dashboard's rate) covers exactly one slice,
    # and [timing].field is that slice's wall time (the solver manifests' key for it).
    mid = slice_screen(slice_zs[NSLICES ÷ 2 + 1])
    (t1, t2), telem = sampled(gpu_backend, gputracefile) do
        (@elapsed far_fields(mid; timer = nothing)), (@elapsed far_fields(mid))
    end
    total = t2 * NSLICES
    @info @sprintf(
        "bench: slice %.1fs (compile run %.1fs) → full cube ≈ %.1f min on this GPU",
        t2, t1, total / 60
    )
    timing = merge(Dict{String, Any}("setup" => T_SETUP, "compile_slice" => t1, "field" => t2,
        "estimate_full_cube" => total, "total" => t1 + t2), slice_stats([t2]))
    finish_manifest(timing, Dict{String, Any}(), telem)
    exit(0)
end

slice_times = Float64[]
if BATCH == 0
    rad = Array{Float32}(undef, n_frames, NSLICES, NT, NT)
    rad_s = Array{Float32}(undef, n_frames, NSLICES, NT, NT)
end

if BATCH > 0
    base = replace(outfile, r"\.jls$" => "")
    batch_file(bi) = @sprintf("%s_batch%03d.jls", base, bi)
    # Several processes may share this output: each writes the (identical) manifest
    # through its own tmp name and an atomic rename.
    mf = "$(base)_manifest.jls"
    fingerprint = cube_fingerprint()
    sealed = count(bi -> isfile(batch_file(bi)), 1:nb)
    if sealed > 0
        old = isfile(mf) ? get(deserialize(mf), :fingerprint, nothing) : nothing
        old == fingerprint || error("$sealed sealed batch file(s) $(base)_batch*.jls were written " *
            "under a different configuration (or before configs were fingerprinted); move them " *
            "away or use a new EDM_RUN_TAG")
    end
    serialize("$mf.$(getpid()).tmp",
        (; slice_zs, txs, tys, frame_times, NSLICES, NT, batch = BATCH, n_batches = nb,
            geometry = GEOMETRY, fingerprint))
    atomic_mv("$mf.$(getpid()).tmp", mf)
    todo = filter(bi -> !isfile(batch_file(bi)), mine)
    length(todo) < length(mine) &&
        @info "resuming: $(length(mine) - length(todo)) of $(length(mine)) batches already sealed"
    buf = Array{Float32}(undef, n_frames, BATCH, NT, NT)
    buf_s = Array{Float32}(undef, n_frames, BATCH, NT, NT)
    t_total, telem = sampled(gpu_backend, gputracefile) do
        @elapsed for bi in todo
            lo, hi = (bi - 1) * BATCH + 1, min(bi * BATCH, NSLICES)
            for (k, si) in enumerate(lo:hi)
                t_slice = @elapsed begin
                    intens, signed = far_fields(slice_screen(slice_zs[si]))
                    buf[:, k, :, :] .= intens
                    buf_s[:, k, :, :] .= signed
                end
                push!(slice_times, t_slice)
                @info @sprintf("slice %3d/%d (X = %+6.1fλ)  %.1fs", si, NSLICES, slice_zs[si] / λ, t_slice)
            end
            n = hi - lo + 1
            bf = batch_file(bi)
            serialize(bf * ".tmp", (; rad = buf[:, 1:n, :, :], rad_s = buf_s[:, 1:n, :, :],
                slices = lo:hi))
            atomic_mv(bf * ".tmp", bf)
            @info "batch $bi/$nb sealed → $bf"
        end
    end
    # _done means the WHOLE cube is sealed, whichever process finishes last
    all(bi -> isfile(batch_file(bi)), 1:nb) && touch("$(base)_done")
    @info @sprintf("radiation batches %s done in %.1f min → %s_batch*.jls (%d batches in the cube)",
        string(mine), t_total / 60, base, nb)
    finish_manifest(
        merge(Dict{String, Any}("setup" => T_SETUP, "total" => t_total,
            "batches_resumed" => length(mine) - length(todo)), slice_stats(slice_times)),
        Dict{String, Any}("manifest" => basename(mf),
            "batches" => [basename(batch_file(bi)) for bi in mine if isfile(batch_file(bi))]),
        telem)
else
    t_total, telem = sampled(gpu_backend, gputracefile) do
        @elapsed for (si, zsl) in enumerate(slice_zs)
            t_slice = @elapsed begin
                intens, signed = far_fields(slice_screen(zsl))
                rad[:, si, :, :] .= intens
                rad_s[:, si, :, :] .= signed
            end
            push!(slice_times, t_slice)
            @info @sprintf("slice %3d/%d (X = %+6.1fλ)  %.1fs", si, NSLICES, zsl / λ, t_slice)
            if si % 16 == 0 && si < NSLICES
                serialize(outfile * ".partial", (; rad, rad_s, slice_zs, txs, tys, frame_times, done = si))
            end
        end
    end
    serialize(outfile, (; rad, rad_s, slice_zs, txs, tys, frame_times))
    rm(outfile * ".partial"; force = true)
    @info @sprintf("radiation cube done in %.1f min → %s (%.2f GB)", t_total / 60, outfile,
        (sizeof(rad) + sizeof(rad_s)) / 1e9)
    finish_manifest(
        merge(Dict{String, Any}("setup" => T_SETUP, "total" => t_total), slice_stats(slice_times)),
        Dict{String, Any}("datafile" => basename(outfile)), telem)
end
