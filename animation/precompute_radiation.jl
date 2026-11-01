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
#   EDM_NSUBSTEPS     retarded-solve substeps           (default recommended_n_substeps)
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

txs = LinRange(-2w₀, 2w₀, NT)
tys = LinRange(-2w₀, 2w₀, NT)
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
@info "radiation precompute" GPU_BACKEND GEOMETRY NT NSLICES n_frames NSUB n_electrons = length(trajs)

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
    (BATCH == 0 || isempty(s)) ? (1:nb) : (let (a, b) = parse.(Int, split(s, ':')); a:min(b, nb) end)
end
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
        trajs, screen, GPUKernelRK4(), gpu_backend;
        n_substeps = NSUB, mode = Val(:split), sync_per_electron = SYNC, timer,
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
        "mode" => IS_INVERSE ? ANIM_MODE : "hero", "a0" => a₀, "N" => length(trajs),
        "Nx" => NT, "Ny" => NT, "N_samples" => n_frames, "n_slices" => NSLICES,
        "frame_stride" => FRAME_STRIDE,
        "t_start_T0" => first(frame_times) / T0, "t_end_T0" => last(frame_times) / T0,
        "z_min_lambda" => ZMIN / λ, "z_max_lambda" => ZMAX / λ,
        "transverse_halfwidth_w0" => 2.0, "n_substeps" => NSUB, "sync_per_electron" => SYNC,
        "field_mode" => "split", "accumulation_alg" => "GPUKernelRK4",
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
    serialize("$mf.$(getpid()).tmp",
        (; slice_zs, txs, tys, frame_times, NSLICES, NT, batch = BATCH, n_batches = nb,
            geometry = GEOMETRY))
    mv("$mf.$(getpid()).tmp", mf; force = true)
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
            mv(bf * ".tmp", bf; force = true)
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
