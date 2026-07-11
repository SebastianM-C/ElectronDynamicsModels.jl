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
#   EDM_GPU_BACKEND   cuda (default, local RTX 4080S) | rocm (cloud MI300X)
#   EDM_RAD_NT        transverse pixels per side        (default 96)
#   EDM_RAD_NSLICES   slices along propagation          (default 128)
#   EDM_NSUBSTEPS     retarded-solve substeps           (default recommended_n_substeps)
#   EDM_BENCH         "1" → time the worst-case mid slice twice (2nd run excludes
#                     kernel compile), print the full-cube estimate, exit
#
# Output: animation/radiation_cube.jls with a NamedTuple
#   (; rad, slice_zs, txs, tys, frame_times)
#   rad :: Float32 (n_frames, NSLICES, NT, NT) = |E_far|² in scene order
#   (X = physics z slices, Y = physics x, Z = physics y) — a frame is rad[f, :, :, :].
# A checkpoint is written every 16 slices (same file + .partial suffix).
#
# Run: julia +release --startup=no --project=animation animation/precompute_radiation.jl

include(joinpath(@__DIR__, "setup.jl"))

using Serialization
using Printf

const GPU_BACKEND = lowercase(get(ENV, "EDM_GPU_BACKEND", "cuda"))
if GPU_BACKEND == "cuda"
    using CUDA
    const gpu_backend = CUDA.CUDABackend()
elseif GPU_BACKEND == "rocm"
    using AMDGPU
    const gpu_backend = AMDGPU.ROCBackend()
else
    error("EDM_GPU_BACKEND must be \"cuda\" or \"rocm\", got $(repr(GPU_BACKEND))")
end

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
slice_zs = LinRange(first(zs), last(zs), NSLICES)
x⁰_samples = c .* frame_times

slice_screen(z) = ObserverScreen(txs, tys, z, x⁰_samples; c)

NSUB = let s = get(ENV, "EDM_NSUBSTEPS", "")
    isempty(s) ? recommended_n_substeps(slice_screen(first(slice_zs))) : parse(Int, s)
end
@info "radiation precompute" GPU_BACKEND NT NSLICES n_frames NSUB n_electrons = length(trajs)

# |E_far|² per (t, x, y), Float32 — the only thing the renderer needs.
function far_intensity(screen)
    fld = accumulate_field(
        trajs, screen, GPUKernelRK4(), gpu_backend;
        n_substeps = NSUB, mode = Val(:split), sync_per_electron = SYNC,
    )
    return Float32.(dropdims(sum(abs2, fld.E_far; dims = 2); dims = 2))
end

if BENCH
    # Mid slice = worst case (contains the electron disk). First call compiles
    # the kernel; the second is the honest per-slice cost.
    mid = slice_screen(slice_zs[NSLICES ÷ 2 + 1])
    t1 = @elapsed far_intensity(mid)
    t2 = @elapsed far_intensity(mid)
    total = t2 * NSLICES
    @info @sprintf(
        "bench: slice %.1fs (compile run %.1fs) → full cube ≈ %.1f min on this GPU",
        t2, t1, total / 60
    )
    exit(0)
end

rad = Array{Float32}(undef, n_frames, NSLICES, NT, NT)
# Orchestration-aware output: run_cell.sh sets EDM_OUTDIR (campaign dir) and
# EDM_RUN_TAG (uuid); a direct local run keeps the plain name in animation/.
OUTDIR = get(ENV, "EDM_OUTDIR", @__DIR__)
RUN_TAG = get(ENV, "EDM_RUN_TAG", "")
outfile = joinpath(OUTDIR, isempty(RUN_TAG) ? "radiation_cube.jls" : "radiation_cube_$(RUN_TAG).jls")
mkpath(OUTDIR)

t_total = @elapsed for (si, zsl) in enumerate(slice_zs)
    t_slice = @elapsed begin
        rad[:, si, :, :] .= far_intensity(slice_screen(zsl))
    end
    @info @sprintf("slice %3d/%d (X = %+6.1fλ)  %.1fs", si, NSLICES, zsl / λ, t_slice)
    if si % 16 == 0 && si < NSLICES
        serialize(outfile * ".partial", (; rad, slice_zs, txs, tys, frame_times, done = si))
    end
end

serialize(outfile, (; rad, slice_zs, txs, tys, frame_times))
rm(outfile * ".partial"; force = true)
@info @sprintf("radiation cube done in %.1f min → %s (%.2f GB)", t_total / 60, outfile, sizeof(rad) / 1e9)
