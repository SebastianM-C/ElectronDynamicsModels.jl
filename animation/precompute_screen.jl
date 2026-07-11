# Detector-screen time series for the animation: the exact LW far-field
# |E_far|² on ONE observer plane placed beyond the +X end of the volume box,
# sampled at the animation frame times — the animation renders it as a textured
# detector plane that lights up as the scattered radiation arrives.
# Cheap by construction: a single ObserverScreen slice (the volume cube is 160
# of these), so it runs on the local GPU in minutes.
#
# ENV knobs:
#   EDM_GPU_BACKEND   cuda (default, local) | rocm
#   EDM_SCREEN_XLAM   screen position along propagation, in λ (default 18)
#   EDM_SCREEN_NT     transverse pixels per side (default 256)
#   EDM_SCREEN_HALFW  transverse half-width in w₀ (default 3 — wider than the
#                     volume box: the emission keeps spreading past the box)
#   EDM_NSUBSTEPS     default 4 (validated); EDM_SYNC_PER_ELECTRON default false
#
# Output: animation/screen_timeseries.jls (EDM_OUTDIR/EDM_RUN_TAG aware):
#   (; scr, txs, tys, z_screen, frame_times), scr :: Float32 (n_frames, NT, NT)

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

XLAM = parse(Float64, get(ENV, "EDM_SCREEN_XLAM", "18"))
NT = parse(Int, get(ENV, "EDM_SCREEN_NT", "256"))
HALFW = parse(Float64, get(ENV, "EDM_SCREEN_HALFW", "3"))
NSUB = parse(Int, get(ENV, "EDM_NSUBSTEPS", "4"))
SYNC = parse(Bool, get(ENV, "EDM_SYNC_PER_ELECTRON", "false"))

z_screen = XLAM * λ
txs = LinRange(-HALFW * w₀, HALFW * w₀, NT)
tys = LinRange(-HALFW * w₀, HALFW * w₀, NT)
x⁰_samples = c .* frame_times

@info "screen precompute" GPU_BACKEND XLAM NT HALFW NSUB n_frames n_electrons = length(trajs)

screen = ObserverScreen(txs, tys, z_screen, x⁰_samples; c)
t_scr = @elapsed begin
    fld = accumulate_field(
        trajs, screen, GPUKernelRK4(), gpu_backend;
        n_substeps = NSUB, mode = Val(:split), sync_per_electron = SYNC,
    )
end
scr = Float32.(dropdims(sum(abs2, fld.E_far; dims = 2); dims = 2))

OUTDIR = get(ENV, "EDM_OUTDIR", @__DIR__)
RUN_TAG = get(ENV, "EDM_RUN_TAG", "")
outfile = joinpath(OUTDIR, isempty(RUN_TAG) ? "screen_timeseries.jls" : "screen_timeseries_$(RUN_TAG).jls")
mkpath(OUTDIR)
serialize(outfile, (; scr, txs, tys, z_screen, frame_times))
@info @sprintf("screen done in %.1f min → %s (%.0f MB)", t_scr / 60, outfile, sizeof(scr) / 1e6)
