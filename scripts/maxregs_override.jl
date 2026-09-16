# maxregs_override.jl — experiment wrapper (GPUDiagnostics issue #4): launch every
# KernelAbstractions CUDA kernel of this process with a per-thread register cap, without touching
# the production call sites. Included by inverse_thomson_scattering.jl ONLY when EDM_MAXREGS is set.
#
# KernelAbstractions' CUDABackend launch does not expose `maxregs`, so this redefines the launch
# method of CUDA.jl 6.3 (CUDACore/src/CUDAKernels.jl) verbatim plus `maxregs=MAXREGS` on the
# kernel_compile call — deliberate, scoped type piracy for an A/B. The cap is a compiler option,
# so the compiled kernel the resource report reads back ([gpu].kernel_registers /
# kernel_isa_spill_*) is the capped one; always_inline / maxthreads behave exactly as before.
# Tied to the pinned CUDA.jl: rewrite alongside any bump (the launch body moved in every major).
const MAXREGS = parse(Int, ENV["EDM_MAXREGS"])
MAXREGS > 0 || error("EDM_MAXREGS must be a positive register count, got $(ENV["EDM_MAXREGS"])")
const MAXREGS_LAST_KERNEL = Ref{Any}(nothing)   # the last compiled kernel, for tests / ad-hoc `CUDA.registers`
@info "maxregs override active: every KA CUDA kernel compiles with maxregs = $MAXREGS"

const CUDACore = CUDA.CUDACore
function (obj::KA.Kernel{CUDACore.CUDABackend})(args...; ndrange=nothing, workgroupsize=nothing)
    backend = KA.backend(obj)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    if KA.workgroupsize(obj) <: KA.StaticSize
        maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        maxthreads = nothing
    end
    call = CUDACore.kernel_call(obj.f, (ctx, args...))
    kernel = CUDACore.kernel_compile(call; always_inline=backend.always_inline, maxthreads, maxregs=MAXREGS)
    MAXREGS_LAST_KERNEL[] = kernel
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        config = CUDACore.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        if backend.prefer_blocks
            threads = min(prod(ndrange), config.threads)
            cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
            threads = cld(prod(ndrange), cu_blocks)
        else
            threads = config.threads
        end
        workgroupsize = CUDACore.CUDAKernels.threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
        call = CUDACore.rebind(call, ctx, 1)
    end
    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))
    blocks == 0 && return nothing
    CUDACore.kernel_launch(kernel, call; threads, blocks)
    return nothing
end
