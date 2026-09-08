module GPUDiagnosticsCUDAExt

# CUDA.jl implementations of the vendor-GPU API declared in src/device_api.jl. Loaded
# automatically when both GPUDiagnostics and CUDA are in the session. Telemetry
# (power/utilization/memory) goes through NVML; device props through CUDA attributes.

using GPUDiagnostics
using CUDA
using CUDA: NVML

const GD = GPUDiagnostics

# NVML handle for the current CUDA device (NVML indexes by UUID, not the CUDA ordinal).
_nvml() = NVML.Device(CUDA.uuid(CUDA.device()))

GD.gpu_device_count(::CUDABackend) = length(CUDA.devices())
GD.gpu_device(::CUDABackend) = CUDA.deviceid(CUDA.device()) + 1          # 0-based CUDA → 1-based API
function GD.gpu_device!(::CUDABackend, i::Integer)
    prev = CUDA.deviceid(CUDA.device()) + 1
    CUDA.device!(i - 1)
    return prev
end
GD.gpu_name(::CUDABackend) = CUDA.name(CUDA.device())
GD.gpu_power(::CUDABackend) = NVML.power_usage(_nvml())                  # Watts (Float64)
GD.gpu_utilization(::CUDABackend) = NVML.utilization_rates(_nvml())     # (compute, memory) ∈ [0,1]
GD.gpu_memory_info(::CUDABackend) = NVML.memory_info(_nvml())           # (total, free, used) bytes

# Device-event timing on the task-local stream (the one KernelAbstractions launches on). CuEvent's
# default flags keep timing enabled; `elapsed` needs the stop event complete → synchronize it.
GD.gpu_event(::CUDABackend) = (e = CUDA.CuEvent(); CUDA.record(e, CUDA.stream()); e)
function GD.gpu_elapsed(start::CUDA.CuEvent, stop::CUDA.CuEvent)
    CUDA.synchronize(stop)
    return Float64(CUDA.elapsed(start, stop))   # seconds
end
GD.gpu_sm_count(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
GD.gpu_max_threads_per_sm(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

GD.gpu_arch(::CUDABackend) = (cc = CUDA.capability(CUDA.device()); "$(cc.major).$(cc.minor)")

# Telemetry child: the CUDA runtime is touched only HERE to map our ordinals to NVML uuids
# (stable under CUDA_VISIBLE_DEVICES); the spawned bin/gputrace_cuda.sh then runs one
# `nvidia-smi -lms` daemon per device from its own process, immune to this process's CUDA
# locks and Julia's GC/timer coupling.
function GD.gpu_telemetry_child_cmd(::CUDABackend, device_ids::AbstractVector{<:Integer},
        dt::Real, stopfile::AbstractString)
    script = joinpath(pkgdir(GD), "bin", "gputrace_cuda.sh")
    cudevs = collect(CUDA.devices())
    specs = ["GPU-$(CUDA.uuid(cudevs[i]))=$i" for i in device_ids]
    return `sh $script $(round(Int, 1000 * dt)) $(getpid()) $stopfile $specs`
end

# ── Compile-time resource report (src/resources.jl hooks) ───────────────────────────────────
# Inventory = the compiled-kernel cache of CUDA.jl's compiler (CUDACore from CUDA 6.3; the
# same names live in CUDA itself before the split). Attributes via the public `registers` /
# `memory` / `maxthreads` accessors (cuFuncGetAttribute underneath), occupancy via the driver's
# `cuOccupancyMaxActiveBlocksPerMultiprocessor`, capacities from the CURRENT device.
const _CC = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA

GD._compiled_kernels(::CUDABackend) = Base.@lock _CC.cufunction_lock begin
    [GD._compiled_kernel(k) for k in values(_CC._kernel_instances) if k isa CUDA.HostKernel]
end

function GD._kernel_attributes(::CUDABackend, k::CUDA.HostKernel)
    mem = CUDA.memory(k)   # (local, shared, constant) bytes; `local` is a keyword → positional
    return (;
        registers = Int(CUDA.registers(k)),
        local_mem_bytes = Int(mem[1]),
        shared_mem_bytes = Int(mem.shared),
        const_mem_bytes = Int(mem.constant),
        max_threads_per_block = Int(CUDA.maxthreads(k)),
    )
end

function GD._kernel_occupancy(::CUDABackend, k::CUDA.HostKernel, block_size::Int)
    dev = CUDA.device()
    return (;
        active_blocks_per_sm = Int(CUDA.active_blocks(k.fun, block_size)),
        warp_size = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)),
        max_threads_per_sm = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)),
        shared_mem_per_sm = Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)),
    )
end

# PTX / SASS binary versions the kernel was built for, plus the `ptxas --verbose` report of a
# regenerated compile. The driver attribute `local_mem_bytes` lumps the call-ABI stack frame
# (arguments/returns of device functions Julia did not inline, private arrays) together with
# true register spills; only ptxas separates them, and it also names every function it
# assembled out of line. CUDA.jl ships ptxas (CUDA_Compiler_jll) and compiles through it, so
# the same binary is run here on the module PTX generated with the kernel's own options
# (`always_inline` from the backend, `maxthreads` = the static KA workgroup size); the register
# count is checked against the runtime attribute so a mismatched regeneration is flagged, not
# trusted. Costs a few seconds of compiler time; nothing is launched.
function GD._kernel_isa_info(backend::CUDABackend, ck::GD.CompiledKernel{<:CUDA.HostKernel})
    k = ck.kernel
    info = Dict{String, Any}()
    try
        v = _CC.version(k)
        info["ptx_version"] = string(v.ptx)
        info["binary_version"] = string(v.binary)
    catch err
        @warn "kernel_resources: PTX/binary version query failed" exception = err
    end
    try
        merge!(info, _ptxas_report(backend, k, ck.workgroup_size))
        if haskey(info, "ptxas_registers")
            info["ptxas_matches_attributes"] = info["ptxas_registers"] == Int(CUDA.registers(k))
        end
    catch err
        @warn "kernel_resources: ptxas report unavailable — reporting driver attributes only" exception = err
    end
    return info
end

_ptxas_cmd() = isdefined(_CC, :CUDA_Compiler_jll) ? _CC.CUDA_Compiler_jll.ptxas() :
    isdefined(CUDA, :CUDA_Compiler_jll) ? CUDA.CUDA_Compiler_jll.ptxas() :
    error("CUDA_Compiler_jll (ptxas) not reachable from CUDA.jl")

function _ptxas_report(backend::CUDABackend, k::CUDA.HostKernel{F, TT}, workgroup_size) where {F, TT}
    io = IOBuffer()
    kw = workgroup_size === nothing ? (;) : (; maxthreads = Int(workgroup_size))
    CUDA.code_ptx(io, k.f, TT; kernel = true, raw = true, dump_module = true,
        always_inline = backend.always_inline, kw...)
    ptx = String(take!(io))
    m = match(r"\.target\s+(sm_\w+)", ptx)
    m === nothing && error("no .target line in the generated PTX")
    arch = m[1]
    ptxfile = tempname(; cleanup = false) * ".ptx"
    write(ptxfile, ptx)
    try
        cmd = `$(_ptxas_cmd()) --verbose --gpu-name $arch --output-file /dev/null $ptxfile`
        buf = IOBuffer()   # ptxas writes its report to stderr; capture both streams
        run(pipeline(ignorestatus(cmd); stdout = buf, stderr = buf))
        return GD._parse_ptxas_verbose(String(take!(buf)))
    finally
        rm(ptxfile; force = true)
    end
end

end
