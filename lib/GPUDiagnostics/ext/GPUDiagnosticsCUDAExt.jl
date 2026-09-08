module GPUDiagnosticsCUDAExt

# CUDA.jl implementations of the vendor-GPU API declared in src/device_api.jl. Loaded
# automatically when both GPUDiagnostics and CUDA are in the session. Telemetry
# (power/utilization/memory, GPM counters) goes through NVML; device props through CUDA attributes.

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

# ── Telemetry sources (src/sampler.jl hooks) ─────────────────────────────────────────────────
# The CUDA runtime is touched only HERE, in the parent, to map our ordinals to NVML uuids (stable
# under CUDA_VISIBLE_DEVICES). The sampler child loads CUDA.jl for its NVML bindings only (this
# extension then provides the `nvml` source kind there too), never creates a CUDA context, and is
# immune to this process's CUDA locks and Julia's GC/timer coupling.
function GD.gpu_sampler_sources(::CUDABackend, device_ids::AbstractVector{<:Integer}, counters::Symbol)
    cudevs = collect(CUDA.devices())
    specs = ["nvml:GPU-$(CUDA.uuid(cudevs[i]))=$i" for i in device_ids]
    return (specs = specs, packages = [Base.PkgId(CUDA)])
end

# NVML per-device source: power / utilization / memory via the plain NVML queries, plus — when
# the device supports GPU Performance Monitoring (Hopper and newer, incl. consumer Blackwell on
# recent drivers; no admin privileges) and counters are wanted — the GPM metrics. GPM is
# interval-based: a metric is the difference of two samples, so the source keeps two sample
# buffers and evaluates each tick against the previous tick's sample (rows cover exactly the
# preceding `dt`; the priming call takes a 100 ms interval).
const _GPM_METRICS = (   # column => (nvmlGpmMetricId_t, unit scale: percent → fraction; MiB/s as is)
    (:sm_util, 2, 0.01), (:sm_occupancy, 3, 0.01), (:fp64_util, 11, 0.01), (:dram_bw_util, 10, 0.01),
    (:fp32_util, 12, 0.01), (:fp16_util, 13, 0.01), (:tensor_util, 5, 0.01), (:int_util, 4, 0.01),
    (:pcie_tx_MiBps, 20, 1.0), (:pcie_rx_MiBps, 21, 1.0), (:nvlink_rx_MiBps, 60, 1.0), (:nvlink_tx_MiBps, 61, 1.0),
)
const _GPM_NAMES = Tuple(m[1] for m in _GPM_METRICS)

mutable struct NVMLSource <: GD.SamplerSource
    device::Int
    dev::NVML.Device
    gpm::Bool
    samples::Vector{NVML.nvmlGpmSample_t}   # two buffers, swapped every tick
    older::Int
    primed::Bool
    buf::Vector{UInt8}                      # nvmlGpmMetricsGet_t scratch (~19 kB, 477-entry metric array)
end

function _gpm_supported(dev::NVML.Device)
    try
        sup = Ref(NVML.nvmlGpmSupport_t(NVML.NVML_GPM_SUPPORT_VERSION, 0))
        NVML.nvmlGpmQueryDeviceSupport(dev, sup)
        return sup[].isSupportedDevice != 0
    catch err   # NVML_ERROR_NOT_SUPPORTED / FUNCTION_NOT_FOUND on old drivers ⇒ simply no GPM
        @debug "GPM support query failed" exception = err
        return false
    end
end

function GD._sampler_source(::Val{:nvml}, rest::AbstractString, counters::Symbol)
    m = match(r"^GPU-([0-9a-fA-F-]+)=(\d+)$", rest)
    m === nothing && throw(ArgumentError("nvml source spec must be GPU-<uuid>=<device>, got $rest"))
    dev = NVML.Device(Base.UUID(m.captures[1]))
    gpm = counters != :none && _gpm_supported(dev)
    samples = NVML.nvmlGpmSample_t[]
    if gpm
        for _ in 1:2
            s = Ref{NVML.nvmlGpmSample_t}()
            NVML.nvmlGpmSampleAlloc(s)
            push!(samples, s[])
        end
    end
    return NVMLSource(parse(Int, m.captures[2]), dev, gpm, samples, 1, false,
        gpm ? zeros(UInt8, sizeof(NVML.nvmlGpmMetricsGet_t)) : UInt8[])
end

function GD.sample!(s::NVMLSource)
    ur = NVML.utilization_rates(s.dev)
    base = (power_W = Float64(NVML.power_usage(s.dev)), compute_util = Float64(ur.compute),
        mem_util = Float64(ur.memory), vram_used_B = Float64(NVML.memory_info(s.dev).used))
    s.gpm || return base
    if !s.primed
        NVML.nvmlGpmSampleGet(s.dev, s.samples[s.older])
        s.primed = true
        sleep(0.1)
    end
    newer = 3 - s.older
    NVML.nvmlGpmSampleGet(s.dev, s.samples[newer])
    vals = _gpm_metrics!(s.buf, s.samples[s.older], s.samples[newer])
    s.older = newer
    return merge(base, NamedTuple{_GPM_NAMES}(Tuple(vals)))
end

function GD.close!(s::NVMLSource)
    for smp in s.samples
        try
            NVML.nvmlGpmSampleFree(smp)
        catch
        end
    end
    empty!(s.samples)
end

# nvmlGpmMetricsGet_t is a ~19 kB struct whose metric array the bindings expose as an opaque
# NTuple; drive it through a byte buffer and field offsets. Per-metric status lands in each
# entry's nvmlReturn (unsupported ⇒ nan); NVML_GPM_METRICS_GET_VERSION is literally 1.
const _GetT = NVML.nvmlGpmMetricsGet_t
const _MetricT = NVML.nvmlGpmMetric_t
function _gpm_metrics!(buf::Vector{UInt8}, older::NVML.nvmlGpmSample_t, newer::NVML.nvmlGpmSample_t)
    fill!(buf, 0)
    vals = fill(NaN, length(_GPM_METRICS))
    GC.@preserve buf begin
        p = pointer(buf)
        unsafe_store!(Ptr{Cuint}(p + fieldoffset(_GetT, 1)), Cuint(NVML.NVML_GPM_METRICS_GET_VERSION))
        unsafe_store!(Ptr{Cuint}(p + fieldoffset(_GetT, 2)), Cuint(length(_GPM_METRICS)))
        unsafe_store!(Ptr{NVML.nvmlGpmSample_t}(p + fieldoffset(_GetT, 3)), older)
        unsafe_store!(Ptr{NVML.nvmlGpmSample_t}(p + fieldoffset(_GetT, 4)), newer)
        mp(i) = Ptr{_MetricT}(p + fieldoffset(_GetT, 5) + (i - 1) * sizeof(_MetricT))
        for (i, m) in enumerate(_GPM_METRICS)
            unsafe_store!(mp(i).metricId, Cuint(m[2]))
        end
        try
            NVML.nvmlGpmMetricsGet(Ptr{_GetT}(p))
            for (i, m) in enumerate(_GPM_METRICS)
                unsafe_load(mp(i).nvmlReturn) == NVML.NVML_SUCCESS || continue
                v = unsafe_load(mp(i).value)
                isfinite(v) && (vals[i] = v * m[3])
            end
        catch err
            @debug "nvmlGpmMetricsGet failed" exception = err
        end
    end
    return vals
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


# ── Static instruction mix (src/instruction_mix.jl hook) ────────────────────────────────────
# CUDA.jl's `code_sass` disassembles the cubin from a CUPTI module-load callback, i.e. it
# LOADS the module on the current device — impossible for a cubin of another architecture.
# So the job is compiled here directly (`CUDACore.compile`: LLVM → PTX → the bundled ptxas →
# cubin, no device involved) and the bundled `nvdisasm` reads the cubin. The compiler config
# is the kernel's own (`always_inline` from the backend, `maxthreads` = the static workgroup
# size, the runtime's default `sm_NNa` for the current device) or, for `target = "sm_90"`,
# the same with `arch` overridden; ptxas's `.sectioninfo @"SHI_REGISTERS=N"` gives the
# register count to check against the runtime attribute.
const _GPUC = _CC.GPUCompiler

function GD._kernel_machine_code(backend::CUDABackend, ck::GD.CompiledKernel{<:CUDA.HostKernel}, target)
    k = ck.kernel
    TT = typeof(k).parameters[2]
    kw = ck.workgroup_size === nothing ? (;) : (; maxthreads = Int(ck.workgroup_size))
    arch_kw = target === nothing ? (;) : (; arch = String(target))
    config = _CC.compiler_config(CUDA.device(); kernel = true, always_inline = backend.always_inline, kw..., arch_kw...)
    job = _GPUC.CompilerJob(_GPUC.methodinstance(typeof(k.f), TT), config)
    compiled = _CC.compile(job)
    cubin = tempname(; cleanup = false) * ".cubin"
    write(cubin, compiled.image)
    text = try
        read(`$(_CC.CUDA_Compiler.nvdisasm()) --print-code $cubin`, String)
    finally
        rm(cubin; force = true)
    end
    isa = _CC.cpu_name(config.params.sm)
    m = match(r"SHI_REGISTERS=(\d+)", text)
    return (; text, vendor = :nvidia, isa, native = target === nothing,
        registers = m === nothing ? nothing : parse(Int, m[1]))
end

end
