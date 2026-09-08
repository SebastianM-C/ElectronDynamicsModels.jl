# ── Compile-time resource report ────────────────────────────────────────────────────────────
#
# What the compiler gave a kernel — registers per thread, spill/stack memory, shared (LDS)
# memory — decides how many waves the hardware can keep resident, i.e. the theoretical
# occupancy, before a single launch runs. Neither KernelAbstractions nor the vendor packages
# surface this for a kernel that has already been launched, and the production kernels here
# are closures inside driver functions that never hand the kernel object out. Both CUDA.jl
# and AMDGPU.jl, however, keep every compiled kernel instance of the process in a cache
# (`_kernel_instances`, keyed by the Julia function + argument types), so the report is
# built from an INVENTORY of what actually went to the GPU: no kernel is modified, wrapped or
# recompiled, and the launch loop is untouched. The one vendor addition is the AMD ISA dump
# (`code_native`, ~0.5 s), which carries the SGPR/VGPR/spill/scratch counts the HIP function
# attributes do not expose.
#
# Identification: on Julia ≥ 1.12 a closure's type name embeds the enclosing function
# (`var"#outer##0#outer##1"`), so the signature string of a `foreachindex`/`@kernel` launch
# names the driver it came from; `compiled_kernels(backend; pattern = r"my_driver!")` picks it
# out. On Julia 1.10/1.11 closures are numbered (`var"#12#13"`) and the caller filters on the
# kernel `name` or argument types instead.

"""
    CompiledKernel

One entry of [`compiled_kernels`](@ref): a kernel this process compiled for the vendor
runtime. `name` is the Julia function the kernel was generated from (`gpu__forindices_global!`
for an AcceleratedKernels `foreachindex`, `gpu_<kernel>` for a KernelAbstractions `@kernel`),
`signature` the argument-type tuple as a string (closure types carry the enclosing function's
name on Julia ≥ 1.12), `workgroup_size` the static KernelAbstractions workgroup size read from
the signature (`nothing` when dynamic or not a KA kernel), and `kernel` the vendor kernel
object (CUDA.jl `HostKernel`, AMDGPU.jl `HIPKernel`).
"""
struct CompiledKernel{K}
    name::String
    signature::String
    workgroup_size::Union{Int, Nothing}
    kernel::K
end

Base.show(io::IO, ck::CompiledKernel) = print(io, "CompiledKernel(", ck.name, ", workgroup_size = ",
    ck.workgroup_size, ", ", length(ck.signature), "-char signature)")

"""    compiled_kernels(backend; pattern = nothing) -> Vector{CompiledKernel}

Inventory of the kernels THIS PROCESS has compiled for `backend`'s vendor runtime, from the
vendor package's kernel-instance cache (every `@cuda`/`@roc`/KernelAbstractions launch lands
there on its first call). `pattern` (a `Regex` or `AbstractString`) keeps only entries whose
`name` or `signature` matches — the way to pick a production kernel that is a closure inside
its driver: `compiled_kernels(backend; pattern = r"_my_driver!")`. The CPU backend compiles
nothing and returns an empty vector. Pass entries to [`kernel_resources`](@ref)."""
function compiled_kernels(backend::KA.Backend; pattern::Union{Regex, AbstractString, Nothing} = nothing)
    ks = _compiled_kernels(backend)
    pattern === nothing && return ks
    return filter(k -> occursin(pattern, k.name) || occursin(pattern, k.signature), ks)
end

# Vendor hook: the raw inventory (ext/ supplies the CUDA/AMDGPU methods).
_compiled_kernels(::KA.CPU) = CompiledKernel[]
_compiled_kernels(b::KA.Backend) = error(
    "compiled_kernels: no GPU vendor extension loaded for ", typeof(b), " — load CUDA.jl or AMDGPU.jl"
)

"""
    kernel_resources(backend, ck::CompiledKernel; block_size = something(ck.workgroup_size, 256))
    kernel_resources(backend, pattern; kwargs...) -> Vector

Compile-time resource report of a compiled kernel (see [`compiled_kernels`](@ref)), and the
theoretical occupancy the vendor runtime computes from it for a launch of `block_size`
threads per block (default: the kernel's static KernelAbstractions workgroup size, else 256).
Returns a NamedTuple:

- `name`, `signature`, `block_size` — identification and the block size the occupancy is for;
- `registers` — architectural registers per thread (NVIDIA registers; AMD VGPRs);
- `local_mem_bytes` — per-thread stack/spill memory (NVIDIA local memory; AMD scratch, the
  private segment). Non-zero means register spills or a stack frame: every access is a
  global-memory round trip;
- `shared_mem_bytes` — static shared memory (LDS) per block the kernel descriptor RESERVES,
  whether or not the source declares any (LLVM's AMDGPU backend promotes private arrays it
  cannot keep in registers to LDS, sized for the kernel's maximum block size);
- `const_mem_bytes`, `max_threads_per_block` — as reported by the runtime (`-1` = not
  reported);
- `active_blocks_per_sm`, `active_warps_per_sm`, `max_warps_per_sm`, `warp_size`,
  `occupancy` — the vendor occupancy calculator's resident blocks per SM (CU / WGP on AMD)
  at `block_size`, converted to warps (waves) and divided by the device's resident-warp
  capacity: `occupancy` ∈ (0, 1] is the theoretical occupancy the launch can reach. It
  accounts for registers, shared memory and block-size granularity together;
- `max_threads_per_sm`, `shared_mem_per_sm` — device capacities, for attributing the limiter
  (`shared_mem_per_sm ÷ shared_mem_bytes` blocks is the shared-memory bound);
- `isa` — a `Dict{String, Any}` of extra figures read from the compiled code where the
  vendor exposes them: on AMD, `sgpr_count`, `vgpr_count`, `agpr_count` (CDNA), the
  `sgpr_spill_count`/`vgpr_spill_count`, `scratch_bytes`, `lds_bytes`,
  `max_flat_workgroup_size`, `wavefront_size`, `code_bytes` and the compiler's own
  `occupancy_waves_per_simd` estimate (register budget only — compare with `occupancy` to see
  whether LDS or the block size is the binding constraint); on NVIDIA the `ptx_version` and
  `binary_version` the kernel was built for, plus the `ptxas --verbose` figures of a
  regenerated compile with the kernel's own options: `ptxas_registers`, `stack_frame_bytes`
  (the call-ABI / private-array frame), `spill_store_bytes`, `spill_load_bytes` (true
  register spills — the driver's local-memory figure lumps both together),
  `cumulative_stack_bytes`, and `ptxas_functions`, the device functions assembled
  out-of-line (a non-empty list beyond the runtime's exception helpers means Julia did not
  inline them; compare `CUDABackend(always_inline = true)`). `ptxas_matches_attributes`
  says whether that regenerated compile reproduced the runtime's register count.

The `pattern` form maps [`compiled_kernels`](@ref)`(backend; pattern)` through the report
(possibly empty). Nothing is launched; the AMD ISA dump and the NVIDIA PTX regeneration re-run
the compiler for a few seconds.
"""
function kernel_resources(backend::KA.Backend, ck::CompiledKernel;
        block_size::Integer = something(ck.workgroup_size, 256))
    block_size > 0 || throw(ArgumentError("kernel_resources: block_size must be > 0 (got $block_size)"))
    attrs = _kernel_attributes(backend, ck.kernel)          # registers, local/shared/const bytes, max threads
    occ = _kernel_occupancy(backend, ck.kernel, Int(block_size))   # active blocks/SM + device capacities
    isa = _kernel_isa_info(backend, ck)                     # vendor extras (may be empty)
    warps_per_block = cld(Int(block_size), occ.warp_size)
    max_warps = occ.max_threads_per_sm ÷ occ.warp_size
    active_warps = occ.active_blocks_per_sm * warps_per_block
    return (;
        name = ck.name, signature = ck.signature, block_size = Int(block_size),
        registers = attrs.registers, local_mem_bytes = attrs.local_mem_bytes,
        shared_mem_bytes = attrs.shared_mem_bytes, const_mem_bytes = attrs.const_mem_bytes,
        max_threads_per_block = attrs.max_threads_per_block,
        active_blocks_per_sm = occ.active_blocks_per_sm, active_warps_per_sm = active_warps,
        max_warps_per_sm = max_warps, warp_size = occ.warp_size,
        occupancy = max_warps > 0 ? active_warps / max_warps : NaN,
        max_threads_per_sm = occ.max_threads_per_sm, shared_mem_per_sm = occ.shared_mem_per_sm,
        isa,
    )
end
kernel_resources(backend::KA.Backend, pattern::Union{Regex, AbstractString}; kwargs...) =
    [kernel_resources(backend, ck; kwargs...) for ck in compiled_kernels(backend; pattern)]
kernel_resources(::KA.CPU, ck::CompiledKernel; kwargs...) = error(
    "kernel_resources: the CPU backend compiles no GPU kernels"
)

# Vendor hooks (ext/): attributes of a compiled kernel object, the runtime's occupancy
# calculator + the device capacities it is measured against, and the optional ISA figures.
# `::Any` kernel objects: the vendor types are only known inside the extensions.
function _kernel_attributes end
function _kernel_occupancy end
_kernel_isa_info(::KA.Backend, ck::CompiledKernel) = Dict{String, Any}()

# ── Pure helpers (vendor-neutral, tested on the CPU) ────────────────────────────────────────

# Build a CompiledKernel from a vendor kernel object of type `K{F, TT}` (both vendors
# parametrize their kernel struct by the Julia function type and the argument tuple type).
function _compiled_kernel(kernel)
    F, TT = typeof(kernel).parameters[1], typeof(kernel).parameters[2]
    return CompiledKernel(string(nameof(F)), string(TT), _static_workgroup_size(TT), kernel)
end

# Static KernelAbstractions workgroup size of a compiled kernel from its argument types: KA
# kernels take a `CompilerMetadata{…, Iterspace}` context first, whose `NDRange{N, Blocks,
# Workitems, …}` records a static workgroup size as `StaticSize{(256,)}`. `nothing` when the
# size is dynamic or the kernel is not a KA kernel.
function _static_workgroup_size(@nospecialize(TT::Type))
    TT <: Tuple && length(TT.parameters) >= 1 || return nothing
    ctx = TT.parameters[1]
    ctx isa DataType && ctx <: KA.CompilerMetadata && length(ctx.parameters) >= 5 || return nothing
    iterspace = ctx.parameters[5]
    iterspace isa DataType && iterspace <: KA.NDIteration.NDRange && length(iterspace.parameters) >= 3 || return nothing
    W = iterspace.parameters[3]
    W isa DataType && W <: KA.NDIteration.StaticSize || return nothing
    return prod(KA.NDIteration.get(W))
end

# Parse the resource figures the LLVM AMDGPU backend writes into a kernel's ISA dump: the
# `; Kernel info:` comment block (`; NumVgprs: 123`, `; ScratchSize: 584`, `; Occupancy: 10`,
# …) and the code-object metadata YAML (`.sgpr_spill_count: 83`, `.group_segment_fixed_size:
# 65536`, …). Returns the figures found (any subset; an empty Dict for text without them).
const _AMDGPU_INFO_KEYS = (
    # "; <Key>: <n>" comment lines                     → report key
    "NumSgprs" => "sgpr_count", "TotalNumSgprs" => "sgpr_count", "NumVgprs" => "vgpr_count",
    "NumAgprs" => "agpr_count", "TotalNumVgprs" => "total_vgpr_count",
    "ScratchSize" => "scratch_bytes", "Occupancy" => "occupancy_waves_per_simd",
    "codeLenInByte" => "code_bytes",
    # ".<key>: <n>" metadata lines
    ".sgpr_count" => "sgpr_count", ".vgpr_count" => "vgpr_count", ".agpr_count" => "agpr_count",
    ".sgpr_spill_count" => "sgpr_spill_count", ".vgpr_spill_count" => "vgpr_spill_count",
    ".group_segment_fixed_size" => "lds_bytes", ".private_segment_fixed_size" => "scratch_bytes",
    ".max_flat_workgroup_size" => "max_flat_workgroup_size", ".wavefront_size" => "wavefront_size",
    ".kernarg_segment_size" => "kernarg_bytes",
)
function _parse_amdgpu_kernel_info(asm::AbstractString)
    info = Dict{String, Any}()
    for line in eachline(IOBuffer(asm))
        m = match(r"^\s*;\s*(\w+)\s*[:=]\s*(\d+)", line)            # "; NumVgprs: 123", "; codeLenInByte = 42124"
        m === nothing && (m = match(r"^\s*(\.\w+):\s*(\d+)\s*$", line))   # ".vgpr_count:     123"
        m === nothing && continue
        key = m[1]
        for (src, dst) in _AMDGPU_INFO_KEYS
            src == key || continue
            # the metadata YAML is authoritative where both forms name the same figure
            (haskey(info, dst) && !startswith(key, '.')) || (info[dst] = parse(Int, m[2]))
            break
        end
        # `; LDSByteSize: 65536 bytes/workgroup (compile time only)`
        m2 = match(r"^\s*;\s*LDSByteSize:\s*(\d+)", line)
        m2 === nothing || haskey(info, "lds_bytes") || (info["lds_bytes"] = parse(Int, m2[1]))
    end
    return info
end

# Parse `ptxas --verbose` output for the entry function: its stack frame / spill bytes and
# register count, plus the names of the other functions ptxas assembled (device functions
# Julia left out of line — the runtime's exception helpers always appear). ptxas prints:
#   ptxas info    : Compiling entry function '<entry>' for 'sm_120a'
#   ptxas info    : Function properties for <entry>
#       1712 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#   ptxas info    : Used 128 registers, used 0 barriers, 1712 bytes cumulative stack size
#   ptxas info    : Function properties for julia_GPUCubicSpline_15704
#       0 bytes stack frame, 0 bytes spill stores, 8 bytes spill loads
# Returns the figures found (an empty Dict for text without them).
function _parse_ptxas_verbose(log::AbstractString)
    info = Dict{String, Any}()
    entry = nothing
    m = match(r"Compiling entry function '([^']+)'", log)
    m === nothing || (entry = m[1])
    current = nothing
    others = String[]
    for line in eachline(IOBuffer(log))
        m = match(r"Function properties for (\S+)", line)
        if m !== nothing
            current = m[1]
            current == entry || push!(others, current)
            continue
        end
        m = match(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m !== nothing && current == entry
            info["stack_frame_bytes"] = parse(Int, m[1])
            info["spill_store_bytes"] = parse(Int, m[2])
            info["spill_load_bytes"] = parse(Int, m[3])
            continue
        end
        m = match(r"Used (\d+) registers.*?(\d+) bytes cumulative stack size", line)
        if m !== nothing
            info["ptxas_registers"] = parse(Int, m[1])
            info["cumulative_stack_bytes"] = parse(Int, m[2])
        end
    end
    isempty(others) || (info["ptxas_functions"] = others)   # the entry name itself is mangled noise
    return info
end
