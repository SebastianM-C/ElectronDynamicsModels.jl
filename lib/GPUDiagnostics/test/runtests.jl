using GPUDiagnostics
using GPUDiagnostics: _fma_chain_reference, _fma_chain_kernel!, _PEAK_CHAINS
using GPUDiagnostics: _static_workgroup_size, _parse_amdgpu_kernel_info, _compiled_kernel, _parse_ptxas_verbose
using GPUDiagnostics: _classify, _parse_machine_code, _natural_loops, FP64_CLASSES, _ir_counts, IR_FP64_CLASSES
import KernelAbstractions as KA
using KernelAbstractions: CPU, Backend
using Test
using Aqua

# A KA backend with no vendor extension — exercises the "load CUDA.jl or AMDGPU.jl" fallbacks.
struct NoVendorBackend <: Backend end

# Stand-in for the LLVM.jl surface `GPUDiagnostics._ir_counts` uses (see the IR walker testset).
module FakeLLVM
    module API
        @enum Opcode LLVMFAdd LLVMFSub LLVMFMul LLVMFDiv LLVMFRem LLVMFNeg LLVMFCmp LLVMSIToFP LLVMUIToFP LLVMFPToSI LLVMFPToUI LLVMFPExt LLVMFPTrunc LLVMCall LLVMAdd LLVMSub LLVMMul LLVMUDiv LLVMSDiv LLVMURem LLVMSRem LLVMShl LLVMLShr LLVMAShr LLVMAnd LLVMOr LLVMXor LLVMICmp LLVMTrunc LLVMZExt LLVMSExt LLVMSelect LLVMLoad LLVMStore LLVMAtomicRMW LLVMAtomicCmpXchg LLVMBr LLVMSwitch LLVMRet LLVMUnreachable LLVMIndirectBr LLVMInvoke LLVMPHI LLVMGetElementPtr
        LLVMCanValueUseFastMathFlags(i) = Int(getfield(i, :op) in (LLVMFAdd, LLVMFSub, LLVMFMul, LLVMFDiv))
    end
    abstract type LLVMType end
    struct LLVMDouble <: LLVMType end
    struct LLVMFloat <: LLVMType end
    struct LLVMHalf <: LLVMType end
    struct LLVMBFloat <: LLVMType end
    struct IntType <: LLVMType end
    struct VectorType <: LLVMType; el::LLVMType; end
    Base.eltype(t::VectorType) = t.el
    struct Val; type::LLVMType; end
    struct Function; name::String; decl::Bool; blocks::Vector; end
    Function(name, decl) = Function(name, decl, [])
    struct Inst; op::API.Opcode; type::LLVMType; ops::Vector; callee::Union{Function, Nothing}; contract::Bool; end
    Inst(op, type, ops, callee = nothing; contract = false) = Inst(op, type, ops, callee, contract)
    fast_math(i::Inst) = (; contract = i.contract)
    struct Block; insts::Vector{Inst}; end
    struct Module; fns::Vector{Function}; end
    functions(m::Module) = m.fns
    isdeclaration(f::Function) = f.decl
    name(f::Function) = f.name
    blocks(f::Function) = f.blocks
    instructions(b::Block) = b.insts
    opcode(i::Inst) = i.op
    value_type(i::Inst) = i.type
    value_type(v::Val) = v.type
    operands(i::Inst) = i.ops
    arguments(i::Inst) = i.ops
    called_operand(i::Inst) = i.callee
end

@testset "GPUDiagnostics" begin
    @testset "Aqua" begin
        Aqua.test_all(GPUDiagnostics)
    end

    @testset "device API: CPU fallbacks + vendor-less errors" begin
        @test gpu_device_count(CPU()) == 1
        @test gpu_device(CPU()) == 1
        @test gpu_device!(CPU(), 1) == 1
        @test gpu_name(CPU()) == "CPU"
        @test gpu_arch(CPU()) == "cpu"
        for f in (gpu_device_count, gpu_device, gpu_name, gpu_power, gpu_utilization,
                gpu_memory_info, gpu_sm_count, gpu_max_threads_per_sm, gpu_arch)
            @test_throws ErrorException f(NoVendorBackend())
        end
        @test_throws ErrorException gpu_device!(NoVendorBackend(), 1)
        @test_throws ErrorException gpu_event(NoVendorBackend())
        @test_throws ErrorException thread_fill_occupancy(CPU(), 1024)   # no SM count on the host
    end

    @testset "device events + LaunchTimer (CPU backend = host clock)" begin
        e0 = gpu_event(CPU()); sleep(0.01); e1 = gpu_event(CPU())
        @test 0.005 < gpu_elapsed(e0, e1) < 5.0
        @test isempty(launch_times(LaunchTimer()))
        # `nothing` timer: every hook is a no-op
        @test launch_lane(nothing, CPU()) == 0
        @test launch_tick(nothing, CPU()) === nothing
        @test launch_tock!(nothing, 0, CPU(), nothing) === nothing
        # a loop instrumented the documented way, from two tasks on the same "device"
        t = LaunchTimer()
        @sync for _ in 1:2
            Threads.@spawn begin
                lane = launch_lane(t, CPU())
                for _ in 1:3
                    e = launch_tick(t, CPU())
                    sleep(0.002)
                    launch_tock!(t, lane, CPU(), e)
                end
            end
        end
        lt = launch_times(t)
        @test collect(keys(lt)) == [1]
        @test length(lt[1]) == 6 && all(>=(0.001), lt[1])
    end

    @testset "telemetry: sources, gpu_sample, child protocol, with_gpu_sampler, stats" begin
        # vendor-less / CPU: capability errors are clear, with_gpu_sampler degrades to unsampled
        @test_throws ErrorException gpu_sampler_sources(NoVendorBackend(), [1], :auto)
        @test_throws ErrorException gpu_sample(NoVendorBackend(), 1)
        @test_throws ErrorException gpu_sampler_sources(CPU(), [1], :auto)
        @test_throws ArgumentError with_gpu_sampler(() -> 1, CPU(), 0.1; counters = :bogus)
        @test_throws ArgumentError with_gpu_sampler(() -> 1, CPU(), 0.0)
        @test_throws ArgumentError gpu_sample(CPU(), 1; counters = :sometimes)
        r, telem = @test_logs (:warn, r"GPU telemetry unavailable") with_gpu_sampler(() -> 42, CPU(), 0.1)
        @test r == 42 && telem isa GPUTelemetry && telem.ticks == 0 && length(telem) == 0
        @test telem.columns == [:t_rel_s, :device] && telem.trace === nothing && !telem.starved && isnan(telem.first_sample_s)
        @test isempty(gpu_telemetry_stats(telem))
        @test_throws ArgumentError with_gpu_sampler(() -> throw(ArgumentError("boom")), NoVendorBackend(), 0.1)

        # sources: spec parsing, the built-in kinds, nan for what a device does not expose
        @test_throws ErrorException sampler_source("nosuch:1", :auto)
        @test_throws ArgumentError sampler_source("sysfs:1:only", :auto)
        s1 = sampler_source("synthetic:1", :auto)
        @test s1 isa SamplerSource && GPUDiagnostics.device_id(s1) == 1 && GPUDiagnostics.close!(s1) === nothing
        nt = GPUDiagnostics.sample!(s1)
        @test nt.power_W == 100 && nt.compute_util == 0.9 && nt.sm_occupancy == 0.3 && nt.fp64_util == 0.8
        @test !haskey(GPUDiagnostics.sample!(sampler_source("synthetic:2", :none)), :sm_util)
        @test isnan(GPUDiagnostics.sample!(sampler_source("synthetic:2", :auto)).mem_util)
        d = mktempdir()
        write(joinpath(d, "p"), "150000000\n"); write(joinpath(d, "b"), "75\n"); write(joinpath(d, "v"), "2048\n")
        sy = sampler_source("sysfs:3:$d/p:$d/b:-:$d/v", :auto)
        @test sy isa GPUDiagnostics.SysfsSource && GPUDiagnostics.device_id(sy) == 3
        nt = GPUDiagnostics.sample!(sy)
        @test nt.power_W == 150 && nt.compute_util == 0.75 && isnan(nt.mem_util) && nt.vram_used_B == 2048
        rm(joinpath(d, "b"))
        @test isnan(GPUDiagnostics.sample!(sy).compute_util) && GPUDiagnostics.sample!(sy).power_W == 150   # transient read failure → nan, row survives

        # gpu_sample in-process through the source cache: give the CPU backend synthetic devices
        GPUDiagnostics.gpu_sampler_sources(::CPU, ids::AbstractVector{<:Integer}, counters::Symbol) =
            (specs = ["synthetic:$i" for i in ids], packages = Base.PkgId[])
        @test gpu_sample(CPU(), 1).sm_util == 0.9 && gpu_sample(CPU()).power_W == 100
        @test !haskey(gpu_sample(CPU(), 2; counters = :none), :sm_util) && gpu_sample(CPU(), 2).power_W == 150

        # the child's argument protocol, command line and formatting
        o = GPUDiagnostics._parse_child_args(["--dt=0.25", "--ppid=12", "--stop=/tmp/x", "--counters=none", "synthetic:1", "synthetic:2"])
        @test o.dt == 0.25 && o.ppid == 12 && o.stopfile == "/tmp/x" && o.counters == :none && o.specs == ["synthetic:1", "synthetic:2"]
        @test GPUDiagnostics._parse_child_args(["--stop=/x"]).counters == :auto
        @test_throws ArgumentError GPUDiagnostics._parse_child_args(["--dt=1"])
        @test_throws ArgumentError GPUDiagnostics._parse_child_args(["--stop=/x", "--dt=0"])
        @test_throws ArgumentError GPUDiagnostics._parse_child_args(["--stop=/x", "--counters=foo"])
        cmd = GPUDiagnostics.telemetry_child_cmd(Base.PkgId[], 0.5, "/tmp/s", :auto, ["synthetic:1"])
        cs = string(cmd)
        @test occursin("--threads=1", cs) && occursin("telemetry_child_main", cs) && occursin("--counters=auto", cs) && occursin("--stop=/tmp/s", cs)
        @test occursin(string(Base.PkgId(GPUDiagnostics).uuid), cs)
        @test any(e -> startswith(e, "JULIA_LOAD_PATH="), cmd.env)
        @test GPUDiagnostics._fmt_value(NaN) == "nan" && GPUDiagnostics._fmt_value(2048.0) == "2048"
        @test GPUDiagnostics._fmt_value(0.93088) == "0.93088" && GPUDiagnostics._fmt_value(1.5e15) == "1.5e15"
        @test GPUDiagnostics._fmt_value(0.123456789) == "0.123457" && GPUDiagnostics._fmt_value(-0.5) == "-0.5"
        @test GPUDiagnostics._fixed(1788879414.9264, 3) == "1788879414.926" && GPUDiagnostics._fixed(2.9996, 3) == "3.000" && GPUDiagnostics._fixed(0.0, 2) == "0.00"
        @test GPUDiagnostics._parent_alive(getpid()) && GPUDiagnostics._parent_alive(0)
        Sys.islinux() && @test !GPUDiagnostics._parent_alive(2^22 - 1)
        @test !GPUDiagnostics._starved(6.4, 4.3, 5, 0.5)     # 4.3 s startup + full-rate ticks over a 6.4 s window
        @test GPUDiagnostics._starved(20.0, 4.0, 3, 0.5)     # ticks missing over the sampled part
        @test !GPUDiagnostics._starved(3.0, 0.5, 1, 0.5)     # too short a window to judge

        # THE REAL CHILD: a separate julia process sampling two synthetic devices
        trace = tempname() * ".tsv"
        t = @elapsed r, telem = with_gpu_sampler(CPU(), 0.1; devices = 1:2, tracefile = trace) do
            sleep(4); :done
        end
        @test r == :done && telem.trace == trace && isfile(trace) && telem.counters == :auto
        @test telem.columns == [:t_rel_s, :device, :power_W, :compute_util, :mem_util, :vram_used_B, :sm_util, :sm_occupancy, :fp64_util]
        @test telem.ticks >= 5 && length(telem) == 2 * telem.ticks && size(telem.samples) == (2 * telem.ticks, 9)
        @test 0 < telem.first_sample_s < 4 && !telem.starved && telem.window >= 4 && t < 12
        @test all(∈((1.0, 2.0)), telem[:device]) && all(∈((100.0, 150.0)), telem[:power_W])
        @test count(isnan, telem[:mem_util]) == telem.ticks && count(isnan, telem[:fp64_util]) == telem.ticks
        @test_throws KeyError telem[:nope]
        @test haskey(telem, :sm_util) && !haskey(telem, :nope) && keys(telem) == telem.columns
        @test occursin("rows", sprint(show, telem))
        @test startswith(readline(trace), "# epoch_s\tdevice\tpower_W\tcompute_util\tmem_util\tvram_used_B\tsm_util")
        @test length(split(readlines(trace)[2], '\t')) == 9
        st = gpu_telemetry_stats(telem)
        @test st["samples"] == telem.ticks && st["busy_samples"] == telem.ticks
        @test st["power_W_mean"] ≈ 125 && st["power_W_peak"] == 150 && st["power_W_busy_mean"] == 100
        @test st["compute_util_mean"] ≈ 0.5 && st["compute_util_peak"] == 0.9
        @test st["sm_occupancy_busy_mean"] ≈ 0.3 && st["sm_occupancy_mean"] ≈ 0.2 && st["sm_occupancy_peak"] == 0.3
        @test st["fp64_util_mean"] ≈ 0.8 && st["fp64_util_busy_mean"] ≈ 0.8 && st["mem_util_mean"] ≈ 0.5
        @test st["vram_used_B_peak"] == 2000 && st["vram_used_B_mean"] == 1500
        @test !haskey(st, "t_rel_s_mean") && !haskey(st, "device_mean")
        st2 = gpu_telemetry_stats(telem; busy_column = :sm_occupancy, busy_threshold = 0.05)
        @test st2["busy_samples"] == length(telem) && st2["power_W_busy_mean"] ≈ 125
        @test gpu_telemetry_stats(telem; busy_column = :absent)["busy_samples"] == 0
        rm(trace; force = true)
        # counters = :none ⇒ base columns only; no tracefile ⇒ temp trace removed
        r, telem = with_gpu_sampler(CPU(), 0.1; counters = :none) do
            sleep(3); 1
        end
        @test r == 1 && telem.columns == [:t_rel_s, :device, :power_W, :compute_util, :mem_util, :vram_used_B]
        @test telem.ticks >= 3 && telem.counters == :none && telem.trace === nothing

        # trace parser + plausibility gate on a handcrafted file
        f = tempname(); t0 = time() - 10
        open(f, "w") do io
            println(io, "$(t0 + 0.5)\t1\t1\t1\t1\t1")                        # before the header → ignored
            println(io, "# epoch_s\tdevice\tpower_W\tcompute_util\tmem_util\tvram_used_B\tsm_occupancy")
            println(io, "# a comment the child left")
            println(io, "$(t0 + 1)\t1\t100\t0.9\tnan\t1000\t0.3")             # good, nan kept
            println(io, "$(t0 + 1)\t2\t150\t0.1\t0.5\t2000\t0.1")             # good
            println(io, "torn\trow")                                           # torn
            println(io, "$(t0 + 2)\t1\t100\t7.0\t0.5\t1000\t0.3")             # util out of range
            println(io, "$(t0 + 2)\t1\t100\t0.9\t0.5\t3.0e20\t0.3")           # glued VRAM+epoch
            println(io, "$(t0 + 2)\t1\t9000\t0.9\t0.5\t1000\t0.3")            # 9 kW
            println(io, "$(t0 + 2)\t0\t100\t0.9\t0.5\t1000\t0.3")             # device 0
            println(io, "$(t0 - 100)\t1\t100\t0.9\t0.5\t1000\t0.3")           # outside the window
            println(io, "$(t0 + 3)\t1\t100\t0.9\t0.5\t1000\t1.5")             # occupancy > 1
            println(io, "$(t0 + 3)\t1\t100\t0.9\t0.5\t1000")                  # short row
            println(io, "$(t0 + 3)\t1\t100\t0.9\t0.5\t1000\t0.25")            # good
        end
        cols, rows = GPUDiagnostics._parse_trace(f, t0)
        @test cols == [:t_rel_s, :device, :power_W, :compute_util, :mem_util, :vram_used_B, :sm_occupancy]
        @test length(rows) == 3 && rows[1][1] ≈ 1 && rows[3][1] ≈ 3 && isnan(rows[1][5]) && rows[2][2] == 2
        @test GPUDiagnostics._parse_trace(tempname(), t0) == ([:t_rel_s, :device], Vector{Float64}[])
        rm(f)

        # a child whose sources cannot be opened exits without rows ⇒ warning, empty telemetry, no trace left
        Base.delete_method(only(methods(GPUDiagnostics.gpu_sampler_sources, (CPU, AbstractVector{<:Integer}, Symbol))))
        GPUDiagnostics.gpu_sampler_sources(::CPU, ids::AbstractVector{<:Integer}, counters::Symbol) =
            (specs = ["bogus:$i" for i in ids], packages = Base.PkgId[])
        trace2 = tempname() * ".tsv"
        r, telem = @test_logs (:warn, r"produced no samples") match_mode = :any with_gpu_sampler(CPU(), 0.1; tracefile = trace2) do
            sleep(2.5); 7
        end
        @test r == 7 && telem.ticks == 0 && !isfile(trace2) && !isfile(trace2 * ".stderr") && telem.window >= 2.5
        Base.delete_method(only(methods(GPUDiagnostics.gpu_sampler_sources, (CPU, AbstractVector{<:Integer}, Symbol))))
        @test_throws ErrorException gpu_sampler_sources(CPU(), [1], :auto)
        empty!(GPUDiagnostics._SOURCE_CACHE)
    end

    @testset "measured FP64 peak: FMA-chain probe (CPU backend) + host peakflops" begin
        n = 64
        out = zeros(n); seed = Float64.(0:(n - 1))
        _fma_chain_kernel!(CPU(), 16)(out, seed, Int32(1000); ndrange = n)
        @test all(i -> isapprox(out[i], _fma_chain_reference(seed[i], 1000); rtol = 1.0e-12), 1:n)
        @test _PEAK_CHAINS == 8
        p = measure_peak_fp64_flops(CPU(); n_threads = 4096, trials = 2, target_seconds = 0.02)
        @test isfinite(p) && p > 1.0e7
        @test_throws ArgumentError measure_peak_fp64_flops(CPU(); trials = 0)
        h = gpu_peak_fp64_flops(CPU())
        @test isfinite(h) && h > 1.0e8
    end

    @testset "compile-time resource report" begin
        # CPU backend compiles nothing; vendor-less backends error
        @test compiled_kernels(CPU()) == CompiledKernel[]
        @test isempty(compiled_kernels(CPU(); pattern = r"anything"))
        @test kernel_resources(CPU(), r"anything") == []
        @test_throws ErrorException kernel_resources(CPU(), CompiledKernel("k", "sig", 256, nothing))
        @test_throws ErrorException compiled_kernels(NoVendorBackend())

        # static workgroup size from a KA kernel signature (what mkcontext + StaticSize produce)
        CI1 = CartesianIndices{1, Tuple{Base.OneTo{Int}}}
        ctx(W) = KA.CompilerMetadata{KA.NDIteration.DynamicSize, KA.NDIteration.DynamicCheck, Nothing, CI1,
            KA.NDIteration.NDRange{1, KA.NDIteration.DynamicSize, W, CI1, Nothing}}
        @test _static_workgroup_size(Tuple{ctx(KA.NDIteration.StaticSize{(256,)}), Int}) == 256
        @test _static_workgroup_size(Tuple{ctx(KA.NDIteration.StaticSize{(16, 16)}), Int}) == 256
        @test _static_workgroup_size(Tuple{ctx(KA.NDIteration.DynamicSize), Int}) === nothing
        @test _static_workgroup_size(Tuple{Int, Float64}) === nothing
        @test _static_workgroup_size(Tuple{}) === nothing

        # CompiledKernel from a vendor-shaped kernel object K{F, TT}: name + signature + workgroup
        struct FakeKernel{F, TT}
            f::F
        end
        closure = let x = 1; i -> x + i; end
        fk = FakeKernel{typeof(closure), Tuple{ctx(KA.NDIteration.StaticSize{(128,)}), typeof(closure), Int}}(closure)
        ck = _compiled_kernel(fk)
        @test ck isa CompiledKernel && ck.workgroup_size == 128 && ck.kernel === fk
        @test ck.name == string(nameof(typeof(closure)))
        @test occursin("StaticSize{(128,)}", ck.signature)
        @test occursin("workgroup_size = 128", sprint(show, ck))

        # AMD ISA dump parser: the LLVM "; Kernel info:" comment block + the code-object metadata
        asm = """
        ; -- End function
        \t.amdhsa_next_free_vgpr 123
        ; Kernel info:
        ; codeLenInByte = 42124
        ; TotalNumSgprs: 107
        ; NumVgprs: 123
        ; ScratchSize: 584
        ; LDSByteSize: 65536 bytes/workgroup (compile time only)
        ; Occupancy: 10
        \t.amdgpu_metadata
        ---
        amdhsa.kernels:
          - .group_segment_fixed_size: 65536
            .kernarg_segment_size: 1016
            .max_flat_workgroup_size: 1024
            .private_segment_fixed_size: 584
            .sgpr_count:     107
            .sgpr_spill_count: 83
            .vgpr_count:     123
            .vgpr_spill_count: 0
            .wavefront_size: 32
        \t.end_amdgpu_metadata
        """
        info = _parse_amdgpu_kernel_info(asm)
        @test info["vgpr_count"] == 123 && info["sgpr_count"] == 107
        @test info["sgpr_spill_count"] == 83 && info["vgpr_spill_count"] == 0
        @test info["scratch_bytes"] == 584 && info["lds_bytes"] == 65536
        @test info["occupancy_waves_per_simd"] == 10 && info["code_bytes"] == 42124
        @test info["max_flat_workgroup_size"] == 1024 && info["wavefront_size"] == 32
        @test info["kernarg_bytes"] == 1016 && !haskey(info, "agpr_count")
        @test isempty(_parse_amdgpu_kernel_info("s_endpgm\n"))

        # ptxas --verbose parser: entry-function frame/spill split + out-of-line functions
        log = """
        ptxas info    : 256 bytes gmem
        ptxas info    : Compiling entry function '_Z23gpu__forindices_global_16CompilerMetadata' for 'sm_120a'
        ptxas info    : Function properties for _Z23gpu__forindices_global_16CompilerMetadata
            1712 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
        ptxas info    : Used 128 registers, used 0 barriers, 1712 bytes cumulative stack size
        ptxas info    : Compile time = 93.809 ms
        ptxas info    : Function properties for gpu_report_exception
            0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
        ptxas info    : Function properties for julia_GPUCubicSpline_15704
            0 bytes stack frame, 0 bytes spill stores, 8 bytes spill loads
        ptxas info    : Function properties for julia__140_15713
            0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
        """
        pi = _parse_ptxas_verbose(log)
        @test pi["stack_frame_bytes"] == 1712 && pi["spill_store_bytes"] == 0 && pi["spill_load_bytes"] == 0
        @test pi["ptxas_registers"] == 128 && pi["cumulative_stack_bytes"] == 1712
        @test pi["ptxas_functions"] == ["gpu_report_exception", "julia_GPUCubicSpline_15704", "julia__140_15713"]
        @test !haskey(pi, "ptxas_entry")
        @test isempty(_parse_ptxas_verbose("ptxas fatal   : Unresolved extern function\n"))
        # comment-only dumps (no metadata) still yield the figures; CDNA's AGPR line is picked up
        info2 = _parse_amdgpu_kernel_info("; NumSgprs: 40\n; NumVgprs: 64\n; NumAgprs: 8\n; TotalNumVgprs: 72\n; ScratchSize: 0\n; Occupancy: 8\n")
        @test info2["sgpr_count"] == 40 && info2["agpr_count"] == 8 && info2["total_vgpr_count"] == 72 && info2["scratch_bytes"] == 0

        # kernel_resources arithmetic on a fake GPU backend: occupancy = active warps / capacity
        struct FakeGPU <: Backend end
        GPUDiagnostics._compiled_kernels(::FakeGPU) = [ck]
        GPUDiagnostics._kernel_attributes(::FakeGPU, k::FakeKernel) =
            (; registers = 123, local_mem_bytes = 584, shared_mem_bytes = 65536, const_mem_bytes = -1, max_threads_per_block = 1024)
        GPUDiagnostics._kernel_occupancy(::FakeGPU, k::FakeKernel, block_size::Int) =
            (; active_blocks_per_sm = min(65536 ÷ 65536, 2048 ÷ block_size), warp_size = 32, max_threads_per_sm = 2048, shared_mem_per_sm = 65536)
        GPUDiagnostics._kernel_isa_info(::FakeGPU, c::CompiledKernel{<:FakeKernel}) = Dict{String, Any}("vgpr_count" => 123)
        @test length(compiled_kernels(FakeGPU())) == 1
        @test length(compiled_kernels(FakeGPU(); pattern = "StaticSize")) == 1
        @test isempty(compiled_kernels(FakeGPU(); pattern = r"no such kernel"))
        r = kernel_resources(FakeGPU(), ck)
        @test r.block_size == 128                      # defaults to the static workgroup size
        @test r.registers == 123 && r.local_mem_bytes == 584 && r.shared_mem_bytes == 65536
        @test r.active_blocks_per_sm == 1 && r.warp_size == 32 && r.max_warps_per_sm == 64
        @test r.active_warps_per_sm == 4 && r.occupancy ≈ 4 / 64
        @test r.isa["vgpr_count"] == 123 && r.name == ck.name && r.signature == ck.signature
        r2 = kernel_resources(FakeGPU(), ck; block_size = 1024)
        @test r2.active_warps_per_sm == 32 && r2.occupancy ≈ 0.5
        @test_throws ArgumentError kernel_resources(FakeGPU(), ck; block_size = 0)
        @test length(kernel_resources(FakeGPU(), "StaticSize")) == 1
    end

    @testset "static instruction mix: classifiers, parsers, loop nest, FP64 floor" begin
        # AMD mnemonics (encoding suffixes dropped; RDNA dual-issue by its first op)
        amd_expect = ("v_fma_f64" => :fp64_fma, "v_fmac_f64_e32" => :fp64_fma, "v_div_fmas_f64" => :fp64_fma,
            "v_add_f64" => :fp64_add, "v_sub_f64_e64" => :fp64_add, "v_mul_f64" => :fp64_mul,
            "v_rcp_f64_e32" => :fp64_trans, "v_rsq_f64_e32" => :fp64_trans, "v_sqrt_f64" => :fp64_trans,
            "v_div_scale_f64" => :fp64_other, "v_div_fixup_f64" => :fp64_other, "v_ldexp_f64" => :fp64_other,
            "v_cmp_lt_f64_e64" => :fp64_other, "v_cvt_f64_i32_e32" => :fp64_other, "v_max_f64" => :fp64_other,
            "v_pk_fma_f64" => :fp64_packed, "v_pk_mul_f64" => :fp64_packed, "v_pk_add_f64" => :fp64_packed,
            "v_mul_f32_e32" => :fp32, "v_rcp_iflag_f32_e32" => :fp32, "v_cvt_f32_u32_e32" => :fp32, "v_fma_f16" => :fp32,
            "v_dual_mov_b32" => :int, "v_cndmask_b32_e64" => :int, "v_accvgpr_read_b32" => :int, "v_mov_b64_e32" => :int,
            "v_writelane_b32" => :int, "v_add_co_u32" => :int, "v_mad_u64_u32" => :int,
            "s_mov_b32" => :salu, "s_cselect_b64" => :salu, "s_and_saveexec_b32" => :salu, "s_mul_hi_u32" => :salu,
            "s_load_b128" => :smem, "s_buffer_load_dword" => :smem,
            "s_waitcnt" => :wait, "s_waitcnt_vscnt" => :wait, "s_waitcnt_depctr" => :wait, "s_wait_loadcnt" => :wait,
            "s_cbranch_execz" => :control, "s_branch" => :control, "s_endpgm" => :control, "s_swappc_b64" => :control,
            "s_setpc_b64" => :control, "s_barrier" => :control,
            "s_delay_alu" => :wait, "s_nop" => :nop, "v_nop" => :nop, "s_clause" => :other, "s_set_inst_prefetch_distance" => :other,
            "global_load_b64" => :mem_load, "global_load_dwordx4" => :mem_load, "scratch_load_b64" => :mem_load,
            "buffer_load_dword" => :mem_load, "scratch_store_dwordx2" => :mem_store, "global_store_b64" => :mem_store,
            "flat_atomic_cmpswap_x2" => :mem_atomic, "global_atomic_add_f64" => :mem_atomic,
            "buffer_gl0_inv" => :other, "buffer_wbl2" => :other, "ds_load_b64" => :lds, "ds_store_b64" => :lds,
            "nonsense" => :other)
        for (op, cls) in amd_expect
            @test _classify(op, :amd) === (cls === :other && op == "nonsense" ? :unclassified : cls)
        end
        # SASS opcodes (base + modifiers; FP64 conversions/compares/MUFU seeds by their modifiers)
        sass_expect = ("DFMA" => :fp64_fma, "DADD" => :fp64_add, "DMUL" => :fp64_mul,
            "MUFU.RCP64H" => :fp64_trans, "MUFU.RSQ64H" => :fp64_trans, "MUFU.RCP" => :fp32, "MUFU.RSQ" => :fp32,
            "DSETP.GEU.AND" => :fp64_other, "DMNMX" => :fp64_other, "F2I.S64.F64.CEIL" => :fp64_other,
            "I2F.F64.S64" => :fp64_other, "FRND.F64.FLOOR" => :fp64_other, "F2F.F64.F32" => :fp64_other,
            "F2I.FTZ.U32.TRUNC.NTZ" => :fp32, "I2F.U64.RP" => :fp32, "FFMA" => :fp32, "FSEL" => :fp32, "HFMA2" => :fp32,
            "IMAD.WIDE.U32" => :int, "IADD3" => :int, "IADD.64" => :int, "LOP3.LUT" => :int, "SHF.R.U32.HI" => :int,
            "ISETP.NE.AND" => :int, "MOV" => :int, "SEL" => :int, "PRMT" => :int, "LEA.HI.X" => :int, "VIMNMX" => :int,
            "R2UR" => :salu, "S2UR" => :salu, "UIADD3" => :salu, "USHF.R.U32.HI" => :salu, "UMOV" => :salu, "UNEWTHING" => :salu,
            "LDG.E.64" => :mem_load, "LD.E.64" => :mem_load, "LDL.64" => :mem_load,
            "STG.E.64" => :mem_store, "ST.E.64" => :mem_store, "STL.64" => :mem_store, "ATOM.E.ADD.64" => :mem_atomic, "RED.E.ADD" => :mem_atomic,
            "LDC" => :smem, "LDCU.64" => :smem, "ULDC.64" => :smem, "LDS.64" => :lds, "STS" => :lds, "ATOMS.ADD" => :lds,
            "BRA" => :control, "BRA.U" => :control, "CALL.REL.NOINC" => :control, "RET.REL.NODEC" => :control, "EXIT" => :control,
            "BSSY.RECONVERGENT" => :control, "BSYNC" => :control, "WARPSYNC.ALL" => :control, "BREAK" => :control, "BAR.SYNC" => :control,
            "DEPBAR.LE" => :wait, "NOP" => :nop, "S2R" => :other, "CS2R" => :other, "MEMBAR.SC.GPU" => :other,
            "ERRBAR" => :other, "CCTL.IVALL" => :other, "LEPC" => :other, "SHFL.IDX" => :other, "XYZZY" => :other)
        for (op, cls) in sass_expect
            @test _classify(op, :nvidia) === (cls === :other && op == "XYZZY" ? :unclassified : cls)
        end
        @test length(MIX_CLASSES) == 18 && all(c in MIX_CLASSES for c in FP64_CLASSES)
        # the tables are ordered data: last rule is the catch-all, a pushfirst! override wins
        @test last(SASS_RULES)[2] === :unclassified && last(AMD_RULES)[2] === :unclassified
        @test all(r -> r isa Pair{Regex, Symbol}, SASS_RULES) && all(r -> r isa Pair{Regex, Symbol}, AMD_RULES)
        pushfirst!(SASS_RULES, r"^XYZZY$" => :int)
        pushfirst!(AMD_RULES, r"^v_mul_f64$" => :fp64_packed)
        try
            @test _classify("XYZZY", :nvidia) === :int
            @test _classify("v_mul_f64_e32", :amd) === :fp64_packed
        finally
            popfirst!(SASS_RULES); popfirst!(AMD_RULES)
        end
        @test _classify("XYZZY", :nvidia) === :unclassified && _classify("v_mul_f64_e32", :amd) === :fp64_mul

        # AMD ISA listing: two-level loop nest with the LLVM asm-printer annotations, a cold
        # exception tail, debug labels and the metadata YAML (whose "key:" lines are not labels)
        amd = """
        \t.text
        \t.globl\tk
        \t.p2align\t8
        \t.type\tk,@function
        k:                                      ; @k
        .Lfunc_begin0:
        ; %bb.0:
        \ts_load_b64 s[0:1], s[4:5], 0x0
        \ts_waitcnt lgkmcnt(0)
        \tv_cmp_gt_u32_e32 vcc_lo, 4, v0
        \ts_cbranch_vccz .LBB0_5
        .Ltmp0:
        .LBB0_1:                                ; %outer
                                                ; =>This Loop Header: Depth=1
        \t.loc\t1 10 0
        \tv_mul_f64 v[2:3], v[2:3], v[4:5]
        \tv_add_f64 v[2:3], v[2:3], 1.0
        \tglobal_load_b64 v[6:7], v[0:1], off
        \ts_waitcnt vmcnt(0)
        .LBB0_2:                                ; %inner
                                                ;   Parent Loop BB0_1 Depth=1
                                                ; =>This Inner Loop Header: Depth=2
        \tv_fma_f64 v[2:3], v[2:3], v[6:7], v[2:3]
        \tv_rcp_f64_e32 v[8:9], v[2:3]
        \ts_add_i32 s2, s2, 1
        \ts_cmp_lt_i32 s2, 8
        \ts_cbranch_scc1 .LBB0_2
        ; %bb.3:                                ;   in Loop: Header=BB0_1 Depth=1
        \ts_delay_alu instid0(VALU_DEP_1)
        \tv_cndmask_b32_e64 v1, 0, 1, vcc_lo
        \tglobal_store_b64 v[0:1], v[2:3], off
        \ts_cbranch_vccnz .LBB0_1
        ; %bb.4:
        \ts_endpgm
        .LBB0_5:
        \ts_mov_b32 s0, 0
        \tds_store_b64 v0, v[2:3]
        \tflat_atomic_cmpswap_b64 v[0:1], v[2:5], off
        \ts_endpgm
        .Lfunc_end0:
        \t.size\tk, .Lfunc_end0-k
        \t.amdgpu_metadata
        ---
        amdhsa.kernels:
          - .vgpr_count:     10
        amdhsa.version:
          - 1
        \t.end_amdgpu_metadata
        """
        blocks = _parse_machine_code(amd, :amd)
        @test [b.label for b in blocks] == ["k", "%bb.0", ".LBB0_1", ".LBB0_2", "%bb.3", "%bb.4", ".LBB0_5"]
        @test blocks[2].targets == [".LBB0_5"] && blocks[2].fallthrough
        @test blocks[4].targets == [".LBB0_2"] && blocks[5].targets == [".LBB0_1"]
        @test !blocks[6].fallthrough && !blocks[7].fallthrough
        @test blocks[3].loop_note == (".LBB0_1", 1) && blocks[4].loop_note == (".LBB0_2", 2) && blocks[5].loop_note == (".LBB0_1", 1)
        loops = _natural_loops(blocks)
        @test length(loops) == 2
        @test loops[1].header == 3 && loops[1].body == [3, 4, 5] && loops[1].depth == 1
        @test loops[2].header == 4 && loops[2].body == [4] && loops[2].depth == 2

        m = instruction_mix(amd, :amd)
        @test m.vendor === :amd && m.total == 22 && m.blocks == 7
        @test m.counts.fp64_mul == 1 && m.counts.fp64_add == 1 && m.counts.fp64_fma == 1 && m.counts.fp64_trans == 1
        @test m.counts.fp64_other == 0 && m.counts.fp64_packed == 0 && m.fp64 == 4
        @test m.counts.smem == 1 && m.counts.wait == 3 && m.counts.int == 2 && m.counts.control == 5 && m.counts.salu == 3
        @test m.counts.other == 0 && m.counts.nop == 0 && m.counts.mem_load == 1 && m.counts.mem_store == 1 && m.counts.lds == 1 && m.counts.mem_atomic == 1
        @test sum(m.counts) == m.total && m.opcodes["s_endpgm"] == 2 && m.opcodes["v_fma_f64"] == 1
        @test m.unclassified == 0 && isempty(m.unclassified_opcodes) && m.coverage == 1.0
        @test length(m.loops) == 2 && m.loops[1].header == ".LBB0_1" && m.loops[1].depth == 1 && m.loops[1].blocks == 3
        @test m.loops[1].total == 13 && m.loops[1].exclusive_total == 8 && m.loops[1].counts.fp64_fma == 1 && m.loops[1].exclusive_counts.fp64_fma == 0
        @test m.loops[2].header == ".LBB0_2" && m.loops[2].depth == 2 && m.loops[2].total == 5 && m.loops[2].exclusive_total == 5
        @test m.hot_loop.header == ".LBB0_1" && m.hot_loop_confidence === :high && m.llvm_loops_agree === true

        # an annotation that contradicts the CFG (block 3 claims a header that is not one) → :low
        amd_bad = replace(amd, "; %bb.3:                                ;   in Loop: Header=BB0_1 Depth=1" =>
            "; %bb.3:                                ;   in Loop: Header=BB0_9 Depth=1")
        mb = instruction_mix(amd_bad, :amd)
        @test mb.llvm_loops_agree === false && mb.hot_loop_confidence === :low && mb.total == 22
        # a wrong depth is caught too
        amd_bad2 = replace(amd, "=>This Inner Loop Header: Depth=2" => "=>This Inner Loop Header: Depth=1")
        @test instruction_mix(amd_bad2, :amd).llvm_loops_agree === false
        # no loops at all
        m0 = instruction_mix("k:\n\ts_load_b64 s[0:1], s[4:5], 0x0\n\ts_endpgm\n", :amd)
        @test m0.total == 2 && isempty(m0.loops) && m0.hot_loop === nothing && m0.hot_loop_confidence === :none && m0.llvm_loops_agree === nothing

        # SASS listing (nvdisasm --print-code): predicated/uniform branches, BSSY targets that
        # are not edges, an unconditional EXIT ending a block, the trap spin after it (dropped)
        sass = """
        \t.target\tsm_90
        \t.section\t.text.k,"ax",@progbits
                .type           k,@function
        k:
        .text.k:
                /*0000*/                   LDC R1, c[0x0][0x28] ;
                /*0010*/                   S2R R0, SR_TID.X ;
        \t//## File "./int.jl", line 520
                /*0020*/                   ISETP.GE.U32.AND P0, PT, R0, 0x4, PT ;
                /*0030*/               @P0 EXIT ;
                /*0040*/                   ULDC.64 UR4, c[0x0][0x210] ;
        .L_x_0:
                /*0050*/                   LDG.E.64 R2, [R4.64] ;
                /*0060*/                   DMUL R2, R2, R6 ;
                /*0070*/                   DADD R2, R2, 1 ;
                /*0080*/                   BSSY B0, `(.L_x_2) ;
        .L_x_1:
                /*0090*/                   DFMA R2, R2, R8, R2 ;
                /*00a0*/                   MUFU.RCP64H R9, R3 ;
                /*00b0*/                   DSETP.GT.AND P1, PT, R2, RZ, PT ;
                /*00c0*/                   IADD3 R10, R10, 0x1, RZ ;
                /*00d0*/                   ISETP.NE.AND P2, PT, R10, 0x8, PT ;
                /*00e0*/               @P2 BRA `(.L_x_1) ;
        .L_x_2:
                /*00f0*/                   BSYNC B0 ;
                /*0100*/                   STG.E.64 [R4.64], R2 ;
                /*0110*/                   F2I.S64.F64.FLOOR R11, R2 ;
                /*0120*/                   FFMA R12, R12, R13, R14 ;
                /*0130*/                   BRA.U !UP0, `(.L_x_0) ;
                /*0140*/                   EXIT ;
        .L_x_3:
                /*0150*/                   BRA `(.L_x_3);
                /*0160*/                   NOP;
                /*0170*/                   NOP;
        """
        sb = _parse_machine_code(sass, :nvidia)
        @test [b.label for b in sb] == ["k", ".text.k", ".L_x_0", ".L_x_1", ".L_x_2", ".L_x_3", "%bb.6"]
        @test sb[2].fallthrough && isempty(sb[2].targets)                 # @P0 EXIT does not end the block
        @test sb[3].targets == String[] && sb[3].fallthrough               # BSSY's operand is not a branch target
        @test sb[4].targets == [".L_x_1"] && sb[4].fallthrough
        @test sb[5].targets == [".L_x_0"] && !sb[5].fallthrough            # BRA.U !UP0 conditional, then EXIT
        @test sb[6].targets == [".L_x_3"] && !sb[6].fallthrough
        ms = instruction_mix(sass, :nvidia)
        @test ms.vendor === :nvidia && ms.total == 24 && ms.blocks == 7
        @test ms.counts.smem == 2 && ms.counts.other == 1 && ms.counts.nop == 2 && ms.counts.int == 3 && ms.counts.control == 7
        @test ms.counts.fp64_mul == 1 && ms.counts.fp64_add == 1 && ms.counts.fp64_fma == 1 && ms.counts.fp64_trans == 1
        @test ms.counts.fp64_other == 2 && ms.counts.fp32 == 1 && ms.counts.mem_load == 1 && ms.counts.mem_store == 1
        @test ms.fp64 == 6 && sum(ms.counts) == ms.total
        @test ms.unclassified == 0 && isempty(ms.unclassified_opcodes) && ms.coverage == 1.0
        # an unknown mnemonic is counted as `other` AND reported as unclassified; a deliberate
        # `other` (MEMBAR) is not
        mu = instruction_mix(replace(sass, "/*0160*/                   NOP;" => "/*0160*/                   XYZZY.FOO R1, R2;\n        /*0168*/                   MEMBAR.SC.GPU;"), :nvidia)
        @test mu.total == 25 && mu.counts.other == 3 && mu.counts.nop == 1
        @test mu.unclassified == 1 && mu.unclassified_opcodes == Dict("XYZZY.FOO" => 1) && mu.coverage ≈ 24 / 25
        mua = instruction_mix(replace(amd, "\ts_mov_b32 s0, 0" => "\tzzz_unknown_op v0\n\tbuffer_gl0_inv"), :amd)
        @test mua.unclassified == 1 && mua.unclassified_opcodes == Dict("zzz_unknown_op" => 1) && mua.counts.other == 2 && mua.coverage ≈ 22 / 23
        @test length(ms.loops) == 2                                        # the .L_x_3 trap spin is dropped
        @test ms.loops[1].header == ".L_x_0" && ms.loops[1].depth == 1 && ms.loops[1].blocks == 3 && ms.loops[1].total == 16 && ms.loops[1].exclusive_total == 10
        @test ms.loops[2].header == ".L_x_1" && ms.loops[2].depth == 2 && ms.loops[2].total == 6
        @test ms.hot_loop.header == ".L_x_0" && ms.hot_loop_confidence === :medium && ms.llvm_loops_agree === nothing
        @test_throws ArgumentError instruction_mix(sass, :intel)

        # FP64-issue floor: lane-instruction rate = peak / 2 (FMA chain), floor = count × slots / rate
        fl = fp64_issue_floor(ms; n_slots = 1.0e6, peak_fp64_flops = 2.0e12, kernel_time_s = 1.2e-5)
        @test fl.scope === :hot_loop && fl.fp64_per_slot == 6 && fl.fp64_lane_instructions == 6.0e6
        @test fl.floor_s ≈ 6.0e-6 && fl.fp64_issue_fraction ≈ 0.5 && fl.confidence === :medium
        fl2 = fp64_issue_floor(ms; n_slots = 10, peak_fp64_flops = 4.0)
        @test fl2.floor_s ≈ 30.0 && isnan(fl2.fp64_issue_fraction) && fl2.kernel_time_s === nothing
        flt = fp64_issue_floor(ms; n_slots = 1, peak_fp64_flops = 2.0, scope = :total)
        @test flt.fp64_per_slot == 6 && flt.confidence === :static_total
        @test_throws ArgumentError fp64_issue_floor(m0; n_slots = 1, peak_fp64_flops = 1.0)
        @test_throws ArgumentError fp64_issue_floor(ms; n_slots = 0, peak_fp64_flops = 1.0)
        @test_throws ArgumentError fp64_issue_floor(ms; n_slots = 1, peak_fp64_flops = 1.0, scope = :loop)

        # kernel_instruction_mix through the vendor hook (fake backend: AMD text natively, SASS for a target)
        ck = CompiledKernel("gpu_k", "Tuple{CompilerMetadata{…StaticSize{(256,)}…}}", 256, nothing)
        struct FakeMixGPU <: Backend end
        GPUDiagnostics._kernel_machine_code(::FakeMixGPU, c::CompiledKernel, target) = target === nothing ?
            (; text = amd, vendor = :amd, isa = "gfx1100", native = true, registers = 10) :
            (; text = sass, vendor = :nvidia, isa = String(target), native = false, registers = 124)
        GPUDiagnostics._kernel_ir_counts(::FakeMixGPU, c::CompiledKernel, target) =
            (; functions = Dict("k" => GPUDiagnostics.IRCounts(ntuple(i -> i, length(IR_CLASSES)))), isa = something(target, "gfx1100"))
        km = kernel_instruction_mix(FakeMixGPU(), ck)
        @test km.name == ck.name && km.signature == ck.signature && km.target == "gfx1100" && km.native && km.registers == 10
        @test km.coverage == 1.0 && km.ir.target == "gfx1100" && km.ir.counts.fp64_fma == 1 && km.ir.counts.fp64_add == 2
        @test km.ir.total == sum(1:length(IR_CLASSES)) && km.ir.fp64 == sum(1:length(IR_FP64_CLASSES)) && haskey(km.ir.functions, "k")
        @test kernel_instruction_mix(FakeMixGPU(), ck; ir = false).ir === nothing
        irm = kernel_ir_mix(FakeMixGPU(), ck; target = "gfx942")
        @test irm.target == "gfx942" && irm.counts == km.ir.counts
        @test_throws ErrorException kernel_ir_mix(CPU(), ck)
        @test_throws ErrorException kernel_ir_mix(NoVendorBackend(), ck)
        @test km.total == 22 && km.counts == m.counts && km.hot_loop.header == ".LBB0_1"
        buf = IOBuffer()
        km2 = kernel_instruction_mix(FakeMixGPU(), ck; target = "sm_90", dump = buf)
        @test km2.target == "sm_90" && !km2.native && km2.total == 24 && String(take!(buf)) == sass
        path = tempname()
        kernel_instruction_mix(FakeMixGPU(), ck; target = :sm_90, dump = path)
        @test read(path, String) == sass
        rm(path)
        @test_throws ErrorException kernel_instruction_mix(CPU(), ck)
        @test_throws ErrorException kernel_instruction_mix(NoVendorBackend(), ck)
    end

    @testset "typed IR walker on a stand-in LLVM.jl module" begin
        # _ir_counts takes the LLVM.jl module binding as an argument (AMDGPU.LLVM / CUDACore.LLVM in
        # the extensions); the same duck-typed surface is provided here by a tiny stand-in so the
        # walker's typing logic (double vs float by operand type, intrinsics by name) is CPU-tested.
        Fake = FakeLLVM
        dbl, flt, i64 = Fake.LLVMDouble(), Fake.LLVMFloat(), Fake.IntType()
        v2d = Fake.VectorType(dbl)
        fma64 = Fake.Function("llvm.fma.f64", true); sqrt64 = Fake.Function("llvm.sqrt.f64", true)
        fabs64 = Fake.Function("llvm.fabs.f64", true); fma32 = Fake.Function("llvm.fma.f32", true)
        lifetime = Fake.Function("llvm.lifetime.start.p0", true); helper = Fake.Function("julia_helper", true)
        A = Fake.API
        insts = [
            Fake.Inst(A.LLVMFMul, dbl, [Fake.Val(dbl)]), Fake.Inst(A.LLVMFAdd, dbl, [Fake.Val(dbl)]),
            Fake.Inst(A.LLVMFSub, dbl, [Fake.Val(dbl)]), Fake.Inst(A.LLVMFDiv, dbl, [Fake.Val(dbl)]),
            Fake.Inst(A.LLVMFNeg, dbl, [Fake.Val(dbl)]), Fake.Inst(A.LLVMFMul, flt, [Fake.Val(flt)]),
            Fake.Inst(A.LLVMFCmp, i64, [Fake.Val(dbl), Fake.Val(dbl)]), Fake.Inst(A.LLVMFCmp, i64, [Fake.Val(flt)]),
            Fake.Inst(A.LLVMSIToFP, dbl, [Fake.Val(i64)]), Fake.Inst(A.LLVMFPToSI, i64, [Fake.Val(dbl)]),
            Fake.Inst(A.LLVMFPTrunc, flt, [Fake.Val(dbl)]), Fake.Inst(A.LLVMSIToFP, flt, [Fake.Val(i64)]),
            Fake.Inst(A.LLVMCall, dbl, [], fma64), Fake.Inst(A.LLVMCall, dbl, [], sqrt64),
            Fake.Inst(A.LLVMCall, dbl, [], fabs64), Fake.Inst(A.LLVMCall, flt, [], fma32),
            Fake.Inst(A.LLVMCall, i64, [], lifetime), Fake.Inst(A.LLVMCall, i64, [], helper),
            Fake.Inst(A.LLVMAdd, i64, []), Fake.Inst(A.LLVMICmp, i64, []), Fake.Inst(A.LLVMSelect, dbl, []),
            Fake.Inst(A.LLVMLoad, dbl, []), Fake.Inst(A.LLVMStore, i64, []), Fake.Inst(A.LLVMAtomicRMW, i64, []),
            Fake.Inst(A.LLVMBr, i64, []), Fake.Inst(A.LLVMRet, i64, []), Fake.Inst(A.LLVMPHI, dbl, []),
            Fake.Inst(A.LLVMGetElementPtr, i64, []), Fake.Inst(A.LLVMFMul, v2d, [Fake.Val(v2d)]),
        ]
        mod = Fake.Module([Fake.Function("kernel", false, [Fake.Block(insts)]), Fake.Function("decl_only", true, [])])
        per = _ir_counts(Fake, mod)
        @test collect(keys(per)) == ["kernel"]
        c = per["kernel"]
        @test c.fp64_mul == 2 && c.fp64_add == 2 && c.fp64_div == 1 && c.fp64_neg == 1 && c.fp64_cmp == 1
        @test c.fp64_cvt == 3 && c.fp64_fma == 1 && c.fp64_sqrt == 1 && c.fp64_intrinsic == 1
        @test c.fp32 == 4 && c.int == 3 && c.call == 1 && c.mem_load == 1 && c.mem_store == 1 && c.mem_atomic == 1
        @test c.control == 2 && c.other == 3 && sum(c) == length(insts)
        # `contract`-flagged double fmul/fadd (what `muladd` lowers to) are counted separately and
        # not added to the fp64 total
        fc = FakeLLVM.Function("k", false, [FakeLLVM.Block([
            FakeLLVM.Inst(FakeLLVM.API.LLVMFMul, FakeLLVM.LLVMDouble(), []; contract = true),
            FakeLLVM.Inst(FakeLLVM.API.LLVMFAdd, FakeLLVM.LLVMDouble(), []; contract = true),
            FakeLLVM.Inst(FakeLLVM.API.LLVMFMul, FakeLLVM.LLVMDouble(), []),
            FakeLLVM.Inst(FakeLLVM.API.LLVMFMul, FakeLLVM.LLVMFloat(), []; contract = true)])])
        cc = _ir_counts(FakeLLVM, FakeLLVM.Module([fc]))["k"]
        @test cc.fp64_mul == 2 && cc.fp64_add == 1 && cc.fp64_contract == 2 && cc.fp32 == 1
        @test :fp64_contract ∉ IR_FP64_CLASSES
    end

    @testset "rocprofv3 counters: CSV parser, normalisation, command wrapper" begin
        fx = joinpath(@__DIR__, "fixtures", "rocprof")   # trimmed real MI300X collections (3 sets)
        S = 401 * 401 * 1666   # slots of one dispatch of the fixture cell (pixels × window samples)
        med = GPUDiagnostics._median

        # CSV: quoted kernel names with commas, bare numbers, RFC 4180 doubled quotes
        @test GPUDiagnostics._csv_fields("1,\"a, b\",\"x\"\"y\",,3.5") == ["1", "a, b", "x\"y", "", "3.5"]
        @test GPUDiagnostics._csv_fields("") == [""]
        @test med([3.0, 1.0, 2.0]) == 2 && med([4.0, 1.0, 2.0, 3.0]) == 2.5 && isnan(med(Float64[]))

        @test_throws ArgumentError rocprof_counters(fx)                      # three collections → name required
        @test_throws ArgumentError rocprof_counters(joinpath(fx, "nonexistent"))
        @test_throws ArgumentError rocprof_counters(fx; name = "nosuch")
        @test_throws ArgumentError rocprof_counters(fx; name = "sq2", kernel = "no_such_kernel")
        @test_throws ArgumentError rocprof_counters(fx; name = "sq2", kernel = r"gpu__")   # forindices + fma_chain
        @test_throws ArgumentError rocprof_counters(fx; name = "sq2", slots = 0)

        rc = rocprof_counters(fx; name = "sq2", slots = S)
        @test rc isa RocprofCounters && rc.name == "sq2" && rc.dir == fx
        @test startswith(rc.kernel, "gpu__forindices_global_(CompilerMetadata") && occursin(", ", rc.kernel)
        @test rc.dispatch_ids == [2, 3, 4] && length(rc) == 3
        @test Set(rc.counters) == Set(ROCPROF_COUNTER_SETS[:sq_issue]) && keys(rc) === rc.counters
        @test rc["GRBM_GUI_ACTIVE"] == [575399494.0, 570161410.0, 537860558.0]
        @test rc["SQ_INSTS_VMEM_RD"] == [365118475.0, 365365297.0, 364581971.0]
        @test haskey(rc, "SQ_INSTS_VALU") && !haskey(rc, "TCC_HIT_sum")
        @test_throws KeyError rc["TCC_HIT_sum"]
        @test rc.duration_s ≈ [0.040019311, 0.039977577, 0.040177312]
        @test rc.resources == (grid_size = 161024, workgroup_size = 256, vgpr_count = 64, agpr_count = 64,
            sgpr_count = 112, lds_bytes = 0, scratch_bytes = 92)
        @test rc.device == (name = "gfx942", product = "AMD Instinct MI300X VF", n_cu = 304, n_xcd = 8, n_se = 32,
            n_simd = 1216, wave_size = 64, max_waves_per_cu = 32)
        @test rc.slots == S
        @test [d.id for d in rc.dispatches] == [1, 2, 3, 4, 10] && occursin("initHeap", rc.dispatches[1].kernel)
        @test rc.dispatches[2].duration_s ≈ 0.040019311   # from the kernel trace
        @test rocprof_median(rc, "GRBM_GUI_ACTIVE") == 570161410.0
        @test occursin("3 dispatches of gpu__forindices_global_", sprint(show, rc))

        # kernel selection by substring: the FMA-chain peak probe of the same run
        fma = rocprof_counters(fx; name = "sq2", kernel = "fma_chain")
        @test fma.dispatch_ids == [10] && fma.resources.grid_size == 1048576 && fma.resources.vgpr_count == 24
        @test fma.slots === nothing && length(fma.dispatches) == 5

        # derived: per-slot instruction counts (× wave64 / slots), the die-corrected clock — each
        # per dispatch, reduced by the median
        d = rocprof_derived(rc)
        @test d["insts_per_slot_vmem_rd"] ≈ 365118475 * 64 / S
        @test d["insts_per_slot_valu"] ≈ med(rc["SQ_INSTS_VALU"]) * 64 / S
        @test 87 < d["insts_per_slot_vmem_rd"] < 88 && 1100 < d["insts_per_slot_valu"] < 1120
        @test d["clock_GHz"] ≈ 570161410 / 8 / 0.039977577 / 1e9 && 1.5 < d["clock_GHz"] < 2.2
        @test !haskey(d, "td_busy") && !haskey(d, "resident_waves") && !haskey(d, "fp64_flop_per_slot")
        d2 = rocprof_derived(rocprof_counters(fx; name = "sq2", slots = S ÷ 2))
        @test d2["insts_per_slot_vmem_rd"] ≈ 2 * d["insts_per_slot_vmem_rd"]
        @test !haskey(rocprof_derived(rocprof_counters(fx; name = "sq2")), "insts_per_slot_vmem_rd")   # no slots

        # unit-busy normalisation: X_BUSY_sum / (GRBM_GUI_ACTIVE / n_xcd × n_cu), per dispatch, median
        l1 = rocprof_counters(fx; name = "c2a")
        @test l1.dispatch_ids == [2, 3, 9] && l1.device.n_cu == 304 && l1.device.n_xcd == 8
        @test [d.id for d in l1.dispatches] == [2, 3, 9]   # no kernel trace → the counter file's dispatches
        e = rocprof_derived(l1)
        @test e["td_busy"] ≈ 18225880212 / (544437400 / 8 * 304)    # dispatch 2 is the median of the three
        @test e["ta_busy"] ≈ med(l1["TA_TA_BUSY_sum"] ./ (l1["GRBM_GUI_ACTIVE"] ./ 8 .* 304))
        @test e["tcp_pending_stall"] ≈ med(l1["TCP_PENDING_STALL_CYCLES_sum"] ./ (l1["GRBM_GUI_ACTIVE"] ./ 8 .* 304))
        @test e["l1_miss"] ≈ 1228667602 / 9642135941
        @test 0.85 < e["td_busy"] < 0.9 && 0.5 < e["ta_busy"] < 0.56 && 0.12 < e["l1_miss"] < 0.13
        # explicit device values override the agent info: die-summed cycles (n_xcd = 1) give 1/8 of
        # the fraction; 38 CUs per die with n_xcd = 1 is the same number as 304 CUs with n_xcd = 8
        e1 = rocprof_derived(rocprof_counters(fx; name = "c2a", n_xcd = 1))
        @test e1["td_busy"] ≈ e["td_busy"] / 8 && e1["clock_GHz"] ≈ 8 * e["clock_GHz"]
        @test rocprof_derived(rocprof_counters(fx; name = "c2a", n_cu = 38, n_xcd = 1))["td_busy"] ≈ e["td_busy"]

        # wave residency: quad-cycle SQ counters (×4); no agent info → device values from the caller
        w0 = rocprof_counters(fx; name = "sq1")
        @test w0.device.n_cu == 0 && w0.device.name == "" && w0.dispatch_ids == [2, 3]
        @test_throws ArgumentError rocprof_derived(w0)      # n_xcd unknown
        w = rocprof_counters(fx; name = "sq1", n_cu = 304, n_xcd = 8, wave_size = 64)
        f = rocprof_derived(w)
        @test f["resident_waves"] ≈ med(4 .* w["SQ_WAVE_CYCLES"] ./ (w["GRBM_GUI_ACTIVE"] ./ 8))
        @test 2190 < f["resident_waves"] < 2200 && f["waves_per_cu"] ≈ f["resident_waves"] / 304
        @test f["wave_wait_frac"] ≈ med(w["SQ_WAIT_ANY"] ./ w["SQ_WAVE_CYCLES"]) && 0.8 < f["wave_wait_frac"] < 0.81
        @test f["wave_active_inst_frac"] ≈ med(w["SQ_ACTIVE_INST_ANY"] ./ w["SQ_WAVE_CYCLES"])
        @test f["wave_active_valu_frac"] < f["wave_active_inst_frac"] < 0.2
        @test f["wave_wait_inst_frac"] ≈ med(w["SQ_WAIT_INST_ANY"] ./ w["SQ_WAVE_CYCLES"])
        @test f["wave_cycles"] ≈ (4 * 39699968966 / 2516 + 4 * 39290248983 / 2516) / 2
        @test !haskey(f, "occupancy") && !haskey(f, "sq_busy")   # need max_waves_per_cu / n_se (agent info)
        td = mktempdir()   # the same collection with an agent file: single collection → name discovered
        cp(joinpath(fx, "sq1_counter_collection.csv"), joinpath(td, "sq1_counter_collection.csv"))
        cp(joinpath(fx, "sq2_agent_info.csv"), joinpath(td, "sq1_agent_info.csv"))
        wa = rocprof_counters(td)
        @test wa.name == "sq1" && wa.device.n_cu == 304 && wa.device.max_waves_per_cu == 32 && wa.device.n_se == 32
        fa = rocprof_derived(wa)
        @test fa["occupancy"] ≈ fa["waves_per_cu"] / 32 && 0.2 < fa["occupancy"] < 0.25
        @test fa["sq_busy"] ≈ med(wa["SQ_BUSY_CYCLES"] ./ (wa["GRBM_GUI_ACTIVE"] ./ 8 .* 32)) && 0.9 < fa["sq_busy"] < 1
        @test fa["resident_waves"] ≈ f["resident_waves"]

        # summary + manifest section: flat, prefixed, TOML-serialisable
        s = rocprof_summary(rc)
        @test s["kernel"] == "gpu__forindices_global_" && s["dispatches"] == 3 && s["slots"] == S
        @test s["dispatch_median_s"] ≈ 0.040019311 && s["dispatch_min_s"] ≈ 0.039977577 && s["dispatch_max_s"] ≈ 0.040177312
        @test s["dispatch_total_s"] ≈ sum(rc.duration_s)
        @test s["grid_size"] == 161024 && s["vgpr_count"] == 64 && s["scratch_bytes"] == 92
        @test s["device"] == "AMD Instinct MI300X VF" && s["n_cu"] == 304 && s["n_xcd"] == 8 && s["wave_size"] == 64
        @test s["SQ_INSTS_SMEM_median"] == 110575.0 && s["SQ_INSTS_SMEM_rel_spread"] == 0
        @test s["GRBM_GUI_ACTIVE_median"] == 570161410.0 && s["GRBM_GUI_ACTIVE_rel_spread"] ≈ (575399494 - 537860558) / 570161410
        @test s["insts_per_slot_vmem_rd"] == d["insts_per_slot_vmem_rd"] && s["counters"] == rc.counters
        @test_throws ArgumentError rocprof_summary(w0)   # the summary carries the derived metrics
        @test !haskey(rocprof_summary(rocprof_counters(fx; name = "sq1", n_xcd = 8)), "n_cu")   # unknown device values are omitted
        ms = rocprof_manifest_section(fx; name = "sq2", slots = S)
        @test ms["rocprof_insts_per_slot_vmem_rd"] == d["insts_per_slot_vmem_rd"] && ms["rocprof_dispatches"] == 3
        @test all(startswith(k, "rocprof_") for k in keys(ms))
        @test all(v isa Union{Real, String, Vector{String}} for v in values(ms))
        @test rocprof_manifest_section(fx; name = "c2a", prefix = "hw_")["hw_td_busy"] ≈ e["td_busy"]

        # the wrapper: timeout -k, rocprofv3 flags, counters from a set or a list, env/dir preserved
        @test rocprof_available() == (Sys.which("rocprofv3") !== nothing)
        c = rocprof_command(`julia -e 1`; counters = :sq_waves, dir = "/tmp/out", name = "cell")
        @test c.exec[1:6] == ["timeout", "-k", "20", "600", "rocprofv3", "--kernel-trace"]
        @test c.exec[7] == "--pmc" && c.exec[8:15] == ROCPROF_COUNTER_SETS[:sq_waves]
        @test c.exec[16:end] == ["--output-format", "csv", "-d", "/tmp/out", "-o", "cell", "--", "julia", "-e", "1"]
        @test c.env === nothing && c.dir == ""
        c2 = rocprof_command(setenv(`env FOO=1 julia -e 1`, ["A=1"]; dir = "/tmp"); counters = ["TCC_HIT_sum", "TCC_MISS_sum"],
            dir = "o", name = "n", timeout_s = 120, kill_after_s = 5, kernel_trace = false, rocprofv3 = "/opt/rocm/bin/rocprofv3")
        @test c2.env == ["A=1"] && c2.dir == "/tmp" && c2.exec[1:5] == ["timeout", "-k", "5", "120", "/opt/rocm/bin/rocprofv3"]
        @test "--kernel-trace" ∉ c2.exec && c2.exec[6:8] == ["--pmc", "TCC_HIT_sum", "TCC_MISS_sum"]
        @test c2.exec[(end - 5):end] == ["--", "env", "FOO=1", "julia", "-e", "1"]
        @test_throws ArgumentError rocprof_command(`julia`; counters = :nosuch, dir = "o", name = "n")
        @test_throws ArgumentError rocprof_command(`julia`; counters = String[], dir = "o", name = "n")
        @test_throws ArgumentError rocprof_command(`julia`; counters = :fp64, dir = "o", name = "")
        @test_throws ArgumentError rocprof_command(`julia`; counters = :fp64, dir = "o", name = "n", timeout_s = 0)
        # every named set carries GRBM_GUI_ACTIVE (the cycle base of the derived metrics) and fits one pass
        @test all(("GRBM_GUI_ACTIVE" in v) && length(v) <= 8 for v in values(ROCPROF_COUNTER_SETS))
        @test Set(keys(ROCPROF_COUNTER_SETS)) ⊇ Set([:sq_issue, :sq_waves, :l1_pipe, :fp64, :l2])
        @test count(startswith("TCC_"), ROCPROF_COUNTER_SETS[:l2]) == 4   # the TCC block's per-pass capacity on gfx942
    end
end
