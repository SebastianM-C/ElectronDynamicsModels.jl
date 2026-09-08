using GPUDiagnostics
using GPUDiagnostics: _fma_chain_reference, _fma_chain_kernel!, _PEAK_CHAINS
using GPUDiagnostics: _static_workgroup_size, _parse_amdgpu_kernel_info, _compiled_kernel, _parse_ptxas_verbose
import KernelAbstractions as KA
using KernelAbstractions: CPU, Backend
using Test
using Aqua

# A KA backend with no vendor extension — exercises the "load CUDA.jl or AMDGPU.jl" fallbacks.
struct NoVendorBackend <: Backend end

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
end
