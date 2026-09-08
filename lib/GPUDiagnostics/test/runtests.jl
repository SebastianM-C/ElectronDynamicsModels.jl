using GPUDiagnostics
using GPUDiagnostics: _fma_chain_reference, _fma_chain_kernel!, _PEAK_CHAINS
using GPUDiagnostics: _static_workgroup_size, _parse_amdgpu_kernel_info, _compiled_kernel
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
        @test_throws ErrorException gpu_telemetry_child_cmd(NoVendorBackend(), [1], 1.0, "stop")
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

    @testset "with_gpu_sampler: no child on the CPU backend, synthetic child parsed" begin
        r, telem = with_gpu_sampler(() -> 42, CPU(), 0.1)
        @test r == 42 && telem.ticks == 0 && isempty(telem.samples) && !telem.starved
        # A fake sampler child for the CPU backend: two devices, one row each per tick, plus a
        # torn row and an implausible row that the plausibility gate must drop.
        GPUDiagnostics.gpu_telemetry_child_cmd(::CPU, ids::AbstractVector{<:Integer}, dt::Real, stop::AbstractString) =
            `sh -c $("i=0; while [ ! -e '$stop' ] && [ \$i -lt 50 ]; do now=\$(date +%s.%N); " *
                     "printf '%s\\t1\\t100.0\\t0.99\\t0.50\\t1000\\n' \$now; " *
                     "printf '%s\\t2\\t150.0\\t1.00\\tnan\\t2000\\n' \$now; " *
                     "printf 'torn\\trow\\n'; printf '%s\\t1\\t100.0\\t7.0\\t0.5\\t1000\\n' \$now; " *
                     "i=\$((i+1)); sleep $dt; done")`
        trace = tempname() * ".tsv"
        r, telem = with_gpu_sampler(CPU(), 0.05; devices = 1:2, tracefile = trace) do
            sleep(0.5); :done
        end
        @test r == :done
        @test telem.ticks >= 3 && telem.trace == trace && isfile(trace)
        @test all(s -> s[2] in (1.0, 2.0) && 0 <= s[4] <= 1 && s[3] > 0, telem.samples)
        @test count(s -> s[2] == 2.0 && isnan(s[5]), telem.samples) == telem.ticks   # nan column kept
        @test !telem.starved
        rm(trace; force = true)
        Base.delete_method(only(methods(GPUDiagnostics.gpu_telemetry_child_cmd, (CPU, AbstractVector{<:Integer}, Real, AbstractString))))
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
        GPUDiagnostics._kernel_isa_info(::FakeGPU, k::FakeKernel) = Dict{String, Any}("vgpr_count" => 123)
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
