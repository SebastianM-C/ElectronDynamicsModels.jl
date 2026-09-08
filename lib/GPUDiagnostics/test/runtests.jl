using GPUDiagnostics
using GPUDiagnostics: _fma_chain_reference, _fma_chain_kernel!, _PEAK_CHAINS
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
end
