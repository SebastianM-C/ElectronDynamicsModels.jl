using ElectronDynamicsModels
using Test
using Aqua
using JET

@testset "ElectronDynamicsModels.jl" verbose = true begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ElectronDynamicsModels)
    end
    @testset "Code linting (JET.jl)" begin
        if VERSION < v"1.12"
            JET.test_package(ElectronDynamicsModels; target_modules = (ElectronDynamicsModels,))
        else
            @test_broken false  # JET causes CI termination on 1.12
        end
    end
    @testset "Classical Electron" begin
        include("classical_electron.jl")
    end
    @testset "Radiation Emission" begin
        include("thomson_scattering.jl")
    end
    @testset "Harmonic maps" begin
        include("harmonics.jl")
    end
    @testset "OAM analysis (ring_pixels + phase_winding_fit)" begin
        include("oam_analysis.jl")
    end
    @testset "GPU Interpolation" begin
        include("gpu_interp.jl")
    end
    @testset "Interpolant acceleration under saveat" begin
        include("interp_saveat_acceleration.jl")
    end
    @testset "GPU Radiation Accumulation" begin
        include("gpu_radiation.jl")
    end
    @testset "Lorenz Gauge" begin
        include("lorenz_gauge.jl")
    end
    @testset "LL regime reach (dynamics + kernel)" begin
        include("ll_regime.jl")
    end
    @testset "Radiation Reaction" begin
        include("radiation.jl")
    end
    @testset "GaussLaser" begin
        include("gauss_laser.jl")
    end
    @testset "LaguerreGauss" begin
        include("laguerre_gauss.jl")
    end
end
