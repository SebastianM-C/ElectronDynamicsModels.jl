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
