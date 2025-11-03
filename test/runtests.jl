using ElectronDynamicsModels
using Test
using Aqua
using JET

@testset "ElectronDynamicsModels.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ElectronDynamicsModels)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(ElectronDynamicsModels; target_defined_modules = true)
    end
    @testset "Classical Electron" begin
        include("test_classical_electron.jl")
    end
    @testset "Radiation Reaction" begin
        include("test_radiation.jl")
    end
    @testset "LaguerreGauss" begin
        include("test_laguerre_gauss.jl")
    end
end
