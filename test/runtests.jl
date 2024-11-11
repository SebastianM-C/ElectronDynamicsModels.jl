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
    # Write your tests here.
end
