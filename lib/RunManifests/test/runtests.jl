using Test
using RunManifests

@testset "expand_sweep — run matrix" begin
    # single axis: NX fixed, A0 swept over two values
    runs = expand_sweep(Dict("NX" => 64), Dict("A0" => [1.0e-3, 0.1]))
    @test length(runs) == 2
    @test all(r -> r isa Dict{String, String}, runs)
    @test all(r -> r["EDM_NX"] == "64", runs)               # base in every run
    @test Set(r["EDM_A0"] for r in runs) == Set(["0.001", "0.1"])

    # 2D cartesian product (no base)
    runs2 = expand_sweep(Dict(), Dict("A0" => [1, 2], "NX" => [64, 128]))
    @test length(runs2) == 4
    @test Set((r["EDM_A0"], r["EDM_NX"]) for r in runs2) ==
          Set([("1", "64"), ("1", "128"), ("2", "64"), ("2", "128")])

    # empty vary → exactly one run, == base alone
    @test expand_sweep(Dict("NX" => 64), Dict()) == [Dict("EDM_NX" => "64")]
end
