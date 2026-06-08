using Test
using RunManifests
using TOML

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

@testset "write_solver_manifest ↔ run_spec_from_manifest contract" begin
    dir = mktempdir()
    prov = run_provenance(; run_id = "rt", gpu_backend = "cuda", repo_dir = pkgdir(RunManifests))
    # [config] carrying EXACTLY the required replay keys (values arbitrary — run_spec stringifies).
    config = Dict{String, Any}(k => 1 for k in REQUIRED_CONFIG_KEYS)

    path = write_solver_manifest(
        dir; run_id = "rt", provenance = prov, config = config,
        laser = Dict("wavelength" => 1.0, "m" => -2), setup = Dict("Z" => 1.0),
        outputs = Dict("datafile" => "f.jls", "plots" => ["a.png"]),
        extra = Dict("model" => Dict("amplitude" => 2.5)),
    )
    m = TOML.parsefile(path)

    # THE contract: a manifest the writer produced must replay with no KeyError. Since `config`
    # here is exactly REQUIRED_CONFIG_KEYS, a successful run_spec also proves the required set
    # covers every [config] key the reader needs — the two lists can't silently drift apart.
    spec = run_spec_from_manifest(m)
    @test spec.commit == m["provenance"]["repo_commit"]
    @test haskey(spec.env, "EDM_A0") && haskey(spec.env, "EDM_SYNC_PER_ELECTRON")
    @test spec.env["EDM_GPU_BACKEND"] == "cuda"      # read from [provenance], not [config]
    @test m["model"]["amplitude"] == 2.5             # `extra` section written verbatim

    # Enforcement: dropping any single required key makes the writer refuse to write.
    for k in REQUIRED_CONFIG_KEYS
        bad = delete!(copy(config), k)
        @test_throws ErrorException write_solver_manifest(
            dir; run_id = "bad", provenance = prov, config = bad,
            laser = Dict(), setup = Dict(), outputs = Dict("plots" => String[]),
        )
    end
end
