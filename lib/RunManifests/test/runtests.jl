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

    # Schema version: the writer stamps the current version at top level, and it survives
    # the TOML round-trip as a top-level Int (not buried in a section).
    @test m["schema_version"] == MANIFEST_SCHEMA_VERSION
    @test manifest_schema_version(m) == MANIFEST_SCHEMA_VERSION
    @test check_schema_version(m) == MANIFEST_SCHEMA_VERSION
end

@testset "check_schema_version — policy" begin
    cur = MANIFEST_SCHEMA_VERSION
    # current ⇒ accepted, returns the version
    @test check_schema_version(Dict("schema_version" => cur)) == cur
    # missing ⇒ legacy v0, warns but proceeds
    @test (@test_logs (:warn,) check_schema_version(Dict{String, Any}())) == 0
    @test manifest_schema_version(Dict{String, Any}()) == 0
    # newer than we understand ⇒ hard error
    @test_throws ErrorException check_schema_version(Dict("schema_version" => cur + 1))
end

@testset "write_derived — single + multi-parent" begin
    dir = mktempdir()
    # single parent: depends_on is a 1-list, filename tagged by its id8 (backward-compatible)
    p1 = write_derived(dir; kind = "phase", label = "∠F", run_id = "aaaaaaaa-1111-2222",
        plot = "p.png", setup = Dict("harmonic" => 1))
    m1 = TOML.parsefile(p1)
    @test m1["derived"]["depends_on"] == ["aaaaaaaa-1111-2222"]
    @test occursin("aaaaaaaa", basename(p1)) && m1["schema_version"] == MANIFEST_SCHEMA_VERSION

    # multi-parent (a comparison): both ids in depends_on, filename tags both
    p2 = write_derived(dir; kind = "comparison", label = "cmp",
        run_id = ["aaaaaaaa-1111-2222", "bbbbbbbb-3333-4444"], plot = "c.png", setup = Dict("harmonic" => 2))
    m2 = TOML.parsefile(p2)
    @test m2["derived"]["depends_on"] == ["aaaaaaaa-1111-2222", "bbbbbbbb-3333-4444"]
    @test occursin("aaaaaaaa-bbbbbbbb", basename(p2))

    # no `[plot_params]` unless asked for (the common single-channel call)
    @test !haskey(m1, "plot_params")

    # plot_params → display-only [plot_params] section; round-trips, and (unlike setup)
    # does NOT influence the filename suffix.
    p3 = write_derived(dir; kind = "phaseE", label = "∠F E", run_id = "cccccccc-5555-6666",
        plot = "e.png", setup = Dict("harmonic" => 1),
        plot_params = Dict("ringtol" => 0.188, "radii" => [0.2, 0.4, 0.6]))
    m3 = TOML.parsefile(p3)
    @test m3["plot_params"]["ringtol"] == 0.188
    @test m3["plot_params"]["radii"] == [0.2, 0.4, 0.6]
    @test occursin("_1_", basename(p3))   # suffix from setup (harmonic=1) only, not plot_params
end
