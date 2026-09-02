using Test
using RunManifests
using TOML
using Dates

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

    # Guarded inverse-Thomson knobs: round-trip when present, absent when absent (legacy manifests
    # replay with the script defaults — a legacy run must never grow new env out of thin air).
    cfg2 = merge(config, Dict{String, Any}(
        "gamma" => 50.0, "tspan_tau" => 1.6, "window_lead" => 0.15, "window_tail" => 0.15,
        "bunch_nb" => 398, "bunch_l" => -2))
    path2 = write_solver_manifest(
        dir; run_id = "rt2", provenance = prov, config = cfg2,
        laser = Dict(), setup = Dict(), outputs = Dict("plots" => String[]),
    )
    env2 = run_spec_from_manifest(TOML.parsefile(path2)).env
    @test env2["EDM_GAMMA"] == "50.0" && env2["EDM_TSPAN_TAU"] == "1.6"
    @test env2["EDM_WINDOW_LEAD"] == "0.15" && env2["EDM_WINDOW_TAIL"] == "0.15"
    @test env2["EDM_BUNCH_NB"] == "398" && env2["EDM_BUNCH_L"] == "-2"
    @test !haskey(spec.env, "EDM_TSPAN_TAU") && !haskey(spec.env, "EDM_WINDOW_LEAD") &&
          !haskey(spec.env, "EDM_BUNCH_NB")

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

@testset "write_comparison — declaration sidecar" begin
    dir = mktempdir()
    # the lpwa-vs-thomson case: two campaign dirs, each with its disambiguating script.
    path = write_comparison(
        dir; label = "LPWA vs numeric", differs = "method",
        sides = [
            (label = "analytical (LPWA)", dir = "lpwa_campaign_899970", script = "lpwa.jl"),
            (label = "numerical (Thomson)", dir = "field_campaign_898572", script = "thomson_scattering.jl"),
        ],
    )
    m = TOML.parsefile(path)
    @test m["schema_version"] == MANIFEST_SCHEMA_VERSION
    c = m["comparison"]
    @test c["label"] == "LPWA vs numeric" && c["differs"] == "method"
    @test !haskey(c, "along")                       # omitted ⇒ the dashboard infers the shared axis
    @test length(c["side"]) == 2
    @test c["side"][1]["dir"] == "lpwa_campaign_899970" && c["side"][1]["script"] == "lpwa.jl"
    @test c["side"][2]["label"] == "numerical (Thomson)"
    # deterministic filename on the side dirs ⇒ idempotent across an a0 sweep (one file, not N).
    @test basename(path) == "comparison_lpwa_campaign_899970__field_campaign_898572.toml"
    path2 = write_comparison(
        dir; label = "LPWA vs numeric", differs = "method",
        sides = [
            (label = "analytical (LPWA)", dir = "lpwa_campaign_899970", script = "lpwa.jl"),
            (label = "numerical (Thomson)", dir = "field_campaign_898572", script = "thomson_scattering.jl"),
        ],
    )
    @test path2 == path
    @test count(f -> startswith(f, "comparison_"), readdir(dir)) == 1

    # tuple sides, explicit `along`, a third side (A vs B vs C), and an optional/absent script.
    p3 = write_comparison(dir; label = "three-way", along = "a0",
        sides = [("a", "dirA"), ("b", "dirB", "x.jl"), ("c", "dirC")])
    m3 = TOML.parsefile(p3)
    @test m3["comparison"]["along"] == "a0"
    @test length(m3["comparison"]["side"]) == 3
    @test !haskey(m3["comparison"]["side"][1], "script")   # tuple without a script ⇒ key omitted
    @test m3["comparison"]["side"][2]["script"] == "x.jl"

    # fewer than two sides is a hard error (a comparison needs something to compare).
    @test_throws ErrorException write_comparison(dir; label = "x", sides = [("a", "dirA")])
end

@testset "units_section ↔ units_from_manifest" begin
    ω, λ, w₀ = 0.057, 2π * 137.03599908330932 / 0.057, 75 * 2π * 137.03599908330932 / 0.057

    # plain rest-frame run: ω₁-preferred, roundtrips through TOML
    u = units_section(ω, λ, w₀)
    io = IOBuffer(); TOML.print(io, Dict("units" => u))
    m = TOML.parse(String(take!(io)))
    r = units_from_manifest(m)
    @test r.n0 == 1
    @test r.preferred["frequency"] == "omega_laser"
    @test r.defs["lambda_laser"]["value"] ≈ λ
    @test r.system == "hartree_atomic"

    # backscatter run: ω_bs = n0·ω₁ preferred
    ub = units_section(ω, λ, w₀; n0 = 398)
    @test ub["preferred"]["frequency"] == "omega_bs"
    @test ub["defs"]["omega_bs"]["value"] ≈ 398ω
    @test units_from_manifest(Dict{String, Any}("units" => ub)).n0 == 398

    # near-rest backscatter: n0 is the EXACT fractional line, never rounded (the integer
    # round is a +2.1% axis/label offset at γ=1.5)
    n_th = 6.854101966249684
    uf = units_section(ω, λ, w₀; n0 = n_th)
    @test uf["defs"]["omega_bs"]["value"] ≈ n_th * ω
    @test units_from_manifest(Dict{String, Any}("units" => uf)).n0 ≈ n_th
    # pre-fix manifests stored omega_bs = round(n_th)·ω₁ in [units] while [config] always
    # carried the exact line — the config value must win when ω_bs is preferred
    ur = units_section(ω, λ, w₀; n0 = 7)
    mfix = Dict{String, Any}(
        "units" => ur, "config" => Dict{String, Any}("backscatter_n0" => n_th),
    )
    @test units_from_manifest(mfix).n0 ≈ n_th

    # Doppler-equivalent run: scaled carrier + unscaled lab reference
    s = 19.949874371066196
    us = units_section(s * ω, λ / s, w₀; omega_scale = s)
    @test us["defs"]["omega_lab"]["value"] ≈ ω
    @test us["defs"]["lambda_lab"]["value"] ≈ λ
    @test units_from_manifest(Dict{String, Any}("units" => us)).n0 == 1

    # extra named scales pass through
    ue = units_section(ω, λ, w₀; moat = (value = 22.4 * λ, tex = "r_F"))
    @test ue["defs"]["moat"]["tex"] == "r_F"

    # LEGACY manifests (no [units]): synthesized from laser/config — the backscatter_n0
    # and omega_scale contracts consumers already rely on.
    legacy = Dict{String, Any}(
        "laser" => Dict{String, Any}("wavelength" => λ, "w0" => w₀),
        "config" => Dict{String, Any}("backscatter_n0" => 398),
    )
    rl = units_from_manifest(legacy)
    @test rl.n0 == 398
    @test rl.defs["omega_laser"]["value"] ≈ ω
    @test rl.preferred["frequency"] == "omega_bs"
    legacy2 = Dict{String, Any}(
        "laser" => Dict{String, Any}("wavelength" => λ / s, "w0" => w₀),
        "config" => Dict{String, Any}("omega_scale" => s),
    )
    rl2 = units_from_manifest(legacy2)
    @test rl2.n0 == 1
    @test rl2.defs["omega_lab"]["value"] ≈ ω
end

@testset "ThomsonScatteringSpec — file ⇄ env ⇄ manifest" begin
    dir = mktempdir()

    # write_spec / load_spec roundtrip: typed coercion, extra passthrough, schema stamp.
    s0 = ThomsonScatteringSpec(; a0 = 2, gamma = 10, samples_per_period = 2048,
        accumulation_alg = "GPUKernelNewton", harmonics = [199, 299],
        extra = Dict{String, Any}("scattering" => "inverse"))
    @test s0.a0 === 2.0 && s0.harmonics == [199.0, 299.0]     # Int → Float64 coercion
    p = write_spec(joinpath(dir, "spec_cell.toml"), s0)
    m = TOML.parsefile(p)
    @test m["schema_version"] == MANIFEST_SCHEMA_VERSION
    @test m["spec"]["scattering"] == "inverse"                 # extra inlined
    s1 = load_spec(p; env = Dict{String, String}())
    @test s1.a0 == 2.0 && s1.gamma == 10.0 && s1.accumulation_alg == "GPUKernelNewton"
    @test s1.extra["scattering"] == "inverse"
    @test s1.N === nothing                                     # unset stays script-default

    # env overrides win over the file; special cases mirror the scripts.
    env = Dict("EDM_A0" => "0.5", "EDM_SYNC_PER_ELECTRON" => "true",
        "EDM_ABSTOL" => "", "EDM_INTERP_SAVEAT" => "16",
        "EDM_GPU_SOLVER" => "rk4",                             # legacy alias honored
        "EDM_HARMONICS" => "1,2.5", "EDM_OUTDIR" => "/ignored")
    s2 = load_spec(p; env)
    @test s2.a0 == 0.5 && s2.gamma == 10.0                     # override + carry-through
    @test s2.sync_per_electron === true && s2.abstol === nothing
    @test s2.interp_saveat == "16" && s2.harmonics == [1.0, 2.5]
    @test s2.accumulation_alg == "GPUKernelRK4"
    @test load_spec(nothing; env).a0 == 0.5                    # env-only legacy path

    # spec_env emission: dual alg knobs, integral harmonics without ".0", ε beats γ,
    # adaptive omitted, unset fields never emit.
    e = spec_env(ThomsonScatteringSpec(; a0 = 1.0, gamma = 1.001, gamma_eps = 0.001,
        accumulation_alg = "GPUKernelNewton", harmonics = [199, 1.0936],
        interp_saveat = "adaptive", sync_per_electron = false))
    @test e["EDM_GAMMA_EPS"] == "0.001" && !haskey(e, "EDM_GAMMA")
    @test e["EDM_ACCUM_ALG"] == "newton" && e["EDM_GPU_SOLVER"] == "newton"
    @test e["EDM_HARMONICS"] == "199,1.0936"
    @test e["EDM_SYNC_PER_ELECTRON"] == "false"
    @test !haskey(e, "EDM_INTERP_SAVEAT") && !haskey(e, "EDM_NX")

    # config_dict → manifest → spec_from_manifest roundtrip; unmapped keys ride extra.
    cfg = config_dict(s0)
    @test cfg["a0"] == 2.0 && cfg["scattering"] == "inverse"
    for k in REQUIRED_CONFIG_KEYS
        get!(cfg, k, 1)                                        # writer contract
    end
    prov = run_provenance(; run_id = "sp", gpu_backend = "cuda", repo_dir = pkgdir(RunManifests))
    path = write_solver_manifest(dir; run_id = "sp", provenance = prov, config = cfg,
        laser = Dict(), setup = Dict(), outputs = Dict("plots" => String[]))
    s3 = spec_from_manifest(TOML.parsefile(path))
    @test s3.a0 == 2.0 && s3.gamma == 10.0 && s3.harmonics == [199.0, 299.0]
    @test s3.extra["scattering"] == "inverse"

    # run_spec_from_manifest is table-driven now: previously-dropped knobs replay, and
    # ε-manifests emit EDM_GAMMA_EPS instead of a rounded EDM_GAMMA.
    cfg2 = Dict{String, Any}(k => 1 for k in REQUIRED_CONFIG_KEYS)
    cfg2["gamma_eps"] = 0.001
    cfg2["gamma"] = 1.001
    cfg2["system"] = "ll"
    cfg2["omega_scale"] = 19.9
    path2 = write_solver_manifest(dir; run_id = "sp2", provenance = prov, config = cfg2,
        laser = Dict(), setup = Dict(), outputs = Dict("plots" => String[]))
    env2 = run_spec_from_manifest(TOML.parsefile(path2)).env
    @test env2["EDM_GAMMA_EPS"] == "0.001" && !haskey(env2, "EDM_GAMMA")
    @test env2["EDM_SYSTEM"] == "ll" && env2["EDM_OMEGA_SCALE"] == "19.9"
    @test env2["EDM_FIELD_MODE"] == "split"                    # pre-mode default preserved
    # screen_zsign / apodization: [config] keys the inverse script writes — typed fields, so a
    # replay no longer silently reruns the script defaults for a −Z screen or a bare-FFT reduce.
    cfg3 = Dict{String, Any}(k => 1 for k in REQUIRED_CONFIG_KEYS)
    cfg3["screen_zsign"] = -1
    cfg3["apodization"] = "none"
    path3 = write_solver_manifest(dir; run_id = "sp3", provenance = prov, config = cfg3,
        laser = Dict(), setup = Dict(), outputs = Dict("plots" => String[]))
    s4 = spec_from_manifest(TOML.parsefile(path3))
    @test s4.screen_zsign === -1 && s4.apodization == "none" && isempty(s4.extra)
    env3 = run_spec_from_manifest(TOML.parsefile(path3)).env
    @test env3["EDM_SCREEN_ZSIGN"] == "-1" && env3["EDM_APODIZATION"] == "none"
    @test !haskey(env2, "EDM_SCREEN_ZSIGN") && !haskey(env2, "EDM_APODIZATION")   # absent stays absent
    s5 = load_spec(nothing; env = Dict("EDM_SCREEN_ZSIGN" => "-1", "EDM_APODIZATION" => "none"))
    @test s5.screen_zsign === -1 && s5.apodization == "none"
    # missing required keys fail loudly, not with a KeyError deep in emission
    @test_throws ErrorException run_spec_from_manifest(Dict{String, Any}(
        "schema_version" => 1, "config" => Dict{String, Any}("a0" => 1)))

    # spec-native sweep expansion: cartesian product, base carried, field vocabulary.
    cells = expand_sweep(ThomsonScatteringSpec(; a0 = 1.0),
        Dict("gamma" => [10, 100], "newton_iters" => [1, 2]))
    @test length(cells) == 4
    @test all(c -> c.a0 == 1.0, cells)
    @test Set((c.gamma, c.newton_iters) for c in cells) ==
          Set([(10.0, 1), (10.0, 2), (100.0, 1), (100.0, 2)])
    @test_throws ErrorException expand_sweep(ThomsonScatteringSpec(), Dict("nope" => [1]))
end

@testset "sweep declarations — write/read + membership stamp" begin
    dir = mktempdir()
    p = write_sweep_declaration(dir; name = "ll_gamma_ladder",
        script = "inverse_thomson_scattering.jl", axes = ["gamma"], design = "oat",
        label = "γ ladder")
    @test basename(p) == "sweep_ll_gamma_ladder.toml"
    m = TOML.parsefile(p)
    @test m["schema_version"] == MANIFEST_SCHEMA_VERSION
    @test m["sweep"]["axes"] == ["gamma"] && m["sweep"]["design"] == "oat"

    # several named declarations in one dir (the γ-ladder + mini-ladder case)
    write_sweep_declaration(dir; name = "n_iters",
        script = "inverse_thomson_scattering.jl", axes = ["newton_iters"])
    decls = read_sweep_declarations(dir)
    @test [d.name for d in decls] == ["ll_gamma_ladder", "n_iters"]
    @test decls[1].design == "oat" && decls[1].label == "γ ladder" && decls[1].hub === nothing
    @test decls[2].design == "grid" && decls[2].label === nothing

    # idempotent: rewriting a name replaces, never accumulates
    write_sweep_declaration(dir; name = "n_iters",
        script = "inverse_thomson_scattering.jl", axes = ["newton_iters", "gamma"])
    decls = read_sweep_declarations(dir)
    @test length(decls) == 2
    @test only(filter(d -> d.name == "n_iters", decls)).axes == ["newton_iters", "gamma"]

    # axes = [] declares a legitimate unstructured group (extras-only card)
    write_sweep_declaration(dir; name = "probe-extras", script = "x.jl", axes = String[])
    @test only(filter(d -> d.name == "probe-extras", read_sweep_declarations(dir))).axes ==
          String[]

    # validation: slug names, known designs, config-key (not env-spelled) axes
    @test_throws ErrorException write_sweep_declaration(dir; name = "Bad Name",
        script = "x.jl", axes = ["a0"])
    @test_throws ErrorException write_sweep_declaration(dir; name = "x",
        script = "x.jl", axes = ["a0"], design = "star")
    @test_throws ErrorException write_sweep_declaration(dir; name = "x",
        script = "x.jl", axes = ["EDM_GAMMA"])

    # duplicate names across files error loudly — declarations decide card structure
    dup = mktempdir()
    write_sweep_declaration(dup; name = "s", script = "a.jl", axes = ["a0"])
    mv(joinpath(dup, "sweep_s.toml"), joinpath(dup, "sweep_s2.toml"))
    write_sweep_declaration(dup; name = "s", script = "b.jl", axes = ["gamma"])
    @test_throws ErrorException read_sweep_declarations(dup)

    # membership stamp: EDM_SWEEP → provenance.sweep_id on solver AND analysis nodes;
    # omitted entirely when unset (legacy manifests stay byte-identical).
    withenv("EDM_SWEEP" => "ll_gamma_ladder") do
        prov = run_provenance(; run_id = "sw", gpu_backend = "cuda",
            repo_dir = pkgdir(RunManifests))
        @test prov["sweep_id"] == "ll_gamma_ladder"
        d2 = mktempdir()
        write_run_manifest(d2; run_id = "sw", script = "x.jl")
        @test TOML.parsefile(joinpath(d2, "run_sw.toml"))["provenance"]["sweep_id"] ==
              "ll_gamma_ladder"
    end
    withenv("EDM_SWEEP" => nothing) do
        prov = run_provenance(; run_id = "sw2", gpu_backend = "cuda",
            repo_dir = pkgdir(RunManifests))
        @test !haskey(prov, "sweep_id")
        @test run_provenance(; run_id = "sw3", gpu_backend = "cuda",
            repo_dir = pkgdir(RunManifests), sweep_id = "explicit")["sweep_id"] == "explicit"
        d3 = mktempdir()
        write_run_manifest(d3; run_id = "sw4", script = "x.jl")
        @test !haskey(TOML.parsefile(joinpath(d3, "run_sw4.toml"))["provenance"], "sweep_id")
    end

    # a nonexistent dir reads as no declarations (callers probe dirs freely)
    @test isempty(read_sweep_declarations(joinpath(dir, "nope")))
end

@testset "dashboard client — browse + lazy load" begin
    # A file:// dashboard fixture: index.json + data/<uuid>/<file>, the same layout the
    # live hosts serve — libcurl's file protocol stands in for Caddy.
    function fileurl(dir)
        p = replace(abspath(dir), '\\' => '/')
        return "file://" * (startswith(p, "/") ? "" : "/") * p
    end
    J = RunManifests.JSON
    S = RunManifests.Serialization

    @testset "fat index: campaigns, selection, caches, integrity" begin
        mktempdir() do dir
            u1 = "aaaa0000-0000-4000-8000-000000000001"
            u2 = "bbbb0000-0000-4000-8000-000000000002"
            payload = (; freqs = [1.0, 2.0, 3.0], ps = [0.1, 0.2, 0.3], n0 = 398)
            mkpath(joinpath(dir, "data", u1))
            S.serialize(joinpath(dir, "data", u1, "powspec_$(u1).jls"), payload)
            nb = filesize(joinpath(dir, "data", u1, "powspec_$(u1).jls"))
            S.serialize(joinpath(dir, "data", u1, "conv_$(u1).jls"), (; slope = 2.0))
            index = Dict(
                "generated" => "2026-08-01T00:00:00",
                "runs" => [
                    Dict("id" => u1, "label" => "#001", "script" => "thomson_scattering.jl",
                        "commit" => "abc123", "host" => "", "backend" => "rocm",
                        "provider" => "hotaisle", "timestamp" => "2026-07-18T10:00:00",
                        "params" => Dict("a0" => 0.1, "N_electrons" => 2000),
                        "plots" => Dict(), "data" => "/data/$(u1)/field_401_$(u1).jls",
                        "dir" => "camp_a", "repo_dirty" => false, "julia_version" => "1.12.6",
                        "config" => Dict("a0" => 0.1, "gamma" => 10.0),
                        "laser" => Dict("wavelength" => 110.21),
                        "setup" => Dict("Z" => 2.0e5),
                        "caches" => [
                            Dict("file" => "powspec_$(u1).jls", "bytes" => nb,
                                "url" => "/data/$(u1)/powspec_$(u1).jls"),
                            Dict("file" => "hmaps_$(u1).jls", "bytes" => 999,
                                "url" => nothing),      # recorded, not yet published
                        ]),
                    Dict("id" => u2, "label" => "#002", "params" => Dict("a0" => 0.5),
                        "plots" => Dict(), "data" => nothing, "dir" => "camp_a",
                        "caches" => Any[]),
                ],
                "sweeps" => [Dict("dir" => "camp_a", "id" => "camp_a:camp_a",
                    "declared" => true,
                    "axes" => [Dict("param" => "a0", "values" => [0.1])],
                    "cells" => [Dict("coord" => [1], "run" => u1)],
                    "extras" => [Dict("run" => u2, "label" => "ctl",
                        "differs" => Dict("a0" => 0.5))],
                    "summaries" => Dict("conv" => Dict("label" => "convergence",
                        "axis" => "a0", "url" => "/plots/$(u1)/conv.png",
                        "data" => "/data/$(u1)/conv_$(u1).jls")))],
                "standalone" => Any[],
                "comparisons" => [Dict("id" => "abc123", "label" => "A vs B",
                    "along" => "a0",
                    "sides" => [Dict("dir" => "camp_a", "sweep" => "camp_a:camp_a"),
                                Dict("dir" => "camp_b", "sweep" => nothing)])])
            write(joinpath(dir, "index.json"), J.json(index))

            dash = dashboard(url = fileurl(dir))
            @test sweeps(dash) == ["camp_a"]
            s = dash["camp_a"]
            @test length(runs(s)) == 2 && length(sweeps(s)) == 1
            n = dash["camp_a:camp_a"]                   # declared stable id → member view
            @test Set(x.id for x in runs(n)) == Set([u1, u2])   # cells + extras
            @test length(n.entries) == 1 && n.dir == "camp_a"
            r = only(runs(s; a0 = 0.1))                 # sweep-coordinate selection
            @test r.id == u1
            @test s["#001"].id == u1                    # gallery label …
            @test s["aaaa"].id == u1                    # … and uuid prefix
            @test_throws ErrorException s["zzzz"]       # no match
            @test isempty(runs(s; a0 = 7.7))
            @test length(runs(dash)) == 2               # whole-index run listing

            sm = only(summaries(s))                     # campaign-level plot rows
            @test sm.kind == "conv" && sm.axis == "a0" && sm.sweep == "camp_a:camp_a"
            @test loaddata(dash, sm.data).slope == 2.0  # lazy summary-datafile fetch
            @test length(comparisons(dash)) == 1
            @test length(comparisons(dash; involving = "camp_a")) == 1
            @test only(comparisons(dash; involving = "camp_a:camp_a"))["label"] == "A vs B"
            @test isempty(comparisons(dash; involving = "nope"))

            @test Set(x.key for x in caches(r)) == Set([:powspec, :hmaps, :field])

            p = cachepath(r, :powspec)                  # lazy fetch → scratch store
            @test isfile(p) && startswith(p, data_store_dir())
            got = loadcache(r, :powspec)
            @test got.freqs == payload.freqs && got.n0 == 398
            write(p, "junk")                            # corrupt copy → size-check refetch
            @test loadcache(r, :powspec).ps == payload.ps
            @test_throws ErrorException cachepath(r, :hmaps)   # no URL yet → clear error
            @test_throws ErrorException cachepath(r, :nope)    # unknown key

            m = manifest(r)
            @test m.config["gamma"] == 10.0 && m.setup["Z"] == 2.0e5
            @test m.provenance.repo_commit == "abc123"
            @test m.provenance.julia_version == "1.12.6"

            # an explicit token becomes the dashboard's auth cookie
            @test dashboard(url = fileurl(dir), token = "sekrit").headers ==
                  ["Cookie" => "research=sekrit"]

            rm(joinpath(data_store_dir(), u1); recursive = true, force = true)
        end
    end

    @testset "legacy index: sweep attribution + download-link fallback" begin
        mktempdir() do dir
            u = "cccc0000-0000-4000-8000-000000000003"
            mkpath(joinpath(dir, "data", u))
            S.serialize(joinpath(dir, "data", u, "powspec_$(u).jls"), (; x = 1))
            index = Dict(
                "runs" => [Dict("id" => u, "label" => "#001",
                    "params" => Dict("a0" => 1.0),
                    "data" => "/data/$(u)/hmaps_$(u).jls",
                    "plots" => Dict("powspec" => Dict("label" => "power spectrum",
                        "url" => "/plots/$(u)/powspec_$(u).png",
                        "data" => "/data/$(u)/powspec_$(u).jls")))],
                "sweeps" => [Dict("dir" => "camp_legacy",
                    "cells" => [Dict("coord" => [1], "run" => u)])])
            write(joinpath(dir, "index.json"), J.json(index))

            dash = dashboard(url = fileurl(dir))
            @test sweeps(dash) == ["camp_legacy"]       # attributed via the sweep's cells
            r = only(runs(sweep(dash, "camp_legacy")))
            ks = Set(x.key for x in caches(r))
            @test :powspec in ks && :hmaps in ks        # run data + plot-entry data links
            @test loadcache(r, :powspec).x == 1         # bytes unknown on legacy → no check
            rm(joinpath(data_store_dir(), u); recursive = true, force = true)
        end
    end

    @testset "auth configuration errors" begin
        withenv("EDM_DASHBOARD_URL" => nothing, "EDM_DASHBOARD_TOKEN" => nothing) do
            @test_throws ErrorException dashboard(:private)
        end
        @test_throws ErrorException dashboard(:elsewhere)
    end
end

@testset "timestamp_utc in every provenance block" begin
    prov = run_provenance(; run_id = "ts", gpu_backend = "rocm", repo_dir = pkgdir(RunManifests))
    @test haskey(prov, "timestamp_utc")
    @test endswith(prov["timestamp_utc"], "Z")
    # parseable after stripping the explicit-UTC suffix, and within a minute of now(UTC)
    t = DateTime(chop(prov["timestamp_utc"]))
    @test abs(Dates.value(now(UTC) - t)) < 60_000
    dir = mktempdir()
    write_run_manifest(dir; run_id = "ts", script = "x.jl")
    m = TOML.parsefile(joinpath(dir, "run_ts.toml"))
    @test endswith(m["provenance"]["timestamp_utc"], "Z")
    write_derived(dir; kind = "k", label = "l", run_id = "ts", plot = "p.png", source = "s")
    d = TOML.parsefile(joinpath(dir, only(filter(f -> startswith(f, "derived_"), readdir(dir)))))
    @test endswith(d["provenance"]["timestamp_utc"], "Z")
end

@testset "record_reduction! — re-reduce refreshes bytes, keeps untouched entries" begin
    dir = mktempdir()
    write(joinpath(dir, "maps.jls"), zeros(UInt8, 100))
    write(joinpath(dir, "traces.jls"), zeros(UInt8, 50))

    # first pass: two products staged, then finalized (run_cell.sh does mv .partial → .reduced)
    partial = record_reduction!(dir, "rr", "maps.jls")
    @test record_reduction!(dir, "rr", "traces.jls") == partial
    final = joinpath(dir, "rr.reduced")
    mv(partial, final)
    @test length(TOML.parsefile(final)["reduction"]) == 2

    # re-reduce: one product regrows on disk; only IT is re-recorded, then finalize again
    write(joinpath(dir, "maps.jls"), zeros(UInt8, 300))
    mv(record_reduction!(dir, "rr", "maps.jls"), final; force = true)
    m = TOML.parsefile(final)
    @test m["run_id"] == "rr"                                # header seeded from the final marker
    @test length(m["reduction"]) == 2                        # replaced, not duplicated
    bytes = Dict(e["file"] => e["bytes"] for e in m["reduction"])
    @test bytes["maps.jls"] == 300                           # byte size refreshed (collector byte-compares)
    @test bytes["traces.jls"] == 50                          # untouched entry survived the re-reduce
end
