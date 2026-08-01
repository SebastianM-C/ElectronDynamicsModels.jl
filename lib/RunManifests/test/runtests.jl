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
                "sweeps" => [Dict("dir" => "camp_a",
                    "axes" => [Dict("param" => "a0", "values" => [0.1, 0.5])],
                    "cells" => [Dict("coord" => [1], "run" => u1),
                                Dict("coord" => [2], "run" => u2)])],
                "standalone" => Any[], "comparisons" => Any[])
            write(joinpath(dir, "index.json"), J.json(index))

            dash = dashboard(url = fileurl(dir))
            @test campaigns(dash) == ["camp_a"]
            c = dash["camp_a"]
            @test length(runs(c)) == 2 && length(sweeps(c)) == 1
            r = only(runs(c; a0 = 0.1))                 # sweep-coordinate selection
            @test r.id == u1
            @test c["#001"].id == u1                    # gallery label …
            @test c["aaaa"].id == u1                    # … and uuid prefix
            @test_throws ErrorException c["zzzz"]       # no match
            @test isempty(runs(c; a0 = 7.7))

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
            @test campaigns(dash) == ["camp_legacy"]    # attributed via the sweep's cells
            r = only(runs(campaign(dash, "camp_legacy")))
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
