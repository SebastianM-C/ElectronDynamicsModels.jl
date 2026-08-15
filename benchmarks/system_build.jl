#     julia --project=benchmarks benchmarks/system_build.jl              # full table
#     julia --project=benchmarks benchmarks/system_build.jl planewave    # one laser, cold
#
# Time-to-first-X of the symbolic pipeline, per laser family: the wall time paid on the
# FIRST call, in a fresh session, for (1) System construction, (2) mtkcompile,
# (3) ODEProblem + first solve. No warm-up — that first-call compilation is the quantity of
# interest. Each stage is a single-shot @elapsed. The machinery is largely laser-independent
# and compiles on first use, so each laser is measured in its OWN cold process (a shared
# session would bill only the first laser for it). No args: orchestrate one fresh julia per
# laser and print the table; a laser name: measure just that one.

using Printf

const LASERS = (:planewave, :gauss, :laguerregauss)
fmt(t) = t < 1 ? @sprintf("%.0f ms", 1.0e3 * t) : @sprintf("%.1f s", t)

# ── orchestrator (no args): one fresh julia per laser. Above the heavy `using` so the parent
# stays lightweight — it only spawns children.
if isempty(ARGS)
    proj = dirname(Base.active_project())

    # Prime package load once (no pipeline work) so every laser is measured with the .ji images
    # equally resident. Not a compile warm-up: each child still pays full first-call cost; this
    # only removes the cold-disk paging that would otherwise inflate whichever laser runs first.
    println(stderr, "priming package load (no measurement)…")
    run(pipeline(`$(Base.julia_cmd()) --project=$proj -e "using ElectronDynamicsModels, ModelingToolkit, OrdinaryDiffEqVerner, SciMLBase, StaticArrays"`;
        stdout = devnull, stderr = devnull))

    results = Dict{Symbol, NTuple{3, Float64}}()
    for kind in LASERS
        println(stderr, "measuring $kind in a cold process…")
        out = read(`$(Base.julia_cmd()) --project=$proj $(@__FILE__) $kind`, String)
        m = match(r"BENCHRESULT (\w+) ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)", out)
        m === nothing && error("no BENCHRESULT from the $kind child:\n$out")
        results[kind] = (parse(Float64, m[2]), parse(Float64, m[3]), parse(Float64, m[4]))
    end

    @printf("\njulia %s  (each laser in its OWN cold process — true first-time latency)\n\n", VERSION)
    @printf("%-28s %14s %14s %14s\n", "", "plane wave", "Gauss", "Laguerre-Gauss")
    for (i, label) in enumerate(("System construction", "mtkcompile", "ODEProblem + first solve"))
        @printf("%-28s %14s %14s %14s\n", label, (fmt(results[k][i]) for k in LASERS)...)
    end

    println("\nLaTeX table rows:")
    for (i, label) in enumerate(("\\texttt{System} construction",
        "\\texttt{mtkcompile}", "\\texttt{ODEProblem} + first \\texttt{solve}"))
        println("      ", label, "  & ", join((fmt(results[k][i]) for k in LASERS), " & "), " \\\\")
    end
    exit()
end

# ── child (laser-name arg): load the stack and measure that ONE laser, cold, once ──
using ElectronDynamicsModels
using ModelingToolkit
using OrdinaryDiffEqVerner
using SciMLBase, StaticArrays

const c_au = 137.03599908330932
const ω = 0.057                    # 800 nm, atomic units
const λ_val = c_au * 2π / ω
const w₀_val = 75 * λ_val
const a₀_val = 2.0
const τspan = (0.0, 2π / ω)        # one cycle is enough for a first call

function build_laser(kind, world)
    if kind === :planewave
        @named laser = PlaneWave(; amplitude = a₀_val * c_au * ω, frequency = ω, world)
    elseif kind === :gauss
        @named laser = GaussLaser(; wavelength = λ_val, a0 = a₀_val, beam_waist = w₀_val, world)
    elseif kind === :laguerregauss
        @named laser = LaguerreGaussLaser(;
            wavelength = λ_val, a0 = a₀_val, beam_waist = w₀_val,
            radial_index = 1, azimuthal_index = 1,
            temporal_profile = :constant, world)
    else
        error("unknown laser $kind; expected one of $(LASERS)")
    end
    return laser
end

# One cold measurement of the three pipeline stages for a single laser. No warm-up.
function measure(kind)
    @named world = Worldline(:τ, :atomic)
    laser = build_laser(kind, world)

    t_build = @elapsed begin
        @named electron = LandauLifshitzElectron(; laser)
    end
    t_compile = @elapsed sys = mtkcompile(electron, allow_symbolic = true)
    t_first = @elapsed begin
        op = [sys.x => zeros(4), sys.u => [c_au, 0.0, 0.0, 0.0]]
        prob = ODEProblem{false, SciMLBase.FullSpecialize}(sys, op, τspan;
            u0_constructor = SVector{8}, fully_determined = true)
        solve(prob, Vern9(); abstol = 1.0e-10, reltol = 1.0e-10)
    end
    return (; t_build, t_compile, t_first)
end

kind = Symbol(ARGS[1])
kind in LASERS || error("unknown laser $kind; expected one of $(LASERS)")
r = measure(kind)

@printf("julia %s, threads = %d\n", VERSION, Threads.nthreads())
@printf("%-28s %14s\n", string(kind), "first-call")
for (label, key) in (("System construction", :t_build),
    ("mtkcompile", :t_compile), ("ODEProblem + first solve", :t_first))
    @printf("%-28s %14s\n", label, fmt(getfield(r, key)))
end
# machine-readable line the orchestrator parses:
@printf("BENCHRESULT %s %.6f %.6f %.6f\n", kind, r.t_build, r.t_compile, r.t_first)
