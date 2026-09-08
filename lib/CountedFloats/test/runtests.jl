using CountedFloats
using CountedFloats: CATEGORIES, FLOP_CATEGORIES, GENERATED
using Test
using Aqua

const CF = Counted{Float64}

# Counts with exactly these category totals (all element types summed) and nothing else.
function only_counts(c::Counts; kw...)
    expected = Dict{Symbol, Int}(kw)
    return all(cat -> c[cat] == get(expected, cat, 0), CATEGORIES)
end

@testset "CountedFloats" begin
    @testset "Aqua" begin
        Aqua.test_all(CountedFloats)
    end

    @testset "construction, conversion, predicates are free" begin
        reset!()
        x = Counted(1.5)
        @test x isa CF
        @test value(x) == 1.5
        @test value(2.0) == 2.0
        @test Counted(x) === x
        @test Counted{Float32}(x) === Counted(1.5f0)
        @test Counted{Float64}(1) === Counted(1.0) && Bool(Counted(1.0)) === true
        @test Float64(x) === 1.5
        @test Int(Counted(3.0)) === 3
        @test float(Counted(2)) === Counted(2.0)
        @test AbstractFloat(x) === x
        @test zero(CF) === Counted(0.0) && one(CF) === Counted(1.0)
        @test eps(CF) === Counted(eps(Float64)) && eps(x) === Counted(eps(1.5))
        @test typemax(CF) === Counted(Inf) && floatmin(CF) === Counted(floatmin(Float64))
        @test precision(CF) == 53
        @test nextfloat(x) === Counted(nextfloat(1.5))
        @test isnan(Counted(NaN)) && isfinite(x) && !isinf(x) && iszero(zero(CF)) && isone(one(CF))
        @test signbit(Counted(-1.0)) && isinteger(Counted(2.0))
        @test isequal(x, Counted(1.5)) && hash(x) == hash(1.5)
        @test sprint(show, x) == "Counted(1.5)"
        @test promote_type(CF, Float64) === CF
        @test promote_type(CF, Int) === CF
        @test promote_type(CF, Bool) === CF
        @test promote_type(Counted{Float32}, CF) === CF
        @test promote_type(Float32, CF) === CF
        @test promote_type(CF, typeof(π)) === CF
        @test iszero(counts())
    end

    @testset "exact unit counts" begin
        x, y, z = Counted(1.5), Counted(2.5), Counted(-0.5)
        @test only_counts(@count(x + y); add = 1)
        @test only_counts(@count(x - y); add = 1)
        @test only_counts(@count(x * y); mul = 1)
        @test only_counts(@count(x / y); div = 1)
        @test only_counts(@count(inv(x)); div = 1)
        @test only_counts(@count(sqrt(x)); sqrt = 1)
        @test only_counts(@count(x * y + z); mul = 1, add = 1)
        @test only_counts(@count(fma(x, y, z)); fma = 1)
        @test only_counts(@count(muladd(x, y, z)); fma = 1)
        @test only_counts(@count(x^2); mul = 1)
        @test only_counts(@count(x^3); mul = 2)
        @test only_counts(@count(x^-1); div = 1)
        @test only_counts(@count(x^y); pow = 1)
        @test only_counts(@count(exp(x)); trans = 1)
        @test only_counts(@count(atan(x, y)); trans = 1)
        @test only_counts(@count(hypot(x, y)); trans = 1)
        @test only_counts(@count(deg2rad(x)); mul = 1)
        @test only_counts(@count(abs2(x)); mul = 1)
        @test only_counts(@count(-x); other = 1)
        @test only_counts(@count(abs(z)); other = 1)
        @test only_counts(@count(floor(Int, x)); other = 1)
        @test only_counts(@count(round(x, RoundNearest)); other = 1)
        @test only_counts(@count(x < y); cmp = 1)
        @test only_counts(@count(x <= y); cmp = 1)
        @test only_counts(@count(x > y); cmp = 1)
        @test only_counts(@count(x == y); cmp = 1)
        @test only_counts(@count(max(x, y)); cmp = 1)
        @test only_counts(@count(min(x, y)); cmp = 1)
        # `clamp` short-circuits: 1 or 2 comparisons, never a FLOP
        c = @count clamp(x, z, y)
        @test c[:cmp] in (1, 2) && flops(c) == 0
        @test only_counts(@count(ifelse(true, x, y)))
        @test only_counts(@count(copysign(x, z)); other = 1)
        # results are right
        @test value(x * y + z) == 1.5 * 2.5 - 0.5
        @test value(sqrt(y)) == sqrt(2.5)
        @test floor(Int, y) === 2 && ceil(Int, y) === 3 && round(Int, y) === 2
        @test max(x, y) === y && clamp(Counted(9.0), z, y) === y
    end

    @testset "promotion with literals counts once" begin
        x = Counted(1.5)
        @test only_counts(@count(0.5 * x); mul = 1)
        @test only_counts(@count(x / 2); div = 1)
        @test only_counts(@count(2 * x); mul = 1)
        @test only_counts(@count((1 / 6) * x); mul = 1)   # 1/6 folds to a Float64 constant
        @test only_counts(@count(x + 1); add = 1)
        @test only_counts(@count(x * true); mul = 1)
        @test only_counts(@count(x < 2.0); cmp = 1)
        @test only_counts(@count(clamp(x, 0, 2)); cmp = 2)   # 1.5 inside → both comparisons run
        @test (0.5 * x) isa CF && (x + 1) isa CF
    end

    @testset "element-type buckets" begin
        a, b = Counted(1.0f0), Counted(2.0f0)
        x, y = Counted(1.0), Counted(2.0)
        c = @count begin
            a * b
            x * y
            x * y
            a * y   # promotes to Float64
        end
        @test c[:mul] == 4
        @test c[:mul, Float32] == 1
        @test c[:mul, Float64] == 3
        @test flops(c) == 4 && flops(c, Float32) == 1 && flops(c, Float64) == 3
        @test (a * y) isa CF
        @test NamedTuple(c) == (; add = 0, mul = 4, div = 0, sqrt = 0, fma = 0, pow = 0, trans = 0, cmp = 0, other = 0)
    end

    @testset "Counts arithmetic and flops weights" begin
        x, y, z = Counted(1.5), Counted(2.5), Counted(-0.5)
        c = @count (x * y + z) / sqrt(x)
        @test c[:mul] == 1 && c[:add] == 1 && c[:div] == 1 && c[:sqrt] == 1
        @test flops(c) == 4
        @test flops(c; div = 4, sqrt = 8) == 14
        @test c + c == c * 2 == 2 * c
        @test (c * 2) / 2 == c
        @test div(c * 3, 2) == c   # floor division
        @test_throws InexactError (c * 3) / 2
        @test iszero(c - c) && zero(Counts) == c - c
        @test occursin("mul", sprint(show, c))
        @test occursin("Float64", sprint(show, MIME("text/plain"), c))
        cf = @count fma(x, y, z)
        @test flops(cf) == 2 && flops(cf; fma = 1) == 1
        r = count_ops(() -> x * y)
        @test r.counts[:mul] == 1 && r.value === x * y
    end

    @testset "generated Base surface" begin
        @test length(GENERATED) > 40
        # first sample input inside each function's domain (asec needs |x| ≥ 1, asech (0, 1], …)
        function sample_args(g, arity)
            for a in (0.5, 1.5, 2.5, 0.75)
                args = arity == 1 ? (a,) : (a, 0.25)
                r = try
                    g(args...)
                catch e
                    e isa DomainError || rethrow()
                    continue
                end
                isfinite(r) && return args
            end
            return nothing
        end
        for (f, arity) in GENERATED
            g = getfield(Base, f)
            args = sample_args(g, arity)
            args === nothing && continue
            cargs = Counted.(args)
            y = g(cargs...)
            @test y isa CF
            @test value(y) == g(args...)
            c = @count g(cargs...)
            @test sum(NamedTuple(c)) == 1                    # exactly one counted call
            @test flops(c) == (f === :mod2pi ? 0 : 1)        # mod2pi is bookkeeping, not a FLOP
        end
    end

    @testset "threads: increments land in per-thread columns, sums are exact" begin
        n = 10_000
        v = Counted.(rand(n))
        c = @count begin
            Threads.@threads for i in 1:n
                v[i] * v[i] + v[i]
            end
        end
        @test c[:mul] == n && c[:add] == n
        c2 = @count begin
            tasks = [Threads.@spawn sum(w -> w * w, v) for _ in 1:4]
            foreach(fetch, tasks)
        end
        @test c2[:mul] == 4n && c2[:add] == 4 * (n - 1)
        reset!()
        @test iszero(counts())
    end

    @testset "arrays and reductions work as plain Reals" begin
        v = Counted.([1.0, 2.0, 3.0])
        @test eltype(v) === CF
        c = @count sum(v)
        @test c[:add] == 2
        @test value(sum(v)) == 6.0
        w = similar(v)
        w .= v .* 2 .+ 1
        @test value.(w) == [3.0, 5.0, 7.0]
        @test only_counts(@count(v .* 2); mul = 3)
    end
end
