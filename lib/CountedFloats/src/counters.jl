# ── Counter storage ──────────────────────────────────────────────────────────────────────
#
# Categories × element-type slots, one column of counters per Julia thread. A thread only ever
# writes its own column, so increments need no atomics: a task migrating between threads just
# splits its increments across two columns and the column sum stays exact. Columns are padded
# to a cache-line multiple so neighbouring threads never false-share. Threads adopted after the
# table was sized (rare: foreign threads) fall back to a small atomic overflow vector.

const CATEGORIES = (:add, :mul, :div, :sqrt, :fma, :pow, :trans, :cmp, :other)
const NCAT = length(CATEGORIES)
const ADD, MUL, DIV, SQRT, FMA, POW, TRANS, CMP, OTHER = 1:NCAT
const FLOP_CATEGORIES = (:add, :mul, :div, :sqrt, :fma, :pow, :trans)   # cmp/other are not FLOPs

const SLOT_TYPES = (Float16, Float32, Float64, Any)   # Any = every other element type
const NSLOT = length(SLOT_TYPES)
const NCELL = NCAT * NSLOT

# Element-type slot, resolved at compile time for concrete T.
@inline _slot(::Type{Float16}) = 1
@inline _slot(::Type{Float32}) = 2
@inline _slot(::Type{Float64}) = 3
@inline _slot(::Type) = 4

@inline _cell(cat::Int, slot::Int) = (slot - 1) * NCAT + cat

const STRIDE = 64   # ≥ NCELL, a multiple of 8 Ints = whole cache lines per thread column
const TABLE = Ref{Matrix{Int}}(zeros(Int, STRIDE, 1))
const OVERFLOW = [Threads.Atomic{Int}(0) for _ in 1:NCELL]

# Size the table for the current thread count (at load and on `reset!`). NOT safe to call while
# counted work runs on other threads (increments to the old table would be lost).
function _ensure_table!()
    n = Threads.maxthreadid()
    if size(TABLE[], 2) < n
        TABLE[] = zeros(Int, STRIDE, n)
    end
    return nothing
end

@inline function _bump!(cat::Int, ::Type{T}) where {T}
    idx = _cell(cat, _slot(T))
    tid = Threads.threadid()
    tbl = TABLE[]
    if tid <= size(tbl, 2)
        @inbounds tbl[idx, tid] += 1
    else
        Threads.atomic_add!(OVERFLOW[idx], 1)
    end
    return nothing
end

# ── Counts: an immutable snapshot / difference of the counter table ──────────────────────

"""
    Counts

Immutable operation counts by category (`$(CATEGORIES)`) and element type
(`$(SLOT_TYPES)`; the last slot collects every other type). Obtained from [`counts`](@ref),
[`@count`](@ref) or [`count_ops`](@ref). Index with `c[:mul]` (summed over element types) or
`c[:mul, Float64]`; convert to per-category sums with `NamedTuple(c)`; weight into a FLOP total
with [`flops`](@ref). Supports `+`, `-`, `*`/`div`/`rem` by an integer, and `/` by an integer
(exact — throws `InexactError` if any cell is not divisible, which is what you want when
fitting a per-iteration cost by differencing two measurements).
"""
struct Counts
    data::NTuple{NCELL, Int}
end

Counts() = Counts(ntuple(_ -> 0, Val(NCELL)))

function _catindex(cat::Symbol)
    i = findfirst(==(cat), CATEGORIES)
    i === nothing && throw(ArgumentError("unknown category $cat; expected one of $(CATEGORIES)"))
    return i
end

Base.getindex(c::Counts, cat::Symbol) = (i = _catindex(cat); sum(c.data[_cell(i, s)] for s in 1:NSLOT))
Base.getindex(c::Counts, cat::Symbol, ::Type{T}) where {T} = c.data[_cell(_catindex(cat), _slot(T))]

Base.NamedTuple(c::Counts) = NamedTuple{CATEGORIES}(ntuple(i -> c[CATEGORIES[i]], Val(NCAT)))

Base.:+(a::Counts, b::Counts) = Counts(a.data .+ b.data)
Base.:-(a::Counts, b::Counts) = Counts(a.data .- b.data)
Base.:*(a::Counts, n::Integer) = Counts(a.data .* n)
Base.:*(n::Integer, a::Counts) = a * n
Base.div(a::Counts, n::Integer) = Counts(div.(a.data, n))
Base.rem(a::Counts, n::Integer) = Counts(rem.(a.data, n))
function Base.:/(a::Counts, n::Integer)
    all(iszero, rem.(a.data, n)) || throw(InexactError(:/, Counts, (a, n)))
    return div(a, n)
end
Base.iszero(c::Counts) = all(iszero, c.data)
Base.zero(::Type{Counts}) = Counts()
Base.zero(::Counts) = Counts()

"""
    flops(c::Counts; fma = 2, div = 1, sqrt = 1, pow = 1, trans = 1) -> Int
    flops(c::Counts, T::Type; kwargs...) -> Int

Weighted FLOP total of `c` (all element types, or only element type `T`): `add` and `mul`
count 1 each; the keyword weights set the cost of a division, square root, fused
multiply-add (2 by the usual benchmark convention), power and transcendental call.
Comparisons and the `other` bucket (negation, abs, rounding, sign copies …) are never FLOPs.
"""
function flops(c::Counts; fma::Real = 2, div::Real = 1, sqrt::Real = 1, pow::Real = 1, trans::Real = 1)
    return c[:add] + c[:mul] + div * c[:div] + sqrt * c[:sqrt] + fma * c[:fma] +
        pow * c[:pow] + trans * c[:trans]
end
function flops(c::Counts, ::Type{T}; fma::Real = 2, div::Real = 1, sqrt::Real = 1, pow::Real = 1,
        trans::Real = 1) where {T}
    return c[:add, T] + c[:mul, T] + div * c[:div, T] + sqrt * c[:sqrt, T] + fma * c[:fma, T] +
        pow * c[:pow, T] + trans * c[:trans, T]
end

function Base.show(io::IO, c::Counts)
    nt = NamedTuple(c)
    parts = ["$k=$v" for (k, v) in pairs(nt) if v != 0]
    print(io, "Counts(", join(parts, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", c::Counts)
    slots = [s for s in 1:NSLOT if any(c.data[_cell(i, s)] != 0 for i in 1:NCAT)]
    if isempty(slots)
        print(io, "Counts: no operations recorded")
        return
    end
    names = [s == NSLOT ? "other" : string(SLOT_TYPES[s]) for s in slots]
    w = max(8, maximum(length, names))
    println(io, "Counts (flops = ", flops(c), "):")
    print(io, rpad("", 8))
    for n in names
        print(io, lpad(n, w + 2))
    end
    println(io)
    for i in 1:NCAT
        vals = [c.data[_cell(i, s)] for s in slots]
        all(iszero, vals) && continue
        print(io, rpad(String(CATEGORIES[i]), 8))
        for v in vals
            print(io, lpad(string(v), w + 2))
        end
        println(io)
    end
end

# ── Public counter access ────────────────────────────────────────────────────────────────

"""
    counts() -> Counts

Snapshot of the global counters (all threads) since the last [`reset!`](@ref).
"""
function counts()
    tbl = TABLE[]
    data = ntuple(Val(NCELL)) do i
        s = 0
        @inbounds for j in axes(tbl, 2)
            s += tbl[i, j]
        end
        s + OVERFLOW[i][]
    end
    return Counts(data)
end

"""
    reset!()

Zero the global counters. Do not call while counted work is running on other threads.
"""
function reset!()
    _ensure_table!()
    fill!(TABLE[], 0)
    for o in OVERFLOW
        o[] = 0
    end
    return nothing
end

"""
    @count expr -> Counts

Evaluate `expr` and return the operations it performed on `Counted` values (the difference
of two [`counts`](@ref) snapshots, so it composes with other measurements and never resets
anything). Work spawned by `expr` on other threads is included; unrelated counted work running
concurrently on other tasks is included too — measure one thing at a time.
"""
macro count(expr)
    return quote
        local c0 = counts()
        $(esc(expr))
        counts() - c0
    end
end

"""
    count_ops(f) -> (counts::Counts, value)

Function form of [`@count`](@ref): run `f()` and return both its operation counts and its
return value.
"""
function count_ops(f)
    c0 = counts()
    value = f()
    return (counts = counts() - c0, value = value)
end
