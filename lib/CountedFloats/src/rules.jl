# ── Generated math surface ───────────────────────────────────────────────────────────────
#
# Every unary and binary Base function DiffRules knows a derivative for gets a counted
# same-type method (ForwardDiff's approach to its Dual surface). Everything not listed in
# `_CATEGORY` is a transcendental call (one `:trans` count); `_SKIP` holds the functions with
# hand-written methods in type.jl or a non-Counted second argument.

const _SKIP = Set{Tuple{Symbol, Symbol}}([
    (:Base, :+), (:Base, :-), (:Base, :*), (:Base, :/), (:Base, :^),
    (:Base, :sqrt), (:Base, :inv), (:Base, :abs), (:Base, :abs2),   # abs2(x::Real) = x*x counts itself
    (:Base, :max), (:Base, :min), (:Base, :ifelse),                  # Base: comparisons / selection
    (:Base, :muladd), (:Base, :fma),
    (:Base, :mod), (:Base, :rem), (:Base, :ldexp), (:Base, :rem2pi),
])

const _CATEGORY = Dict{Symbol, Int}(
    :deg2rad => MUL, :rad2deg => MUL,   # x * (π/180)
    :mod2pi => OTHER,
)

const GENERATED = Tuple{Symbol, Int}[]   # (function, arity) actually defined, for tests/docs

for (M, f, arity) in DiffRules.diffrules(filter_modules = nothing)
    M === :Base || continue
    (M, f) in _SKIP && continue
    cat = get(_CATEGORY, f, TRANS)
    if arity == 1
        @eval @inline function Base.$f(x::Counted{T}) where {T}
            _bump!($cat, T)
            return Counted(Base.$f(x.v))
        end
        push!(GENERATED, (f, 1))
    elseif arity == 2
        @eval @inline function Base.$f(x::Counted{T}, y::Counted{T}) where {T}
            _bump!($cat, T)
            return Counted(Base.$f(x.v, y.v))
        end
        push!(GENERATED, (f, 2))
    end
end
