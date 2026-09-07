# ── The counting number type ─────────────────────────────────────────────────────────────
#
# `Counted{T} <: Real` (not `AbstractFloat`: Base has bit-level `AbstractFloat` methods that
# would be wrong for a wrapper). Only same-type binary methods are defined; mixed calls
# (`Counted{Float64}` with a `Float64`/`Int`/`Bool` literal, or `Counted{Float32}` with
# `Counted{Float64}`) go through Base's `f(x::Number, y::Number) = f(promote(x, y)...)`
# fallbacks and the promotion rules below. Defining mixed methods here would create
# ambiguities with those fallbacks.

"""
    Counted{T<:Real} <: Real

A scalar of type `T` whose arithmetic is counted (see [`CountedFloats`](@ref)). Construct
with `Counted(x)` / `Counted{T}(x)`; read back with [`value`](@ref). Construction,
conversion and predicates are never counted.
"""
struct Counted{T <: Real} <: Real
    v::T
    # Restricted to `Real` arguments: the default converting constructor `Counted{T}(v)` is
    # ambiguous with Base's `(::Type{T})(::Complex)`, `(::Type{T})(::TwicePrecision)` and
    # `(::Type{T})(::AbstractChar)` for `T <: Number`.
    Counted{T}(v::Real) where {T <: Real} = new{T}(convert(T, v))
end

Counted(v::T) where {T <: Real} = Counted{T}(v)
Counted(x::Counted) = x
Counted{T}(x::Counted) where {T <: Real} = Counted{T}(convert(T, x.v))

"""
    value(x::Counted) -> x.v
    value(x::Real) -> x

Unwrap a `Counted` (identity on other reals).
"""
value(x::Counted) = x.v
value(x::Real) = x

# ── conversion / promotion (never counted) ──
Base.promote_rule(::Type{Counted{T}}, ::Type{S}) where {T, S <: Real} = Counted{promote_type(T, S)}
Base.promote_rule(::Type{Counted{T}}, ::Type{Counted{S}}) where {T, S} = Counted{promote_type(T, S)}
Base.widen(::Type{Counted{T}}) where {T} = Counted{widen(T)}

(::Type{T})(x::Counted) where {T <: AbstractFloat} = T(x.v)
(::Type{T})(x::Counted) where {T <: Integer} = T(x.v)
Base.Bool(x::Counted) = Bool(x.v)   # disambiguates against Base's `Bool(::Real)`
Base.AbstractFloat(x::Counted) = float(x)
Base.float(x::Counted) = Counted(float(x.v))
Base.float(::Type{Counted{T}}) where {T} = Counted{float(T)}

Base.zero(::Type{Counted{T}}) where {T} = Counted(zero(T))
Base.one(::Type{Counted{T}}) where {T} = Counted(one(T))
Base.oneunit(::Type{Counted{T}}) where {T} = Counted(oneunit(T))
Base.typemin(::Type{Counted{T}}) where {T} = Counted(typemin(T))
Base.typemax(::Type{Counted{T}}) where {T} = Counted(typemax(T))
Base.floatmin(::Type{Counted{T}}) where {T} = Counted(floatmin(T))
Base.floatmax(::Type{Counted{T}}) where {T} = Counted(floatmax(T))
Base.eps(::Type{Counted{T}}) where {T} = Counted(eps(T))
Base.eps(x::Counted) = Counted(eps(x.v))
Base.precision(::Type{Counted{T}}) where {T} = precision(T)
Base.nextfloat(x::Counted) = Counted(nextfloat(x.v))
Base.prevfloat(x::Counted) = Counted(prevfloat(x.v))

Base.hash(x::Counted, h::UInt) = hash(x.v, h)
Base.isequal(x::Counted, y::Counted) = isequal(x.v, y.v)
Base.show(io::IO, x::Counted) = print(io, "Counted(", x.v, ")")

# ── predicates (never counted) ──
for p in (:isnan, :isfinite, :isinf, :iszero, :isone, :signbit, :isinteger)
    @eval Base.$p(x::Counted) = $p(x.v)
end

# ── counted arithmetic ──
for (op, cat) in ((:+, ADD), (:-, ADD), (:*, MUL), (:/, DIV))
    @eval @inline function Base.$op(x::Counted{T}, y::Counted{T}) where {T}
        _bump!($cat, T)
        return Counted($op(x.v, y.v))
    end
end
@inline function Base.:-(x::Counted{T}) where {T}
    _bump!(OTHER, T)
    return Counted(-x.v)
end
Base.:+(x::Counted) = x
@inline function Base.inv(x::Counted{T}) where {T}
    _bump!(DIV, T)
    return Counted(inv(x.v))
end
@inline function Base.sqrt(x::Counted{T}) where {T}
    _bump!(SQRT, T)
    return Counted(sqrt(x.v))
end
for op in (:fma, :muladd)
    @eval @inline function Base.$op(x::Counted{T}, y::Counted{T}, z::Counted{T}) where {T}
        _bump!(FMA, T)
        return Counted($op(x.v, y.v, z.v))
    end
end
# `x ^ y` with a real exponent is one pow call. Integer exponents keep Base's generic path
# (`power_by_squaring`, i.e. counted multiplications) — the literal `x^2` / `x^3` forms below
# count exactly like the hardware-float `literal_pow` specialisations (1 and 2 mul).
@inline function Base.:^(x::Counted{T}, y::Counted{T}) where {T}
    _bump!(POW, T)
    return Counted(x.v^y.v)
end
@inline Base.literal_pow(::typeof(^), x::Counted, ::Val{0}) = one(x)
@inline Base.literal_pow(::typeof(^), x::Counted, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::Counted, ::Val{2}) = x * x
@inline Base.literal_pow(::typeof(^), x::Counted, ::Val{3}) = x * x * x
@inline Base.literal_pow(::typeof(^), x::Counted, ::Val{-1}) = inv(x)
@inline Base.literal_pow(::typeof(^), x::Counted, ::Val{-2}) = (i = inv(x); i * i)

# ── comparisons (category :cmp) ──
for op in (:<, :<=, :(==), :isless)
    @eval @inline function Base.$op(x::Counted{T}, y::Counted{T}) where {T}
        _bump!(CMP, T)
        return $op(x.v, y.v)
    end
end

# ── non-FLOP value operations (category :other) ──
for op in (:abs, :sign, :floor, :ceil, :trunc, :round)
    @eval @inline function Base.$op(x::Counted{T}) where {T}
        _bump!(OTHER, T)
        return Counted($op(x.v))
    end
end
for op in (:floor, :ceil, :trunc, :round)
    @eval @inline function Base.$op(::Type{I}, x::Counted{T}) where {I <: Integer, T}
        _bump!(OTHER, T)
        return $op(I, x.v)
    end
end
@inline function Base.round(x::Counted{T}, r::RoundingMode) where {T}
    _bump!(OTHER, T)
    return Counted(round(x.v, r))
end
for op in (:mod, :rem, :copysign, :flipsign, :div, :fld, :cld)
    @eval @inline function Base.$op(x::Counted{T}, y::Counted{T}) where {T}
        _bump!(OTHER, T)
        return Counted($op(x.v, y.v))
    end
end
@inline function Base.ldexp(x::Counted{T}, n::Integer) where {T}
    _bump!(OTHER, T)
    return Counted(ldexp(x.v, n))
end
@inline function Base.rem2pi(x::Counted{T}, r::RoundingMode) where {T}
    _bump!(OTHER, T)
    return Counted(rem2pi(x.v, r))
end
