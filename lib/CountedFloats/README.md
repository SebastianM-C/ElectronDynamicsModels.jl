# CountedFloats.jl

Count the floating-point operations of type-generic Julia code by running it on a counting
number type. No compiler hooks, no Cassette: `Counted{T} <: Real` wraps a scalar and bumps a
counter on every arithmetic call it receives, so the code under measurement runs unmodified on
the CPU and the counts follow the code as it evolves.

```julia
using CountedFloats

f(x, y) = sqrt(x * x + y * y) / 2

c = @count f(Counted(3.0), Counted(4.0))
# Counts(add=1, mul=2, div=1, sqrt=1)
flops(c)              # 5   (add + mul + div + sqrt + 2·fma + pow + trans)
c[:mul]               # 2
c[:mul, Float64]      # 2   — counts are bucketed by element type (Float16/32/64/other)
NamedTuple(c)         # per-category sums, ready for a manifest
```

`@count expr` returns the difference of two snapshots of the global counters (`counts()`),
so it composes and never resets anything; `count_ops(f)` returns counts and value;
`reset!()` zeroes the counters. `Counts` values support `+`, `-`, `*`/`div`/`rem` by an
integer and an *exact* `/` by an integer (throws `InexactError` otherwise) — handy for
fitting a per-iteration cost by differencing two problem sizes.

## What is counted

| category | operations | `flops` weight |
|---|---|---|
| `add` | `+`, `-` (binary) | 1 |
| `mul` | `*`, `abs2`, `deg2rad`, `rad2deg` | 1 |
| `div` | `/`, `inv` | `div = 1` |
| `sqrt` | `sqrt` | `sqrt = 1` |
| `fma` | `fma`, `muladd` | `fma = 2` |
| `pow` | `x ^ y` with a `Counted` exponent | `pow = 1` |
| `trans` | every other Base function DiffRules has a rule for (`exp`, `log`, `sin`, `atan`, `hypot`, `cbrt`, …) | `trans = 1` |
| `cmp` | `<`, `<=`, `==`, `isless` (`>`, `>=`, `max`, `min`, `clamp` reduce to these) | 0 |
| `other` | unary `-`, `abs`, `sign`, rounding, `mod`/`rem`, `copysign`/`flipsign`, `ldexp` | 0 |

Integer exponents keep Base's generic path: the literal forms `x^2` / `x^3` count 1 / 2
multiplications exactly like the hardware-float specialisations; `x^n` for a runtime `n` is
Base's `power_by_squaring`, i.e. counted multiplications. Construction, conversion,
predicates (`isnan`, `iszero`, …) and `ifelse` are free.

## What the number means

Counts are the arithmetic **as written at the Julia level**: after Julia's inlining and
promotion, before LLVM's common-subexpression elimination, dead-code elimination and
FMA contraction. That is the usual "algorithmic FLOP" definition (the one used to report
GFLOP/s of a solver against a hardware peak), not a hardware instruction count. Hardware
costs of `div`, `sqrt`, `pow` and transcendentals differ from 1 — the keyword weights of
`flops` are there to apply your own convention.

## Limitations

- **The measured code must be type-generic.** Anything pinned to a concrete float type
  (`::Float64` annotations, `zeros(n)`, `Float64(x)` casts) either errors or silently stops
  counting from that point on.
- **Nothing outside Julia is visible**: BLAS, LAPACK, FFTW and other `ccall`ed libraries
  contribute zero. Generic Julia fallbacks (e.g. `Matrix{Counted{Float64}}` products) are
  counted, slowly.
- Only same-type binary methods are defined; mixed operands rely on Base's promotion
  fallbacks (`f(x::Number, y::Number) = f(promote(x, y)...)`). Nesting with other wrapper
  types (`ForwardDiff.Dual`, `Measurement`) should be done explicitly, e.g.
  `Counted{Dual{…}}`.
- Counters are global. Each thread owns a column (no atomics, exact sums under task
  migration), so work spawned inside `@count` is included — and so is *unrelated* counted
  work running concurrently on other tasks. Measure one thing at a time; never `reset!`
  while counted work runs elsewhere.
- Data-dependent branches (`clamp`, `abs(x::Real)`'s `ifelse`) can make `cmp`/`other`
  counts vary between inputs; FLOP categories of branch-free code are exactly reproducible.
