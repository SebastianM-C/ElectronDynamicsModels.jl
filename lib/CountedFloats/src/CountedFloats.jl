"""
    CountedFloats

Count floating-point operations of type-generic Julia code by running it on a counting
number type. `Counted{T} <: Real` wraps a scalar and bumps a per-category, per-element-type
counter on every arithmetic call it receives; the code under measurement is executed
normally on the CPU, so what is counted is the arithmetic *as written* at the Julia level
(after inlining, before LLVM's CSE / dead-code elimination / FMA contraction). Nothing that
leaves Julia (BLAS, LAPACK, FFTW, other `ccall`s) and nothing pinned to a concrete float type
is visible. See the README for conventions and limitations.

    c = @count my_kernel(Counted.(x), Counted(0.5))   # -> Counts
    flops(c)                                          # add + mul + div + sqrt + 2 fma + pow + trans
    c[:mul], c[:mul, Float64]                         # per category, per element type
"""
module CountedFloats

using DiffRules

export Counted, Counts, @count, count_ops, counts, reset!, flops, value

include("counters.jl")
include("type.jl")
include("rules.jl")

function __init__()
    _ensure_table!()
    return nothing
end

end # module
