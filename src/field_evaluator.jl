"""
    FieldEvaluator(laser, ref_frame)

Evaluate electromagnetic fields E(x,y,z,t) and B(x,y,z,t) at arbitrary spacetime
points without running an ODE solver. Internally wraps the laser in a ClassicalElectron
to produce a compilable system, then uses `build_explicit_observed_function` to get
a fast compiled function for the field observables.

# Usage
The input is `[t, x, y, z]` (bare time, not c*t). The speed of light `c` is
obtained from the reference frame and applied internally.

```julia
@named ref_frame = ProperFrame(:atomic)
@named laser = GaussLaser(; wavelength=1.0, a0=1.0, ref_frame)
fe = FieldEvaluator(laser, ref_frame)
result = fe([t, x, y, z])  # (E = [...], B = [...])
```
"""
struct FieldEvaluator{F,S,P,T}
    h::F           # compiled observed function for [E; B]
    set_x::S       # out-of-place setter for x coordinates
    prob::P        # reference problem (holds initialized parameters)
    c::T           # speed of light from the reference frame
end

function FieldEvaluator(laser, ref_frame)
    # Internally create a minimal electron system — just for compilation
    @named _probe = ClassicalElectron(; laser, ref_frame)
    sys = mtkcompile(_probe)

    laser_name = nameof(laser)
    laser_sys = getproperty(sys, laser_name)

    E_B_syms = SymbolicT[laser_sys.E[i] for i in 1:3]
    append!(E_B_syms, SymbolicT[laser_sys.B[i] for i in 1:3])
    h = build_explicit_observed_function(sys, E_B_syms)

    # Create problem to initialize parameters (trivial initial conditions)
    u0 = [sys.x => zeros(SVector{4}), sys.u => SVector{4}(0.0, 0.0, 0.0, 1.0)]
    prob = ODEProblem(sys, u0, (0.0, 1.0))

    # Build coordinate setter
    set_x = setsym_oop(prob, sys.x)

    # Extract speed of light from the system constants
    c_val = prob.ps[sys.c]

    FieldEvaluator(h, set_x, prob, c_val)
end

function Base.show(io::IO, ::MIME"text/plain", fe::FieldEvaluator)
    sys = fe.prob.f.sys
    ps = parameters(sys)
    sep = Symbolics.NAMESPACE_SEPARATOR
    laser_ps = filter(p -> occursin(sep, string(p)), ps)
    println(io, "FieldEvaluator with parameters:")
    cio = IOContext(io, :compact => true)
    for p in laser_ps
        name = last(split(string(p), sep))
        val = fe.prob.ps[p]
        println(cio, "  ", name, " = ", val)
    end
    print(cio, "  c = ", fe.c)
end

function Base.show(io::IO, fe::FieldEvaluator)
    print(io, "FieldEvaluator(...)")
end

function (fe::FieldEvaluator)(txyz)
    t, x, y, z = txyz
    u0, p = fe.set_x(fe.prob, SVector{4}(fe.c * t, x, y, z))
    result = fe.h(u0, p, 0.0)
    return (E = result[SVector{3}(1:3)], B = result[SVector{3}(4:6)])
end
