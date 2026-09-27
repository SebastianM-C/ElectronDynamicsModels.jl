# Initial-condition layouts shared by the solver and analysis scripts.

const GOLDEN_RATIO = (1 + √5) / 2

sunflower_radius(k, n, b) = k > n - b ? 1.0 : sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)

"""
    sunflower(n, α) -> Vector{Vector{Float64}}

`n` points on the unit disc in Vogel's sunflower spiral: point `k` at angle `k·2π/ϕ²` (golden-angle stride,
ϕ = (1 + √5)/2) and radius `√(k − ½)/√(n − (b + 1)/2)`, a uniform areal density. The outermost
`b = round(Int, α√n)` points sit exactly ON the boundary ρ = 1 (`α` sets how many; the production layout uses
`α = 2`, i.e. 2√n edge points), so tests like "ρ > R" flag them from the start unless excluded.

Scale by the disc radius: `Rmax * sunflower(N, 2)`. This is the electron layout of the production solvers
(thomson_scattering.jl, inverse_thomson_scattering.jl, lpwa.jl); analysis scripts that rebuild initial
conditions must use it to reproduce them bit for bit.
"""
function sunflower(n, α)
    points = Vector{Vector{Float64}}()
    angle_stride = 2π / GOLDEN_RATIO^2
    b = round(Int, α * sqrt(n))
    for k in 1:n
        r = sunflower_radius(k, n, b)
        θ = k * angle_stride
        push!(points, [r * cos(θ), r * sin(θ)])
    end
    return points
end
