using HypergeometricFunctions: _₁F₁, pochhammer
using SpecialFunctions: erf, erfi, dawson, besselj
using LinearAlgebra

const p = 2
const m = -2
const a₀ = 1e-5
const ω = 0.057
const c = 137.03599908330932
const w₀ = 75*2π/0.057*137.03599908330932
const qme = -1.0
const ϕ₀ = π/2
const s = 150

# A₀ = a₀*c/qme*sqrt(pochhammer(p + 1, abs(m)))/√2
# A(ρ) = A₀ * (√2*ρ/w₀)^abs(m)*_₁F₁(-p, abs(m) + 1, 2*(ρ/w₀)^2)*exp(-(ρ/w₀)^2)
a(ρ) = a₀ *sqrt(pochhammer(p + 1, abs(m)))/√2 * (√2*ρ/w₀)^abs(m)*_₁F₁(-p, abs(m) + 1, 2*(ρ/w₀)^2)*exp(-(ρ/w₀)^2)

const 𝔟 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
const ζx = -im*my_laser.polarization.ξx
const ζy = -my_laser.polarization.ξy

function 𝔍(N, ϵL, ϕ₀, θ₀, aρ₀)
    cis(N*ϵL*ϕ₀)*besselj(N, N*aρ₀*θ₀/√2)
end

function 𝔈(x_obs, x_part, N)
    Φ₀ = atan(x_part[2], x_part[1])
    X₀ = norm(x_obs - x_part)
    Z₀ = x_obs[3]
    ρ₀ = norm(x_part)
    aρ₀ = a(ρ₀)

    n₀ = (x_obs - x_part)*inv(X₀)

    ν_set = [n₀ × (n₀ × eᵢ) for eᵢ in 𝔟]
    ν_plus = ζx*ν_set[1] + ζy*ν_set[2] 
    ν_minus = ζx*ν_set[1] - ζy*ν_set[2] 

    α = 1 + (1 - n₀[3])*aρ₀^2/4

    pf = -im*ω*N*√(2π)*aρ₀/(8*π*α*c*X₀)

    phase_term = cis(ω*N*(X₀-Z₀)+N*m*Φ₀)
    # nota bene: there's a parnthesis here that'd be zero
    # for circular polarization
    ϕ₀_obs = atan(n₀[2], n₀[1])
    θ₀ = acos(n₀[3])
    νz_term = (aρ₀/2)*ν_set[3]*𝔍(N, ζy/ζx, ϕ₀_obs, θ₀, aρ₀) 
    νp_term = ν_plus*𝔍(N-1, ζy/ζx, ϕ₀_obs, θ₀, aρ₀) 
    νm_term = ν_minus*𝔍(N+1, ζy/ζx, ϕ₀_obs, θ₀, aρ₀)  

    ν_term = νz_term + νp_term + νm_term
    return pf*phase_term*ν_term
end

𝔈([w₀, w₀, 2e5λ], [8λ, 16λ, 0.], 2)

heatmap(-25w₀:w₀/100:25w₀, -25w₀:w₀/100:25w₀, (x, y) -> real(𝔈([x, y, 2e5λ], [10λ, 10λ, 0.], 2)[2]), colormap = :seismic)
heatmap(range(-25w₀, 25w₀, 50), range(-25w₀, 25w₀, 50), (x, y) -> real(𝔈([x, y, 2e5λ], [w₀, 10λ, 0.], 1)[2]), colormap = :seismic)


# using HCubature

# function 𝔼(x_obs, N)
#     ∫, ϵ = hcubature(v -> 𝔈(x_obs, [v; 0.], N), [-3.25w₀, -3.25w₀], [3.25w₀, 3.25w₀])
#     return ∫
# end
