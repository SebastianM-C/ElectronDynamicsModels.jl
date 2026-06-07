using LinearAlgebra
using HypergeometricFunctions: pochhammer, _₁F₁
using SpecialFunctions
a₀ = 0.1
c = 137.035999177
qme = -1.0
p = 2
m = -2
ω = 0.057
ϕ₀ = 0.0
λ = c * 2π / ω
w₀ = 75λ
s = 150

A₀ = a₀ * c / qme * sqrt(pochhammer(p + 1, abs(m))) / √2
#
A(ρ) = A₀ * (√2 * ρ / w₀)^abs(m) * _₁F₁(-p, abs(m) + 1, 2 * (ρ / w₀)^2) * exp(-(ρ / w₀)^2)

function trajectory(τ, ℜ₀)
    x₀, y₀, z₀ = ℜ₀
    ρ₀ = norm(ℜ₀)
    φ = m * atan(y₀, x₀) + ϕ₀

    k = ω / c
    u⁰ = c

    χ = k * u⁰ * τ

    Δx = inv(k) * (A(ρ₀) * qme / c) * s * exp(-(χ / s)^2) * real(im * cis(φ + χ) * dawson(s / 2 + im * χ / s))
    ẋ = -u⁰ * (A(ρ₀) * qme / c) * exp(-(χ / s)^2) * cos(φ + χ)
    Δy = inv(k) * (A(ρ₀) * qme / c) * s * exp(-(χ / s)^2) * real(cis(φ + χ) * dawson(s / 2 + im * χ / s))
    ẏ = -u⁰ * (A(ρ₀) * qme / c) * exp(-(χ / s)^2) * sin(φ + χ)
    Δz = inv(2k) * (A(ρ₀) * qme / c)^2 * s / 2 * sqrt(π / 2) * (1 + erf(sqrt(2) * χ / s))
    ż = u⁰ / 2 * (A(ρ₀) * qme / c)^2 * exp(-2 * (χ / s)^2)

    # (u⁰)² − (uˣ² + uʸ² + uᶻ²) = c²
    # u⁰ = √(c² + |u|²)
    # u⁰ − uᶻ = c   for plane wave
    cγ = c + ż #c^2 + hypot(ẋ, ẏ, ż)
    x⁰ = c * τ + Δz

    return [x⁰, x₀ + Δx, y₀ + Δy, z₀ + Δz, cγ, ẋ, ẏ, ż]
end
