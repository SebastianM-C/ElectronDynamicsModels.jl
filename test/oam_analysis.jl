using ElectronDynamicsModels
using Test

wrapπ(x) = mod(x + π, 2π) - π   # wrap to (-π, π]

@testset "ring_pixels" begin
    g = range(-10.0, 10.0; length = 101)          # centred grid (step 0.2)
    idxs, az = ring_pixels(g, g, 5.0; tol = 0.25)
    @test !isempty(idxs)
    @test all(abs(hypot(g[ci[1]], g[ci[2]]) - 5.0) < 0.25 for ci in idxs)  # within the annulus
    @test issorted(az)                                                      # sorted by azimuth
    @test all(-π .≤ az .≤ π)
    @test length(az) == length(idxs)
end

@testset "phase_winding_fit" begin
    az = collect(range(-π, π; length = 400))      # ascending azimuths (as ring_pixels returns)

    # exact line phase = wrap(ℓ·az + b): unwrap must recover slope = ℓ, intercept ≡ b (mod 2π)
    ℓ, b = 2.0, 0.5
    fit = phase_winding_fit(az, wrapπ.(ℓ .* az .+ b))
    @test isapprox(fit.slope, ℓ; atol = 1e-6)
    @test isapprox(wrapπ(fit.intercept - b), 0.0; atol = 1e-6)

    # clean vortices e^{iℓφ} (integer ℓ) — slope recovers the winding number, both signs.
    # The negative cases MUST wrap (|ℓ| ≥ 2 over az ∈ [-π,π]) to guard the unwrap *direction*:
    # an abs()-based jump test silently returns +2.3 for ℓ = -3.
    for ℓ in (3, -2, -3, 5, -5)
        fit = phase_winding_fit(az, angle.(cis.(ℓ .* az)))
        @test isapprox(fit.slope, float(ℓ); atol = 1e-6)
    end

    @test_throws ArgumentError phase_winding_fit([0.0, 1.0], [0.0])  # length mismatch
end
