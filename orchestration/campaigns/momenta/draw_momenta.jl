# Per-electron (θx, θy, δ) for EDM_MOMENTA: θ ~ N(0, θe) per axis (mrad), δ = γ/γ₀ − 1 ~ N(0, 0.2)
# truncated at |δ| ≤ 3σ (redraw). One standard-normal draw, scaled per beam, so every beam shares it.
using Random, Printf, Statistics
rng = MersenneTwister(20260929)          # positions used 20260928
N = 500
Z = zeros(N, 3)
for i in 1:N
    Z[i, 1], Z[i, 2] = randn(rng), randn(rng)
    z = randn(rng); while abs(z) > 3; z = randn(rng); end
    Z[i, 3] = z
end
for (name, θe) in (("wei_g_warm_div2.0_s20", 2.0), ("wei_lg_warm_div1.2_s20", 1.2))
    s = join((@sprintf("%.6f,%.6f,%.6f", θe * Z[i, 1], θe * Z[i, 2], 0.2 * Z[i, 3]) for i in 1:N), ";")
    write("$name.txt", s)
    @printf("%s: θ rms %.3f/%.3f mrad, δ rms %.4f, δ extrema %.3f/%.3f, mean δ %.4f\n", name,
        θe * std(Z[:, 1]; corrected = false), θe * std(Z[:, 2]; corrected = false),
        0.2 * sqrt(mean(abs2, Z[:, 3])), 0.2 * minimum(Z[:, 3]), 0.2 * maximum(Z[:, 3]), 0.2 * mean(Z[:, 3]))
end
