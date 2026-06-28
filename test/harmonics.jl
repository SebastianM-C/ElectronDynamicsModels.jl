# Included from runtests.jl (ElectronDynamicsModels + Test already in scope).
const _EDM = ElectronDynamicsModels

@testset "harmonic_bins" begin
    # rfftfreq(100, 1.0) = 0:0.01:0.5; fundamental ω/2π = 0.1 ⇒ harmonics at freq 0.1, 0.2
    @test harmonic_bins(100, 1.0, 2π * 0.1, (1, 2)) == [11, 21]
    @test harmonic_bins(100, 1.0, 2π * 0.1, [3]) == [31]
end

@testset "harmonic_maps" begin
    N, nc, nx, ny = 8, 2, 3, 3
    cube = randn(Float64, N, nc, nx, ny)
    bins = [2, 4]
    got = harmonic_maps(cube, bins, window = nothing)
    @test size(got) == (length(bins), nc, nx, ny)
    @test eltype(got) <: Complex
    # must equal a per-component rfft sliced at `bins` (reference uses EDM's own FFTW import)
    for j in 1:nc
        Fω = _EDM.rfft(cube[:, j, :, :], 1)
        for (k, b) in enumerate(bins)
            @test got[k, j, :, :] ≈ Fω[b, :, :]
        end
    end
end

@testset "harmonic_maps — field (E,B)" begin
    N, nx, ny = 8, 3, 3
    fld = (E = randn(Float64, N, 3, nx, ny), B = randn(Float64, N, 3, nx, ny))
    bins = [2, 4]
    got = harmonic_maps(fld, bins)
    @test size(got) == (length(bins), 6, nx, ny)   # Eˣ Eʸ Eᶻ Bˣ Bʸ Bᶻ
    @test got[:, 1:3, :, :] ≈ harmonic_maps(fld.E, bins)
    @test got[:, 4:6, :, :] ≈ harmonic_maps(fld.B, bins)
end
