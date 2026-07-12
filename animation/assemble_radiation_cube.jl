# Assemble a batched radiation-slice download (EDM_RAD_BATCH output of
# precompute_radiation.jl) into the single radiation_cube.jls NamedTuple the
# renderers expect. Usage:
#   julia --project=animation animation/assemble_radiation_cube.jl <base> [out]
# where <base> is the path prefix, i.e. <base>_manifest.jls and
# <base>_batch***.jls exist (default out: animation/radiation_cube.jls).
using Serialization
using Printf

base = ARGS[1]
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "radiation_cube.jls")
m = deserialize("$(base)_manifest.jls")
rad = Array{Float32}(undef, length(m.frame_times), m.NSLICES, m.NT, m.NT)
rad_s = similar(rad)
for bi in 1:m.n_batches
    b = deserialize(@sprintf("%s_batch%03d.jls", base, bi))
    rad[:, b.slices, :, :] .= b.rad
    rad_s[:, b.slices, :, :] .= b.rad_s
    println("batch $bi/$(m.n_batches): slices $(b.slices)")
end
serialize(out, (; rad, rad_s, m.slice_zs, m.txs, m.tys, m.frame_times))
@info @sprintf("assembled %s (%.2f GB)", out, 2 * sizeof(rad) / 1e9)
