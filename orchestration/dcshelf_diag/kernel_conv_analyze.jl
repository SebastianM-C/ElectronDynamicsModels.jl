# kernel_conv_diag analysis: h1/h2 maps of RK4 (n_substeps 1, 4) against the Newton reference, per a₀, E and B.
#   julia +1.12 --project=scripts orchestration/dcshelf_diag/kernel_conv_analyze.jl <campaign_dir>
using Serialization, Printf, LinearAlgebra

relL2(a, b) = norm(a .- b) / norm(b)

camp = ARGS[1]
rows = split.(readlines(joinpath(camp, "cells.tsv"))[2:end], '\t')
uuid = Dict(r[1] => r[2] for r in rows)                          # last row per label wins
hm(label) = deserialize(joinpath(camp, "hmaps_$(uuid[label]).jls"))
for a in ("a1", "a10")
    ref, ns1, ns4 = hm("$(a)_newton"), hm("$(a)_rk4ns1"), hm("$(a)_rk4ns4")
    @printf "\n%s  (windows: %s / %s / %s)\n" a ref.window ns1.window ns4.window
    for (i, n) in enumerate(ref.harmonics), (fld, c) in (("E", 1:3), ("B", 4:6))
        n > 2 && continue
        r = ref.fields_h[i, c, :, :]
        @printf "  h%d %s  rk4 ns1 vs Newton %.2e   ns4 vs Newton %.2e   ns1 vs ns4 %.2e\n" n fld relL2(ns1.fields_h[i, c, :, :], r) relL2(ns4.fields_h[i, c, :, :], r) relL2(ns1.fields_h[i, c, :, :], ns4.fields_h[i, c, :, :])
    end
end
