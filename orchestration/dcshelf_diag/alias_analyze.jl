# alias_spp_diag analysis: per a₀, h1–h4 map deviation of SPP 16/32 from SPP 64 (E, B) + power near Nyquist.
#   julia +1.12 --project=scripts orchestration/dcshelf_diag/alias_analyze.jl <campaign_dir>
using Serialization, TOML, Printf, LinearAlgebra

relL2(a, b) = norm(a .- b) / norm(b)

camp = ARGS[1]
runs = map(filter(f -> startswith(f, "run_") && endswith(f, ".toml"), readdir(camp))) do f
    m = TOML.parsefile(joinpath(camp, f)); id = m["provenance"]["run_id"]
    (; a0 = m["laser"]["a0"], spp = m["config"]["samples_per_period"], id, Ns = m["config"]["N_samples"],
       h = deserialize(joinpath(camp, "hmaps_$id.jls")), ps = deserialize(joinpath(camp, "powspec_$id.jls")))
end
for a0 in sort(unique(r.a0 for r in runs))
    rs = sort(filter(r -> r.a0 == a0, runs); by = r -> r.spp)
    ref = last(rs)
    @printf "\na0 = %g  (reference SPP %d)\n" a0 ref.spp
    for r in rs
        fr, ps = r.ps.freqs, vec(sum(r.ps.ps; dims = 2))          # summed over E,B components
        top = fr .>= 0.9 * maximum(fr)
        @printf "  SPP %3d: power in top 10%% of band = %.2e of total" r.spp sum(ps[top]) / sum(ps)
        if r !== ref
            for (i, n) in enumerate(r.h.harmonics)
                j = findfirst(==(n), ref.h.harmonics)
                a, b = r.h.fields_h[i, :, :, :] ./ r.Ns, ref.h.fields_h[j, :, :, :] ./ ref.Ns   # unnormalized DFT ∝ Ns
                @printf "  | h%d ΔE=%.1e ΔB=%.1e" n relL2(a[1:3, :, :], b[1:3, :, :]) relL2(a[4:6, :, :], b[4:6, :, :])
            end
        end
        println()
    end
end
