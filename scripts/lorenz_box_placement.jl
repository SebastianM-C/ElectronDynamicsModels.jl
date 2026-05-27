# Overlay the Lorenz-gauge verification patch on the 2ω₁ Aʸ harmonic heatmap, so
# the box placement can be inspected before running the (expensive) verification.
# The box is a small square in the upper-left quadrant, centred on the local Aʸ
# peak near the quadrant middle.  Box geometry here MUST match scripts/verify_lorenz_gauge.jl.
#
#   julia --project=scripts scripts/lorenz_box_placement.jl

using Serialization
using CairoMakie
using Printf

const c = 137.03599908330932
const ω = 0.057
const λ = 2π * c / ω
const w₀ = 75λ

# ── Box geometry (shared with verify_lorenz_gauge.jl) ──
const CX, CY = 126, 299     # centre pixel on the 400×400 production grid
const NB = 10               # half-width in pixels → (2NB+1)=21-pixel square

const cachefile = "A_rk4_400_N10000_Ns8000_spp16_hslices.jls"   # μ=3 (Aʸ) slices
cache = deserialize(cachefile)
xg = collect(cache.x_grid)
yg = collect(cache.y_grid)

field = real.(cache.fields[2, :, :])        # 2ω₁ Aʸ
cr = maximum(abs, field)

# Box extent in physical coordinates (interior eval pixels CX-NB..CX+NB).
ix_lo, ix_hi = CX - NB, CX + NB
iy_lo, iy_hi = CY - NB, CY + NB
xlo, xhi = xg[ix_lo], xg[ix_hi]
ylo, yhi = yg[iy_lo], yg[iy_hi]
box_x = [xlo, xhi, xhi, xlo, xlo]
box_y = [ylo, ylo, yhi, yhi, ylo]

@printf("box centre  pixel (%d,%d)  →  x=%.2f w₀  y=%.2f w₀\n", CX, CY, xg[CX] / w₀, yg[CY] / w₀)
@printf("box extent  x∈[%.2f, %.2f] w₀   y∈[%.2f, %.2f] w₀   (%d×%d pixels, %.0fλ)\n",
    xlo / w₀, xhi / w₀, ylo / w₀, yhi / w₀, 2NB + 1, 2NB + 1, (xhi - xlo) / λ)

fig = Figure()
gl = fig[1, 1] = GridLayout()
ax = Axis(gl[1, 1], width = 460, height = 460, xlabel = "x", ylabel = "y",
    title = "Aʸ  2ω₁  with Lorenz-gauge verification box")
hm = heatmap!(ax, xg, yg, field, colorrange = (-cr, cr), colormap = :seismic)
lines!(ax, box_x, box_y, color = :black, linewidth = 2)
scatter!(ax, [xg[CX]], [yg[CY]], color = :black, marker = :xcross, markersize = 10)
Colorbar(gl[1, 2], hm, width = 12, height = 460)
resize_to_layout!(fig)

out = "lorenz_box_placement.png"
save(out, fig)
println("saved → ", out)
