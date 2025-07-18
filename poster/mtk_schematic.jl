using CairoMakie
using Colors

# Create MTK architecture schematic with equations
fig = Figure(resolution = (900, 700), backgroundcolor = :white)

ax = Axis(fig[1, 1], 
    title = "ModelingToolkit Component Architecture",
    titlesize = 26,
    aspect = DataAspect(),
    limits = (0, 12, 0, 9)
)
hidedecorations!(ax)
hidespines!(ax)

# Colors
julia_purple = RGB(149/255, 88/255, 178/255)
julia_green = RGB(56/255, 152/255, 38/255)
julia_red = RGB(203/255, 60/255, 51/255)
accent_blue = RGB(64/255, 99/255, 216/255)

# Component boxes
# External fields with equations
poly!(ax, Point2f[(0.5, 6), (4, 6), (4, 8), (0.5, 8)], color = julia_green, strokecolor = :black, strokewidth = 2)
text!(ax, 2.25, 7.6, text = "External Fields", fontsize = 20, align = (:center, :center), font = "bold")
text!(ax, 2.25, 7.2, text = "PlaneWave, GaussLaser", fontsize = 16, align = (:center, :center))
text!(ax, 2.25, 6.7, text = "F^{μν} = ∂^μA^ν - ∂^νA^μ", fontsize = 14, align = (:center, :center), font = "italic")
text!(ax, 2.25, 6.3, text = "E_i, B_i components", fontsize = 13, align = (:center, :center))

# Particle dynamics with equations
poly!(ax, Point2f[(0.5, 3.5), (4, 3.5), (4, 5.5), (0.5, 5.5)], color = julia_purple, strokecolor = :black, strokewidth = 2)
text!(ax, 2.25, 5.1, text = "ParticleDynamics", fontsize = 20, align = (:center, :center), font = "bold")
text!(ax, 2.25, 4.7, text = "dx^μ/dτ = u^μ", fontsize = 14, align = (:center, :center), font = "italic")
text!(ax, 2.25, 4.3, text = "m du^μ/dτ = F^μ_{total}", fontsize = 14, align = (:center, :center), font = "italic")
text!(ax, 2.25, 3.9, text = "u_μu^μ = -c²", fontsize = 13, align = (:center, :center), font = "italic")

# Radiation models with equations
poly!(ax, Point2f[(0.5, 1), (4, 1), (4, 3), (0.5, 3)], color = julia_red, strokecolor = :black, strokewidth = 2)
text!(ax, 2.25, 2.6, text = "Radiation Models", fontsize = 20, align = (:center, :center), font = "bold")
text!(ax, 2.25, 2.2, text = "Landau-Lifshitz", fontsize = 16, align = (:center, :center))
text!(ax, 2.25, 1.8, text = "F^μ_{rad} ∝ F^{μν}F_{νλ}u^λ", fontsize = 14, align = (:center, :center), font = "italic")
text!(ax, 2.25, 1.4, text = "- ¼F_{αβ}F^{αβ}u^μ", fontsize = 13, align = (:center, :center), font = "italic")

# Central MTK system with Lorentz force
poly!(ax, Point2f[(5.5, 3.5), (11, 3.5), (11, 6), (5.5, 6)], color = accent_blue, strokecolor = :black, strokewidth = 3)
text!(ax, 8.25, 5.4, text = "Composed System", fontsize = 22, align = (:center, :center), font = "bold")
text!(ax, 8.25, 4.9, text = "ChargedParticle", fontsize = 18, align = (:center, :center))
text!(ax, 8.25, 4.4, text = "F^μ_{Lorentz} = qF^{μν}u_ν", fontsize = 15, align = (:center, :center), font = "italic")
text!(ax, 8.25, 3.9, text = "F^μ_{total} = F^μ_{Lorentz} + F^μ_{rad}", fontsize = 14, align = (:center, :center), font = "italic")

# Arrows with labels
# Fields to System
lines!(ax, [4, 5.5], [7, 5.2], color = :black, linewidth = 2)
scatter!(ax, [5.5], [5.2], color = :black, markersize = 15, marker = '▶')
text!(ax, 4.75, 6.3, text = "F^{μν}", fontsize = 14, align = (:center, :center), rotation = -0.5)

# Particle to System
lines!(ax, [4, 5.5], [4.5, 4.7], color = :black, linewidth = 2)
scatter!(ax, [5.5], [4.7], color = :black, markersize = 15, marker = '▶')
text!(ax, 4.75, 4.6, text = "x^μ, u^μ", fontsize = 14, align = (:center, :center))

# Radiation to System
lines!(ax, [4, 5.5], [2, 4.2], color = :black, linewidth = 2)
scatter!(ax, [5.5], [4.2], color = :black, markersize = 15, marker = '▶')
text!(ax, 4.75, 3.1, text = "F^μ_{rad}", fontsize = 14, align = (:center, :center), rotation = 0.5)

# Add "swappable" annotation
text!(ax, 8.25, 0.5, text = "Components are swappable via MTK connectors", 
      fontsize = 16, align = (:center, :center), font = "italic")

save("mtk_schematic.svg", fig)
save("mtk_schematic.png", fig, px_per_unit = 2)

println("MTK schematic with equations saved")