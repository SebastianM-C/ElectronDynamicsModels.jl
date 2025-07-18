using CairoMakie
using Colors

# Create a 3D visualization of a Gaussian pulse
fig = Figure(resolution = (800, 600), backgroundcolor = :white)

ax = Axis3(fig[1, 1],
    title = "Gaussian Laser Pulse Electric Field",
    titlesize = 24,
    xlabel = "x",
    ylabel = "y", 
    zlabel = "E",
    xlabelsize = 20,
    ylabelsize = 20,
    zlabelsize = 20,
    azimuth = 1.3π,
    elevation = 0.3π
)

# Parameters for Gaussian pulse
λ = 1.0  # wavelength
w0 = 3.0  # beam waist
τ = 10.0  # pulse duration
t = 0.0   # time snapshot

# Grid
x = range(-10, 10, length=100)
y = range(-10, 10, length=100)
z = 0.0  # propagation direction

# Gaussian envelope in transverse direction and time
function gaussian_pulse(x, y, z, t)
    r² = x^2 + y^2
    envelope = exp(-r² / w0^2) * exp(-(z - t)^2 / τ^2)
    phase = 2π * (z - t) / λ
    return envelope * cos(phase)
end

# Calculate field
E = [gaussian_pulse(xi, yi, z, t) for xi in x, yi in y]

# Create surface plot
surface!(ax, x, y, E,
    colormap = :RdBu,
    colorrange = (-1, 1),
    shading = true,
    transparency = false
)

# Add contour lines at the bottom
contour!(ax, x, y, E,
    levels = 15,
    linewidth = 1,
    colormap = :RdBu,
    colorrange = (-1, 1),
    transformation = (:xy, -1.2)
)

# Style
ax.xgridvisible = false
ax.ygridvisible = false
ax.zgridvisible = true

# Save
save("gaussian_pulse_3d.svg", fig)
save("gaussian_pulse_3d.png", fig, px_per_unit = 2)

println("3D Gaussian pulse visualization saved")