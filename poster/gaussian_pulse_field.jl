using CairoMakie
using Colors

# Create a 3D visualization of a Gaussian laser pulse
fig = Figure(resolution = (800, 600), backgroundcolor = :white)

ax = Axis3(fig[1, 1],
    title = "Gaussian Laser Pulse Electric Field",
    titlesize = 24,
    xlabel = "x/w₀",
    ylabel = "y/w₀", 
    zlabel = "E/E₀",
    xlabelsize = 20,
    ylabelsize = 20,
    zlabelsize = 20,
    azimuth = 1.3π,
    elevation = 0.3π
)

# Parameters for Gaussian pulse (normalized units)
w₀ = 1.0   # beam waist
z = 0.0    # at focus
λ = 0.2    # wavelength relative to beam waist
k = 2π / λ
z_R = π * w₀^2 / λ  # Rayleigh range

# Grid in units of beam waist
x = range(-3, 3, length=100)
y = range(-3, 3, length=100)

# Gaussian beam electric field
function gaussian_beam(x, y, z)
    r² = x^2 + y^2
    w_z = w₀ * sqrt(1 + (z/z_R)^2)
    
    # Amplitude envelope
    amplitude = (w₀/w_z) * exp(-r²/w_z^2)
    
    # Phase including Gouy phase
    ψ_G = atan(z/z_R)
    phase = k*z - ψ_G + k*r²*z/(2*(z^2+z_R^2))
    
    return amplitude * cos(phase)
end

# Calculate field
E_field = [gaussian_beam(xi*w₀, yi*w₀, z) for xi in x, yi in y]

# Create surface plot
surface!(ax, x, y, E_field,
    colormap = :RdBu,
    colorrange = (-1, 1),
    shading = true,
    transparency = false
)

# Add contour lines at the bottom
contour!(ax, x, y, E_field,
    levels = 10,
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
save("gaussian_pulse_field.svg", fig)
save("gaussian_pulse_field.png", fig, px_per_unit = 2)

println("Gaussian laser field visualization saved")