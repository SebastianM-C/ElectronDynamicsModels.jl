# Shared Makie theme for the EDM plotting scripts: LaTeX (Computer Modern) fonts
# via theme_latexfonts(), so titles/labels match the dashboard's LaTeX descriptions.
# Include after `using CairoMakie`, then use L"..." strings for math labels.
using CairoMakie
set_theme!(theme_latexfonts())
