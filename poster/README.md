# ElectronDynamicsModels.jl JuliaCon Poster

## Setup

1. Install Typst: https://github.com/typst/typst
2. The poster uses the `poster-boards` template which will be automatically downloaded

## Building

### Using Julia with Typst_jll
```julia
julia build.jl
```

### Or using Typst directly
```bash
typst compile poster.typ poster.pdf
```

## Customization

### Content
- Edit `poster.typ` to modify text, equations, and layout
- The poster is divided into 3 columns
- Add your name, affiliation, and email in the header section

### Styling
- Colors are set to Julia's brand colors (purple, green, red)
- Adjust `primary-color`, `secondary-color`, etc. in the template
- Font sizes can be modified with `#set text(size: ...)`

### Adding Figures
Replace placeholder sections with your actual figures:
```typst
#figure(
  image("path/to/your/figure.png", width: 100%),
  caption: "Your caption"
)
```

### A0 Size
The poster is set to A0 size (841mm × 1189mm) which is standard for academic conferences.

## Tips
- Keep text concise for poster readability
- Use high-resolution images (300 DPI minimum)
- Test print on A4 to check readability before final printing