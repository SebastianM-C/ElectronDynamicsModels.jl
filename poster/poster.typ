// A0 Poster Template for JuliaCon
// Component Based Modeling for Relativistic Electrons

#set document(
  title: "Component Based Modeling for Relativistic Electrons",
  author: ("Sebastian Micluța-Câmpeanu",),
)

#set page(
  paper: "a0",
  margin: (left: 3cm, right: 3cm, top: 2cm, bottom: 2cm),
  fill: white,
)

#set text(size: 18pt)
#set par(justify: true, leading: 0.52em)
#set heading(numbering: none)

// Modern color palette
#let julia-purple = rgb("#9558b2")
#let julia-green = rgb("#389826")
#let julia-red = rgb("#cb3c33")
#let accent-blue = rgb("#4063D8")
#let dark-bg = rgb("#2d2d2d")
#let light-gray = rgb("#f5f5f5")

// Modern title section with gradient-like effect
#rect(fill: gradient.linear(julia-purple.darken(10%), julia-purple.lighten(10%), angle: 45deg), width: 100%, inset: 0pt)[
  #pad(x: 30pt, y: 25pt)[
    #text(size: 54pt, weight: "bold", fill: white)[
      Component Based Modeling for Relativistic Electrons
    ]

    #v(10pt)

    #text(size: 32pt, fill: white.transparentize(10%))[
      Sebastian Micluța-Câmpeanu#super[1, 2]
    ]

    #v(5pt)

    #text(size: 24pt, fill: white.transparentize(20%))[
      #super[1]JuliaHub #h(3em) #super[2]University of Bucharest
    ]
  ]
]

#v(12pt)

// Modern abstract
#rect(
  fill: light-gray,
  stroke: none,
  radius: 8pt,
  inset: 18pt
)[
  #text(size: 24pt, weight: "bold")[Abstract]

  #text(size: 18pt)[
  ElectronDynamicsModels.jl provides composable models for relativistic electron dynamics within ModelingToolkit, enabling radiation reaction and photon emission studies. Building on LaserTypes.jl, it offers AD compatibility and symbolic manipulation through modular components connected via MTK.
  ]
]

#v(10pt)

// Main content in 3 columns
#columns(3, gutter: 24pt)[

  = Introduction

  ElectronDynamicsModels.jl provides a framework for modeling relativistic electron dynamics within ModelingToolkit. Building upon our previous work in LaserTypes.jl, this new approach offers:

  #rect(fill: accent-blue.lighten(90%), stroke: none, radius: 4pt, inset: 10pt)[
    • Automatic differentiation compatibility

    • Symbolic manipulation capabilities

    • Composable model architecture

    • Swappable radiation models
  ]

  = Physical Models

  == Covariant Formulation

  #rect(
    fill: gradient.linear(julia-purple.lighten(95%), julia-purple.lighten(90%), angle: 90deg),
    stroke: julia-purple,
    radius: 6pt,
    inset: 12pt
  )[
    #text(weight: "bold")[Equations of Motion]

    #text(size: 20pt)[$ (d x^mu) / (d tau) = u^mu $]

    #text(size: 20pt)[$ m (d u^mu) / (d tau) = F^mu_"total" = F^mu_"Lorentz" + F^mu_"rad" $]

    #text(size: 18pt)[where $F^mu_"Lorentz" = q F^(mu nu) u_nu$]
  ]


  = Modular Code Structure

  #rect(
    fill: dark-bg,
    stroke: 0.5pt + gray,
    radius: 6pt,
    inset: 10pt
  )[
    #text(size: 20pt, fill: julia-green.lighten(20%))[
```julia
@component function ParticleDynamics(; name, mass = 1.0, spacetime)
    @unpack c = spacetime

    @variables begin
        t(τ), [description = "Universal time"]
        γ(τ), [description = "Lorentz factor"]
        x(τ)[1:4], [guess = [c * t, 0, 0, 0]]
        u(τ)[1:4], [guess = [c, 0, 0, 0]]
        p(τ)[1:4]
        F_total(τ)[1:4]
    end
    @parameters m = mass

    eqs = [
        t ~ x[1] / c
        p ~ m * u
        u[1] ~ γ * c
        dτ(x) ~ u
        dτ(u) ~ F_total / m
    ]

    System(eqs, τ, [t, γ, x, u, p, F_total], [m];
      name, systems = [spacetime])
end
```
    ]
  ]

  = External Fields

  == Plane Wave

  #rect(
    fill: gradient.linear(julia-red.lighten(95%), julia-red.lighten(90%), angle: 90deg),
    stroke: julia-red,
    radius: 6pt,
    inset: 10pt
  )[
    #text(weight: "bold")[Monochromatic plane wave]

    #text(size: 20pt)[$ bold(E) = E_0 hat(x) cos(bold(k) dot bold(r) - omega t) $]

    #text(size: 20pt)[$ bold(B) = (bold(k) times bold(E))/omega = (hat(k) times hat(x)) E_0/c cos(bold(k) dot bold(r) - omega t) $]

    #text(size: 18pt)[Propagating along $hat(k)$ with linear polarization in $hat(x)$ direction]
  ]

  == Gaussian Laser Pulse

  #rect(
    fill: gradient.linear(julia-green.lighten(95%), julia-green.lighten(90%), angle: 90deg),
    stroke: julia-green,
    radius: 6pt,
    inset: 10pt
  )[
    #text(weight: "bold")[Electric field at focus]

    #text(size: 20pt)[$ E(r, z, t) = E_0 (w_0)/(w(z)) exp(-(r^2)/(w(z)^2)) exp(i(k z - omega t + phi(r,z))) $]

    #text(size: 18pt)[with beam waist $w_0$, Rayleigh range $z_R = pi w_0^2 / lambda$]
  ]

  #figure(
    rect(
      image("gaussian_pulse_field.png", width: 100%),
      stroke: 0.5pt + gray,
      radius: 4pt
    ),
    caption: [3D visualization of focused Gaussian beam]
  )

  = Creating Custom Components

  #rect(
    fill: dark-bg,
    stroke: 0.5pt + gray,
    radius: 6pt,
    inset: 12pt
  )[
    #text(size: 20pt, fill: julia-green.lighten(20%))[
```julia
# Define custom field configurations
@component function CustomField(; name, spacetime)
    @named field_dynamics = EMFieldDynamics(; spacetime)
    @variables x(τ)[1:4] t(τ)
    E, B = field_dynamics.E, field_dynamics.B

    # Define your field components
    eqs = [
        E[1] ~ my_Ex_function(x, t),
        E[2] ~ my_Ey_function(x, t),
        E[3] ~ my_Ez_function(x, t),
        B[1] ~ my_Bx_function(x, t),
        B[2] ~ my_By_function(x, t),
        B[3] ~ my_Bz_function(x, t)
    ]

    System(eqs, τ; name, systems=[field_dynamics])
end

# Compose with electron
@named custom_field = CustomField(; spacetime)
@named electron = ChargedParticle(external_field = custom_field)
```
    ]
  ]

  #colbreak()

  = ModelingToolkit Architecture

  #figure(
    image("mtk_schematic.png", width: 100%),
    caption: [Component-based architecture with swappable models]
  )

  #rect(
    fill: dark-bg,
    stroke: 0.5pt + gray,
    radius: 6pt,
    inset: 10pt
  )[
    #text(size: 20pt, fill: julia-green.lighten(20%))[
```julia
# Modular force composition
# Without radiation:
F_total ~ F_lorentz

# Lorentz force from field tensor
F_lorentz[μ] ~ q * sum(Fμν[μ,ν] * u[ν] for ν in 1:4)

# With radiation reaction (optional):
if radiation_model == :abraham_lorentz
    @named radiation = AbrahamLorentzRadiation(
        charge = q,
        spacetime = spacetime,
        particle = particle
    )
    push!(systems, radiation)
    F_total ~ F_lorentz + radiation.F_rad
end
```
    ]
  ]

  = System Transformation

  #text(size: 20pt)[== Unsimplified System (24 equations)]

  #rect(
    fill: gradient.linear(accent-blue.lighten(95%), accent-blue.lighten(90%), angle: 90deg),
    stroke: accent-blue,
    radius: 6pt,
    inset: 10pt
  )[
    #text(size: 18pt, weight: "bold")[
    Differential equations (2):
    ]
    #align(left)[
      #text(size: 20pt)[
      $D("particle.x") &= "particle.u"$ \
      $D("particle.u") &= "particle.F_total" / "particle.m"$
      ]
    ]

    #v(8pt)
    #text(size: 18pt, weight: "bold")[
    Algebraic equations (22):
    ]
    #align(left)[
      #text(size: 20pt)[
      $x &= "particle.x"$ \
      $u &= "particle.u"$ \
      $"plane_wave.x" &= "particle.x"$ \
      $"plane_wave.t" &= "particle.t"$ \
      $"particle.t" &= "particle.x"_1 / c$ \
      $"particle.p" &= "particle.m" times "particle.u"$ \
      $"particle.u"_1 &= c times "particle.γ"$ \
      $E_1 &= A cos(bold(k) dot bold(x) - omega t)$ \
      $E_2 &= 0$ \
      $E_3 &= 0$ \
      $B_1 &= 0$ \
      $B_2 &= A cos(bold(k) dot bold(x) - omega t) / c$ \
      $B_3 &= 0$ \
      $lambda &= 2 pi c / omega$ \
      $F_"lorentz" &= q F^(mu nu) (g_(mu nu) u)$ \
      $"particle.F_total" &= F_"lorentz"$
      ]
    ]

    #v(8pt)
    #text(size: 18pt, weight: "bold")[
    Field tensor:
    ]
    #text(size: 20pt)[
      $F^(mu nu) = mat(
        0, -E_1/c, -E_2/c, -E_3/c;
        E_1/c, 0, -B_3, B_2;
        E_2/c, B_3, 0, -B_1;
        E_3/c, -B_2, B_1, 0
      )$
    ]

    #v(8pt)
    #text(size: 18pt, weight: "bold")[
    Stress-energy tensor:
    ]
    #text(size: 20pt)[
      $T^(mu nu) = 1/(mu_0) [F^(mu alpha) F_alpha^nu - 1/4 g^(mu nu) F_(alpha beta) F^(alpha beta)]$
    ]
  ]

  #v(10pt)
  #text(size: 20pt)[== Compiled System (8 ODEs only!)]

  #rect(
    fill: gradient.linear(julia-green.lighten(95%), julia-green.lighten(90%), angle: 90deg),
    stroke: julia-green,
    radius: 6pt,
    inset: 12pt
  )[
    #text(size: 20pt, weight: "bold")[Full equations after structural simplification:]

    #v(12pt)

    #text(size: 22pt)[
      $D(x^1) = u^1$
    ]
    #v(8pt)
    #text(size: 22pt)[
      $D(x^2) = u^2$
    ]
    #v(8pt)
    #text(size: 22pt)[
      $D(x^3) = u^3$
    ]
    #v(8pt)
    #text(size: 22pt)[
      $D(x^4) = u^4$
    ]

    #v(10pt)

    #text(size: 20pt)[
      $D(u^1) = (-g_(21) u^1 - g_(22) u^2 - g_(23) u^3 - g_(24) u^4) (A q cos(phi))/(c m)$
    ]

    #v(8pt)

    #text(size: 20pt)[
      $D(u^2) = (q A cos(phi))/m [(g_(41) u^1 + g_(42) u^2 + g_(43) u^3 + g_(44) u^4)/c$
      $ + (g_(11) u^1 + g_(12) u^2 + g_(13) u^3 + g_(14) u^4)/c]$
    ]

    #v(8pt)

    #text(size: 20pt)[
      $D(u^3) = 0$
    ]

    #v(8pt)

    #text(size: 20pt)[
      $D(u^4) = (-g_(21) u^1 - g_(22) u^2 - g_(23) u^3 - g_(24) u^4) (A q cos(phi))/(c m)$
    ]

    #v(12pt)

    #text(size: 18pt)[where $phi = -omega x^1/c + k_1 x^2 + k_2 x^3 + k_3 x^4$]

    #v(10pt)

  ]

  #v(20pt)

  #align(center)[
    #rect(
      fill: white,
      stroke: 2pt + julia-purple,
      radius: 8pt,
      inset: 16pt
    )[
      #text(size: 22pt, weight: "bold")[Get Started]

      #text(size: 18pt)[
      *GitHub:* github.com/SebastianM-C/ElectronDynamicsModels.jl
      ]
    ]
  ]

  #v(15pt)
  
  #align(center)[
    #text(size: 14pt, style: "italic", fill: gray)[
      Poster created with assistance from Claude
    ]
  ]

  #colbreak()

  = Figure-8 Motion Example

  Charged particles in plane waves can exhibit figure-8 motion #cite(<sarachik>), providing an excellent numerical accuracy test.

  #rect(
    fill: dark-bg,
    stroke: 0.5pt + gray,
    radius: 6pt,
    inset: 12pt
  )[
    #text(size: 20pt, fill: julia-green.lighten(20%))[
```julia
using ElectronDynamicsModels, ModelingToolkit
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5

# Physical parameters
k = 1; c = 1; a₀ = 1.0

# Setup plane wave and electron
@named plane_wave = PlaneWave(
    amplitude = a₀,
    frequency = c * k,
    k_vector = [0, 0, k]
)
@named electron = ChargedParticle(
    external_field = plane_wave,
    radiation_model = nothing
)
sys = mtkcompile(electron)

# Initial conditions for figure-8 orbit
v₀_z = -c / ((2/a₀)^2 + 1)
γ₀ = 1.0 / sqrt(1 - v₀_z^2)
u0 = [
    sys.x => [0.0, 0.0, 0.0, 0.0],
    sys.u => [γ₀, 0.0, 0.0, γ₀*v₀_z]
]

# Solve with different accuracies
prob = ODEProblem(sys, u0, (0.0, 500.0))
sol_tsit5 = solve(prob, Tsit5())
sol_vern9 = solve(prob, Vern9(),
    abstol=1e-9, reltol=1e-9)
```
    ]
  ]

  = Numerical Results

  == Conservation Metrics

  #rect(
    fill: accent-blue.lighten(95%),
    stroke: accent-blue,
    radius: 4pt,
    inset: 10pt
  )[
    #align(left)[
      #text(size: 20pt)[
      #text(weight: "bold")[Four-velocity norm:] $u_mu u^mu &= -c^2$ (constant) \
      #text(weight: "bold")[Lightfront momentum:] $p^(-) = p^0 - p^3$ (conserved in plane wave)
      ]
    ]

    #text(size: 20pt)[Conservation of these quantities validates numerical accuracy]
  ]

  == Solver Comparison

  #figure(
    rect(
      image("figure8.svg", width: 100%),
      stroke: 0.5pt + gray,
      radius: 4pt
    ),
    caption: [Figure-8 trajectories with conservation analysis]
  )

  #table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    stroke: none,
    fill: (x, y) => if y == 0 { julia-purple.lighten(85%) } else if calc.odd(y) { white } else { light-gray },
    table.header(
      [*Metric*], [*Tsit5 (1e-6)*], [*Vern9 (1e-6)*], [*Vern9 (1e-9)*]
    ),
    [Time steps], [761], [366], [1287],
    [Function evaluations], [4563], [5842], [20578],
    [Accepted steps], [760], [365], [1286],
    [4-velocity norm error], [~$3.65 times 10^(-4)$], [~$1.65 times 10^(-7)$], [~$7 times 10^(-12)$],
  )

  == Energy Evolution

  #figure(
    rect(
      image("energy_evolution.svg", width: 100%),
      stroke: 0.5pt + gray,
      radius: 4pt
    ),
    caption: [Energy oscillations in plane wave interaction]
  )

  #rect(
    fill: gradient.linear(julia-purple.lighten(95%), julia-purple.lighten(90%), angle: 90deg),
    stroke: julia-purple,
    radius: 6pt,
    inset: 12pt
  )[
    #text(size: 20pt)[
      In plane wave interactions, energy is #text(weight: "bold")[not conserved] due to work done by the electromagnetic field. The electron gains and loses energy periodically as it oscillates in the wave field.
    ]
  ]
]

= References

#bibliography("references.bib", style: "ieee")
