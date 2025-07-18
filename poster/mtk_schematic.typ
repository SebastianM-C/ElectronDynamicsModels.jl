#set page(width: 32cm, height: 20cm, margin: 0.5cm, fill: white)
#set text(size: 20pt)

// Match poster color palette
#let julia-purple = rgb("#9558b2")
#let julia-green = rgb("#389826")
#let julia-red = rgb("#cb3c33")
#let accent-blue = rgb("#4063D8")
#let light-gray = rgb("#f5f5f5")


// Main grid layout showing the main components
#grid(
  columns: (1fr, 1fr),
  rows: (1fr, 1fr),
  row-gutter: 30pt,
  column-gutter: 35pt,
  
  // Top left - External Fields
  rect(
    fill: gradient.linear(julia-green.lighten(95%), julia-green.lighten(90%), angle: 90deg),
    stroke: julia-green,
    radius: 8pt,
    inset: 20pt
  )[
    #text(size: 26pt, weight: "bold")[External Fields]
    #v(10pt)
    #text(size: 22pt)[
      PlaneWave, GaussLaser
      
      #v(8pt)
      
      Faraday tensor: $F^(mu nu) = mat(
        0, -E_x slash c, -E_y slash c, -E_z slash c;
        E_x slash c, 0, -B_z, B_y;
        E_y slash c, B_z, 0, -B_x;
        E_z slash c, -B_y, B_x, 0
      )$
    ]
  ],

  // Top right - Particle Dynamics
  rect(
    fill: gradient.linear(julia-purple.lighten(95%), julia-purple.lighten(90%), angle: 90deg),
    stroke: julia-purple,
    radius: 8pt,
    inset: 20pt
  )[
    #text(size: 26pt, weight: "bold")[ParticleDynamics]
    #v(10pt)
    #text(size: 22pt)[
      $d x^mu slash d tau = u^mu$
      
      #v(8pt)
      
      $m d u^mu slash d tau = F^mu_"total"$
      
      #v(8pt)
      
      $u_mu u^mu = -c^2$
    ]
  ],

  // Bottom left - Radiation Models
  rect(
    fill: gradient.linear(julia-red.lighten(95%), julia-red.lighten(90%), angle: 90deg),
    stroke: julia-red,
    radius: 8pt,
    inset: 20pt
  )[
    #text(size: 26pt, weight: "bold")[Radiation Models]
    #v(10pt)
    #text(size: 22pt)[
      Abraham-Lorentz, Landau-Lifshitz
      
      #v(8pt)
      
      Abraham-Lorentz: $F^mu_"rad" prop (d^2 u^mu)/(d tau^2)$
    ]
  ],
  
  // Bottom right - Composed System
  rect(
    fill: gradient.linear(accent-blue.lighten(95%), accent-blue.lighten(90%), angle: 90deg),
    stroke: accent-blue,
    radius: 8pt,
    inset: 20pt
  )[
    #text(size: 26pt, weight: "bold")[Composed System: ChargedParticle]
    #v(10pt)
    #text(size: 22pt)[
      $F^mu_"Lorentz" = q F^(mu nu) u_nu$
      
      #v(8pt)
      
      $F^mu_"total" = F^mu_"Lorentz" + F^mu_"rad"$
    ]
  ]
)

#v(10pt)

#align(center)[
  #text(size: 20pt, style: "italic")[Components are swappable via MTK connectors]
]