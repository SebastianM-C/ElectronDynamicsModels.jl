# smalln_layouts_g3p5 — few-electron image formation at γ = 3.5 (n₀ = 46.98), a₀ = 0.01, with
# explicit layouts (EDM_POSITIONS, w₀ units) instead of the sunflower: one electron, three pairs that
# isolate the two source phases (the vortex e^{imθ}, m = −2, and the signed-u radial π flip), then
# rings of 3 → 24 on the inner LG lobe (r₁ = 0.587 w₀) that show fringes turning into rings.
# Same geometry as image_a0_g3p5 / alias_ladder_g3p5. Fringe period of a pair at separation d:
# λ_n Z/d = 0.757/d w₀. Feeds the screen-image-formation report.
CAMPAIGN=smalln_layouts_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=layout
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.01
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571 EDM_SCREEN_HW=3.0
  EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576
)
CELLS=(
  "one|EDM_LAYOUT=one EDM_POSITIONS=0.587,0"   # one electron on the inner lobe: flat |E|, no fringes
  "pair_180|EDM_LAYOUT=pair_180 EDM_POSITIONS=0.587,0;-0.587,0"   # Δθ=180°: vortex phase −2·180° ≡ 0 → bright centre fringe, period 0.757/d = 0.645 w₀
  "pair_90|EDM_LAYOUT=pair_90 EDM_POSITIONS=0.587,0;0,0.587"   # Δθ=90°: vortex phase −2·90° = π → dark centre line, period 0.912 w₀
  "pair_lobes|EDM_LAYOUT=pair_lobes EDM_POSITIONS=0.587,0;-1.338,0"   # inner vs second lobe (signed u flips, π) at Δθ=180° → dark centre, period 0.393 w₀
  "ring3|EDM_LAYOUT=ring3 EDM_POSITIONS=0.587,0;-0.2935,0.5084;-0.2935,-0.5084"   # 3 on the inner-lobe circle
  "ring6|EDM_LAYOUT=ring6 EDM_POSITIONS=0.587,0;0.2935,0.5084;-0.2935,0.5084;-0.587,0;-0.2935,-0.5084;0.2935,-0.5084"   # 6 on the inner-lobe circle
  "ring12|EDM_LAYOUT=ring12 EDM_POSITIONS=0.587,0;0.5084,0.2935;0.2935,0.5084;0,0.587;-0.2935,0.5084;-0.5084,0.2935;-0.587,0;-0.5084,-0.2935;-0.2935,-0.5084;0,-0.587;0.2935,-0.5084;0.5084,-0.2935"   # 12: approaching a continuous ring
  "ring24|EDM_LAYOUT=ring24 EDM_POSITIONS=0.587,0;0.567,0.1519;0.5084,0.2935;0.4151,0.4151;0.2935,0.5084;0.1519,0.567;0,0.587;-0.1519,0.567;-0.2935,0.5084;-0.4151,0.4151;-0.5084,0.2935;-0.567,0.1519;-0.587,0;-0.567,-0.1519;-0.5084,-0.2935;-0.4151,-0.4151;-0.2935,-0.5084;-0.1519,-0.567;0,-0.587;0.1519,-0.567;0.2935,-0.5084;0.4151,-0.4151;0.5084,-0.2935;0.567,-0.1519"   # 24: a sampled continuous ring → Bessel J₂ rings
)
