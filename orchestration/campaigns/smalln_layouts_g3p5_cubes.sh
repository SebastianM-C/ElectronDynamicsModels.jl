# smalln_layouts_g3p5_cubes (branch campaigns/smalln-cubes) — supplement to smalln_layouts_g3p5 (same CAMPAIGN dir, same BASE). The *_cube cells
# rerun one / pair_180 / ring24 ONLY to keep the cube (E_x(t) snapshots + per-pixel power spectra); their own
# layout labels keep the layout axis one run per label. ring4/5/16 on the inner lobe (r₁ = 0.587 w₀) complete
# the 3 → 24 ring sequence.
CAMPAIGN=smalln_layouts_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
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
  "one_cube|EDM_LAYOUT=one_cube EDM_POSITIONS=0.587,0"
  "pair_180_cube|EDM_LAYOUT=pair_180_cube EDM_POSITIONS=0.587,0;-0.587,0"
  "ring24_cube|EDM_LAYOUT=ring24_cube EDM_POSITIONS=0.587,0;0.567,0.1519;0.5084,0.2935;0.4151,0.4151;0.2935,0.5084;0.1519,0.567;0,0.587;-0.1519,0.567;-0.2935,0.5084;-0.4151,0.4151;-0.5084,0.2935;-0.567,0.1519;-0.587,0;-0.567,-0.1519;-0.5084,-0.2935;-0.4151,-0.4151;-0.2935,-0.5084;-0.1519,-0.567;0,-0.587;0.1519,-0.567;0.2935,-0.5084;0.4151,-0.4151;0.5084,-0.2935;0.567,-0.1519"
  "ring4|EDM_LAYOUT=ring4 EDM_POSITIONS=0.587,0;0,0.587;-0.587,0;0,-0.587"   # 4 on the inner-lobe circle
  "ring5|EDM_LAYOUT=ring5 EDM_POSITIONS=0.587,0;0.1814,0.5583;-0.4749,0.345;-0.4749,-0.345;0.1814,-0.5583"   # 5: odd count, no antipodal pairs
  "ring16|EDM_LAYOUT=ring16 EDM_POSITIONS=0.587,0;0.5423,0.2246;0.4151,0.4151;0.2246,0.5423;0,0.587;-0.2246,0.5423;-0.4151,0.4151;-0.5423,0.2246;-0.587,0;-0.5423,-0.2246;-0.4151,-0.4151;-0.2246,-0.5423;0,-0.587;0.2246,-0.5423;0.4151,-0.4151;0.5423,-0.2246"   # 16
)
