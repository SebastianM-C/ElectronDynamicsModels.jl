# zsweep_g3p5 — near field to far field at FIXED γ, by moving the screen (EDM_Z). γ = 3.5
# (n₀ = 46.98), a₀ = 0.01, the production 2000-electron sunflower. The disk's Fresnel number
# N_F = Rmax²n₀/(λZ) = 14 × (2×10⁵λ / Z): the same electrons, dynamics and wavelength paint the
# image of the disk at Z = 2×10⁵λ and its Fourier transform at 10⁷λ (N_F = 0.28, the rest-electron
# regime). Screens grow with the transform (∝ λZ/(n₀Rmax)); the lattice-alias radius grows ∝ Z too,
# so only the nearest cell has a halo in view. Far cells first: they also check the retarded-time
# solve at a 50× larger distance. Feeds the screen-image-formation report.
CAMPAIGN=zsweep_g3p5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=0
SWEEP_AXES=Z
BASE=(
  EDM_NX=201 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_A0=0.01
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow
  EDM_APODIZATION=none
  EDM_DIRECT_READ=1
  EDM_ELECTRON_BATCH=250
  EDM_GAMMA_EPS=2.5 EDM_SPP=256 EDM_TSPAN_TAU=4.571428571428571
  EDM_HARMONICS=46.5735,46.9788,92.8386,93.9576
)
CELLS=(
  "z1e7|EDM_Z=1.0e7 EDM_SCREEN_HW=30"
  "z4e6|EDM_Z=4.0e6 EDM_SCREEN_HW=12"
  "z2e6|EDM_Z=2.0e6 EDM_SCREEN_HW=8"
  "z1e6|EDM_Z=1.0e6 EDM_SCREEN_HW=6"
  "z5e5|EDM_Z=5.0e5 EDM_SCREEN_HW=4"
  "z2e5|EDM_Z=2.0e5 EDM_SCREEN_HW=3"
)
