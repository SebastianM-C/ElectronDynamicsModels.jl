# campaigns/gamma_equiv.sh — Doppler-equivalent Thomson: run the laser at the wavelength a
# counter-propagating electron of γ = {2, 5, 10} would see (ω′ = (γ+√(γ²−1))·ω₀), at a0 = 1e-2.
# EDM_OMEGA_SCALE pins the transverse/detector geometry to the lab λ₀ layout (transverse lengths
# are boost-invariant) and keeps the pulse cycle count.
# Screen zoom (v2, corrected): the pattern does NOT scale ∝ λ′ across the ladder — with geometry
# pinned, the source's Fraunhofer distance a²/λ′ grows as 1/λ′ (1.1·Z at γ=2, 5.9·Z at γ=10), so
# the screen slides into the Fresnel zone and the pattern shrinks slower than 1/s (measured ×13.6
# at γ=10, not ×19.95). g2/g5 keep the full ±25 w₀ production frame (screen-identical to base;
# corner-path spread 42/111 T′ < the 191 T′ pulse half-span, so the sampling window keeps the
# peak). Only g10 needs a zoom — at ±25 w₀ its 224 T′ spread would open the window after the
# peak passed the central pixels; ±8 w₀ (spread 36 T′) is safe and frames ~6 Fresnel zones.
# "base" = same cell at λ₀ (scale 1): the pure-wavelength-change control.
# Sizing: 201²/N2k/Ns6k ≈ 21 min/cell on the W7900 (scaled from lowa0_maps 400²/N6k = 12.9 ks).
CAMPAIGN=gamma_equiv
SCRIPT=scripts/thomson_scattering.jl
BASE=(
  EDM_NX=201 EDM_NSAMPLES=6000 EDM_SPP=16 EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-12
  EDM_A0=1e-2
  EDM_INTERP_SAVEAT=16                 # uniform trajectory output (the 2ω-floor fix)
  EDM_INITIAL_PHASE=-1.5707963267948966   # φ0 = -π/2: corpus continuity (arbitrary since PR #62)
)
CELLS=(
  "g2|EDM_OMEGA_SCALE=3.7320508075688776"
  "g5|EDM_OMEGA_SCALE=9.898979485566356"
  "g10|EDM_OMEGA_SCALE=19.949874371066196 EDM_SCREEN_HALFW=8"
  "base|"
)
