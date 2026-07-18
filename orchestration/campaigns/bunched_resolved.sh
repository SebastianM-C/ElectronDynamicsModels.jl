# campaigns/bunched_resolved.sh — the coherence-deficit discriminators on a spot-resolved screen.
#
# Production inverse_bunched measured ≈0.1 coherence vs the 0.65 W7900 smoke ceiling — but its
# ±5 w₀ / 400 px screen point-sampled the coherent spot at its first Airy null (even grid, pixel
# = 1.8 δr; speckle report §7.2). This campaign re-measures on a screen sized by §7.2's recipe:
#   EDM_SCREEN_HW=0.4 (±30λ — contains the r_F = √(λₙZ) ≈ 22.4λ moat), EDM_NX=401 (odd ⇒ a
#   grid node exactly on-axis), pixel = 0.15λ = δr/6.9.
# Discriminator ladder (each arm isolates one deficit term; base cells share the sunflower disk
# of their group, so bunched_diff_products.jl pairs within (a0, kernel) groups):
#   1-2 base/lm2         rk4, a0=1     — the RESOLUTION term (vs production lm2)
#   3-4 base_nt/lm2_nt   newton it2    — the KERNEL term (γ=10 regime guidance)
#   5-6 base_a03/lm2_a03 a0=0.3       — the a0 term (smoke ceiling at production N)
#   7   lm2_nb199        n_b=199       — Fresnel-moat falsifier: moat ∝ √(Z/n_b) ⇒ 31.7λ,
#                                        vs grain-scaling ⇒ 51.5λ (§7.3; exploratory SNR)
#   8   l0               winding |2|   — the vortex ring (~6 px radius): first resolvable OAM
#                                        image at production N
# Cubes are ~10 GB at the collapsed window — kept + drained (they feed band/moat post-analysis).
CAMPAIGN=bunched_resolved
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
BASE=(
  EDM_A0=1 EDM_GAMMA=10 EDM_WINDOW=narrow EDM_FIELD_MODE=total
  EDM_N=2000 EDM_NSUBSTEPS=1 EDM_INTERP_SAVEAT=16
  EDM_TSPAN_TAU=1.6 EDM_SPP=2048 EDM_SCREEN_HW=0.4 EDM_NX=401
  EDM_WINDOW_LEAD=0.3 EDM_WINDOW_TAIL=0.3 EDM_HARMONICS=199,299,398,597
)
CELLS=(
  "base|"
  "lm2|EDM_BUNCH_NB=398 EDM_BUNCH_L=-2"
  "base_nt|EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2"
  "lm2_nt|EDM_BUNCH_NB=398 EDM_BUNCH_L=-2 EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2"
  "base_a03|EDM_A0=0.3"
  "lm2_a03|EDM_A0=0.3 EDM_BUNCH_NB=398 EDM_BUNCH_L=-2"
  "lm2_nb199|EDM_BUNCH_NB=199 EDM_BUNCH_L=-2"
  "l0|EDM_BUNCH_NB=398"
)
