# campaigns/inverse_a0_ladder_g5.sh — the inverse-Thomson a₀ ladder at γ = 5 (γ_eps 4, n₀ = 4γ²-line = 97.99):
# a₀ = 0.01 / 0.1 / 0.3 / 1 / 2 (+ 5, conditional), N = 16000, the mgpu STRONG framing (±3 w₀ at 401², narrow
# window, tspan 3.2 τ) at SPP 1024 (Nyquist 512 ω₁ ⊇ 3n₀). Newton, cubes kept (any line re-extracts CPU-only).
#
# Harmonics per cell: the linear-limit anchors n₀, 2n₀, 3n₀; the redshifted line n_a = n₀/(1 + a₀²) (circular
# polarization, the |a|² = a₀² convention) and its 2nd/3rd harmonics; and the chirp-band centre n₀/(1 + a₀²/2) —
# the envelope sweeps the local line from n₀ (wings) to n_a (peak), so the band n_a…n₀ is the chirp. Anchors
# within one rfft bin (SPP/Ns ≈ 0.31 ω₁) of another are omitted. The full band is in the powspec and the cube.
#
# Window: the narrow window's burst term is a₀-free, but the longitudinal slowdown stretches the arrival span by
# (1 + a(t)²) locally ⇒ by f = 1 + a₀²·√(π/2)/(2√ln100) ≈ 1 + 0.292 a₀² over the 1 %-field span. The built-in
# ×1.4 margin covers f ≤ 1.4 (a₀ ≲ 1.1); above that each side gets (1.25 f − 1.4)·B₀/2 extra (B₀ = 1.05 λ at
# γ = 5): lead = tail = 0.75 (a₀ 1), 1.25 (a₀ 2), 5.25 λ (a₀ 5). Derived, not yet measured: the solver's window-
# coverage check (host, before any GPU time) and the window-budget chip confirm or refute it per cell.
#
# Sizes (48 B·Ns·401²): Ns 3332 → 24.0 GiB (a₀ ≤ 0.3), 3844 → 27.6 (a₀ 1), 4868 → 35.0 (a₀ 2, NVL/PCIe only),
# 13060 → 93.9 GiB (a₀ 5: over every local card's guard — and electron sharding does NOT split the cube, each
# device holds a full buffer set — so a₀ 5 needs one ≥ 105 GiB card: a 1×B200 VM (162 GiB guard; host 103 GiB
# solve, 141 GiB reduce ⇒ fits 170 GB RAM without overlap)).
# EDM_ELECTRON_BATCH=4000 keeps the trajectory splines at ~19 GiB of host RAM instead of ~76 (exact reduction).
CAMPAIGN=inverse_a0_ladder_g5
SCRIPT=scripts/inverse_thomson_scattering.jl
KEEP_CUBE=1
REDUCE_HOOK='export EDM_DIRECT_READ=1; _reduce_cell "$uuid"'   # overlap reducer reads ITS env, not BASE: O_DIRECT keeps the reduce at ~1.5× cube
SWEEP_AXES=a0
BASE=(
  EDM_NX=401 EDM_FIELD_MODE=total
  EDM_N=16000 EDM_ELECTRON_BATCH=4000 EDM_NSUBSTEPS=1 EDM_RELTOL=1e-13
  EDM_INTERP_SAVEAT=16
  EDM_INITIAL_PHASE=-1.5707963267948966
  EDM_WINDOW=narrow EDM_APODIZATION=none EDM_DIRECT_READ=1
  EDM_ACCUM_ALG=newton EDM_NEWTON_ITERS=2
  EDM_EMISSION_TIME=1 EDM_GAMMA_TRACE_OVERSAMPLE=4
  EDM_GAMMA_EPS=4 EDM_SPP=1024 EDM_TSPAN_TAU=3.2 EDM_SCREEN_HW=3.0
)
CELLS=(
  "a1em2|EDM_A0=0.01 EDM_HARMONICS=97.9898,195.9796,293.9694"
  "a1em1|EDM_A0=0.1 EDM_HARMONICS=97.0196,97.5023,97.9898,194.0392,195.9796,291.0588,293.9694"
  "a3em1|EDM_A0=0.3 EDM_HARMONICS=89.8989,93.7701,97.9898,179.7978,195.9796,269.6967,293.9694"
  "a1|EDM_A0=1 EDM_WINDOW_LEAD=0.75 EDM_WINDOW_TAIL=0.75 EDM_HARMONICS=48.9949,65.3265,97.9898,146.9847,195.9796,293.9694"
  "a2|EDM_A0=2 EDM_WINDOW_LEAD=1.25 EDM_WINDOW_TAIL=1.25 EDM_HARMONICS=19.598,32.6633,39.1959,58.7939,97.9898,195.9796,293.9694"
  # a₀ = 5 at SPP 1024 (Sebastian, 2026-09-28): 93.9 GiB ⇒ one ≥ 105 GiB card (B200); lane wrapper inverse_a0_ladder/a5.sh
  "a5|EDM_A0=5 EDM_WINDOW_LEAD=5.25 EDM_WINDOW_TAIL=5.25 EDM_HARMONICS=3.7688,7.2585,7.5377,11.3065,97.9898,195.9796,293.9694"
)
