# S1 + S2: the wall model and the separation-axis decomposition

*Written: 2026-08-13 16:05 UTC by claude:model-zoo-cartography. Analyses on
committed census JSONs only (no weights). These register predictions for
Phase B Wave 1 (train/phase_b.py, launching 2026-08-14 02:00 UTC).*

## S2 — the expressivity wall is sharp, and the registered prediction

Logistic fit to convergence fraction vs C (20 seeds/point, sep 3.0, 20k steps):

- **C₅₀ = 38.2, logistic width s = 2.0** — the transition spans barely ±4
  classes. Below the wall the mean gap-to-Bayes grows almost exactly linearly
  (gap ∝ C^0.97, r² = 0.94); at the wall, seed variance explodes 10×.
- Mid-net activation code ceiling: eff dim ≈ 14.0 (saturates for every C ≥ 25,
  converged or not). **C₅₀ / ceiling = 2.73.**

**Registered prediction for B1 (width sweep):** C₅₀(w) ≈ 2.7 × ceiling(w).
If the ceiling scales with width (measure it at w = 64/128/512), the wall
crossing measured by b1_* configs should land at 2.7× it. Failure modes are
informative either way: constant ceiling → capacity is architectural, not
width-bound; proportional ceiling with different multiplier → the 2.7 is
budget- or optimizer-dependent (B3's budget arms test that).

## S1 — separation anomaly resolved: rank is geometry, amplitude is economics

L0 effective dim is non-monotone in separation (33/28/27/36/57 for sep
1.5/2/3/4.5/6) while significant dims stay exactly C−1 = 9 everywhere.
Decomposition via the two-floor census (all 20-seed means):

| sep | eff dim | bulk scale | spike mass | top-1 frac |
|-----|---------|-----------|------------|------------|
| 1.5 | 33.2 | 0.592 | 0.501 | 0.064 |
| 2.0 | 27.6 | 0.496 | 0.556 | 0.069 |
| 3.0 | 26.8 | 0.481 | 0.565 | 0.070 |
| 4.5 | 36.4 | 0.648 | 0.475 | 0.058 |
| 6.0 | 56.8 | 0.792 | 0.356 | 0.045 |

The eff-dim curve is the mirror image of **spike mass fraction** (energy in
the C−1 simplex directions / total), which peaks at intermediate difficulty;
bulk depletion co-varies (deepest at sep 2–3). Spike *spread* (cv) is flat —
the spikes stay proportioned; only their collective amplitude moves.

Reading: the input layer's structural **rank** is fixed by task geometry
(always the class simplex), but its structural **amplitude** follows training
economics — maximal where learnable signal × achievable margin peaks. Easy
tasks (sep 6): small weight updates suffice, bulk barely touched. Near-chance
tasks (sep 1.5, Bayes 0.505): little margin to earn, weaker gradients, less
investment than the sweet spot. This cleanly separates the two axes the
two-floor census was built to separate, on the first dataset it was applied to.
