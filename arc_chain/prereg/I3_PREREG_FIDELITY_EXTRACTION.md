# I3 preregistration: extraction judged by fidelity, in ARC's own currency

*Written: 2026-10-26 14:02 UTC, before any fidelity curve is computed. Motivation: James's objection to I1/I2 (95% accuracy recovery discards signal; anything that contributes is structure).
I1: I1_PREREG (d7f23f6); I2: I2_PREREG (89fc574).*

## Metric
For a trained net and a rebuilt net (same architecture, some layers' weights replaced), eps = ||m' - m||^2 / ||m||^2, where m, m' are the per-neuron mean final-layer post-ReLU
activations of the original and rebuilt net over the same 16 384 draws of N(0, I) input (paired, so sampling noise cancels). This is ARC's score, applied to the rebuilt net
as if it were the predictor. Reference: eps_chain = the frozen 331937 chain's relative error ||pred - truth||^2 / ||truth||^2 at the final layer on the same net (truth = the 4e6-sample
Monte Carlo mean; noise-debiased). A discarded part is "free" if the signal it costs, eps, is no larger than eps_chain.

## Methods compared (same rest-replacement as before; fresh He draw at the layer's alpha or alpha-hat)
- INIT: the aligned slice from Delta = W - alpha W0 (needs the init; regenerated exactly from the seed, verified: alpha and rho match stored values to 4 digits).
- I1: top-k singular subspaces of each W_l (weights only). - I2: one shared subspace across layers (weights only).
k in {0,2,4,8,16,32,64,128,256,512,1024}. k_fid(method) = smallest k with eps <= eps_chain. (k = 1024 keeps the net, eps = 0, so k_fid <= 1024 always.)

## Data
The 64 dependent nets of I1 (init-based re-randomised accuracy < 0.5 x full), and the grid-4 nets with layer-8 alpha <= 0.2 (24 nets), restricted to nets with a chain prediction on disk
(grid 4 all; grids 1-3 seed-0). Statistics that involve eps_chain use only nets with a chain prediction and finite output; the count is reported.

## Predictions
**I3-P1.** At the INIT method's accuracy-based k (k_init, the smallest k restoring 95% accuracy), eps > eps_chain in at least 50% of dependent nets: 95% accuracy recovery discards more signal than the estimator's error.
**I3-P2.** For INIT, median over dependent nets of k_fid / max(k_init, 1) >= 4: fidelity needs at least 4x the directions that accuracy does.
**I3-P3.** For both weights-only methods, median k_fid is >= INIT's median k_fid (weights-only is no better than the init-based extraction at fidelity), reported per method.
**I3-P4.** In the grid-4 high-decay nets (alpha <= 0.2, "inert remnant"): INIT at k = 0 (everything outside the update replaced) has eps > eps_chain in at least 70% of nets: the remnant is not inert at the chain's resolution.
Descriptive, no threshold: draw-to-draw spread of eps at k_init over two independent fresh draws, and the full curves eps(k) per method.
Groups with fewer than 6 nets: not evaluable, not passed.

## Commitments
Failed predictions first; no threshold changes after seeing data; deviations labelled.
