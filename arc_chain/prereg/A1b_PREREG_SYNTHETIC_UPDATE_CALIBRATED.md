# A1b preregistration: A1 redesigned after A1 proved uninformative

*Written: 2026-10-23 17:30 UTC, before any A1b net is built. A1: `A1_PREREG_SYNTHETIC_UPDATE.md` (commit 0da83a9, sha256 32eba9f0...).*

## What happened to A1
The A1 nets were built and run as preregistered. The chain returned non-finite values on 14 of 15 nets with an update (S1-S5). Diagnosis
(from the saved truths, not from any chain statistic): the synthetic nets are in the wrong activation regime. Their per-layer mean activation
decays by 2-9 orders of magnitude over 16 layers (S1: rms 0.12 at layer 0 to 5e-4 at layer 15; S2: to 2e-11), because the update's gain
applies only to a 16-dimensional signal subspace that receives ~1.6% of the N(0,I) input variance, while trained nets keep O(1) activations.
A1's P1-P3 are therefore **not evaluable**, neither passed nor failed; its results file records that. A1b is a design change, so it has
its own preregistration; it is not a re-run of A1 with tuned parameters.

## Change (the only one)
After building W_l = alpha W0_l + Delta_l exactly as in A1, rescale each layer by a scalar g_l so the layer's post-ReLU root-mean-square
activation equals 1 (the He fixed point) on a calibration draw of 4096 N(0,I) inputs, propagated through the already-rescaled earlier layers.
ReLU is positively homogeneous, so this changes no direction or structure, only per-layer scale.
Gate (regime, not outcome): a net is analysed only if the Monte Carlo final-layer mean activation rms lies in [0.3, 2.0] (trained
nets: ~0.8); gated-out nets are counted and reported.

## Everything else is A1's
Settings S0-S5 (3 seeds each), truth (4e6 samples), chain (frozen 331937, flopscope 0.12.3), statistics, and **the predictions A1-P1 to A1-P4
with the same thresholds** (P1: S1 relative final-layer error >= 1e-3; P2: S1 held-out coherent share >= 0.5 and slope a in [-0.20, -0.05];
P3: S2 coherent share < 0.3 and error <= 0.1 x S1's; P4: direction of relative error across r = 4, 16, 64). Interpretation as in A1:
all of P1-P3 pass means an aligned low-rank update on a shrunken remnant is sufficient for the coherent shortfall.
Non-finite chain outputs are excluded and counted; if fewer than 2 finite nets remain in S1 or S2, that prediction is reported not evaluable.
