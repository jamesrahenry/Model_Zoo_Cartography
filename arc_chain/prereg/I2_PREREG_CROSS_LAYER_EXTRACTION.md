# I2 preregistration: cross-layer, weights-only extraction of trained structure

*Written: 2026-10-25 23:14 UTC, before the shared-subspace extractor is run on any net. I1: `I1_PREREG_INIT_FREE_EXTRACTION.md` (d7f23f6), result committed (25c3c55): per-layer
spectral extraction mostly failed.*

## Motivation and what was already seen (exploratory premise check, 10 MZC corpus nets, run before this preregistration)
The trained update Delta_l = W_l - alpha_l W0_l has input/output subspaces that chain across layers: mean squared cosine between layer l's output subspace and layer l+1's input
subspace (rank 16) is 0.42-0.65, against chance 16/1024 = 0.016. The top-16 singular subspaces of W itself (no init) chain at 0.31 (wd 0, 0.003) to 0.58 (wd 0.3). So
cross-layer structure is visible from the weights alone in exactly the nets where per-layer spectral extraction (I1) failed. This preregistration tests whether a single subspace
shared across layers, estimated from W alone, recovers the load-bearing structure.

## Procedure (weights only: no init, no data, no truth)
1. Per layer l: top-k left/right singular vectors U_l, V_l of W_l (k as below).
2. Shared subspace Q-hat = top-k eigenvectors of M = sum_l (U_l U_l^T + V_l V_l^T).
3. alpha_hat_l from the bulk (median singular value / He median, as in I1).
4. Extraction at rank k: P = Q-hat Q-hat^T; W'_l = [P W + W P - P W P] + alpha_hat_l (I-P) Wn (I-P), Wn a fresh He draw.
   Control: the same with a random orthonormal Q of rank k. k in {0,2,4,8,16,32,64,128,256,512,1024}; the singular-subspace rank used in step 1 is k_s = 16 throughout.
5. Chain score S(W) = mean over consecutive layers of ||U_{l+1}^T V_l||_F^2 / k_s.

## Data
(a) For P1-P2 (need the init for ground truth): the MZC corpus nets that store init_w (p2_w1024_d16 families, wd 0.3 and 0, and the wd-sweep nets).
(b) For P3-P4 (init used only for the pre-existing init-based baseline): the 137 GMM nets of I1 (grid 4, grids 1-3 seed-0), same "dependent" rule (init-based re-randomised accuracy < 0.5 x full)
and the same k_init, success rule max(2*k_init, 32), and groups (layer-8 alpha <= 0.6, >= 0.95) as I1.

## Predictions
**I2-P1 (chaining is detectable).** S(W) >= 0.156 (10 x chance) for at least 90% of corpus nets, while S of their own initialisations (init_w, He random) is <= 0.047 (3 x chance) for all of them.
**I2-P2 (shared subspace recovered, also without decay).** Truth Q_Delta = top-16 eigenspace of the same projector sum built from Delta. Capture = ||P_Qhat Q_Delta||_F^2 / 16. Median capture >= 0.8 among corpus nets with
layer-8 alpha >= 0.95, and >= 0.8 among nets with alpha <= 0.6.
**I2-P3 (extraction works).** Success rate (k_Q <= max(2*k_init, 32), k_Q = smallest k restoring >= 0.95 x full accuracy) >= 70% among dependent nets with alpha <= 0.6 AND >= 70% among dependent nets with alpha >= 0.95.
**I2-P4 (beats I1).** On the same dependent nets, success rate of the shared-subspace extractor minus I1's per-layer success rate >= 0.30 (I1: 18% for alpha <= 0.6, 0% for alpha >= 0.95).
**I2-P5 (beats a random subspace).** Among successful nets, median k of the random-subspace control >= 4 x median k_Q.
Groups with fewer than 6 nets: not evaluable, not passed. Reading: all pass => trained structure is recoverable from the weights alone once the cross-layer dimension is used,
including in undecayed nets where per-layer analysis cannot see it. P1-P2 pass but P3 fails => the structure is detectable but the shared-subspace model is not what the function uses.

## Commitments
Failed predictions first; no threshold changes after seeing data; deviations labelled.
