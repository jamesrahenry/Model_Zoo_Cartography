# I1 preregistration: can trained structure be extracted from the weights alone?

*Written: 2026-10-25 14:10 UTC, before the init-free procedure is run on any net.*

## Procedure (uses only a net's final weights W_l; no init, no truth, no data)
1. **alpha-hat.** alpha_hat_l = median(sigma(W_l)) / m0, m0 the median singular value of a He-Gaussian 1024 x 1024 matrix (entry std sqrt(2/n)).
2. **Structure subspaces.** SVD of W_l; U_k, V_k = top-k left/right singular vectors; P = U_k U_k^T, Q = V_k V_k^T.
3. **Extraction.** Keep the part of W_l touching those subspaces and replace the rest with a fresh He draw at amplitude alpha_hat:
   W'_l = [P W + W Q - P W Q] + alpha_hat * (I-P) Wn (I-Q), Wn a fresh He-Gaussian draw. (k = 0: full re-randomisation at alpha_hat; k = n: W.)
4. **Outlier count.** r_hat_l = number of singular values of W_l above 1.05 * alpha_hat_l * s_max, s_max the largest singular value of the same He reference.
Control: the same with random orthonormal U, V of rank k (init-free random-subspace control).
Verification: task accuracy of W' (GMM validation draw; val seed = net seed + 1 500 000) against the full net.

## Data
All GMM nets with weights on disk and an init-based analysis record: grid 4 (72 nets), and the seed-0 weights of grids 1, 2 (GMM rows) and 3. MNIST and Fashion nets are not used (task data
path differs). A net is "dependent" if its init-based re-randomised accuracy is < 0.5 x its full accuracy (from `analyze_net` output). Extraction is evaluated on dependent nets only
(on non-dependent nets full re-randomisation already works, so extraction is vacuous). k_free = smallest k in {0,2,4,...,1024} restoring >= 0.95 x full accuracy.
k_init = the same quantity from the init-based aligned-slice search already stored per net. **Success** on a net: k_free <= max(2 * k_init, 32).

## Predictions
**I1-P1 (alpha-hat).** |alpha_hat_l - alpha_l| <= 0.03 for at least 90% of (net, layer) pairs, all layers, all nets in the data.
**I1-P2 (works when the remnant is shrunk).** Among dependent nets with layer-8 alpha <= 0.6: success rate >= 70%.
**I1-P3 (fails when it is not: BBP).** Among dependent nets with layer-8 alpha >= 0.95: success rate <= 30%.
**I1-P4 (detectable from the spectrum).** Across dependent nets, the Spearman correlation between the mean over layers 3..depth-2 of r_hat and the success indicator is >= 0.5.
**I1-P5 (beats a random subspace).** Among successful nets, the median k needed by the init-free random-subspace control is >= 4 x the median k_free.
Minimum group size: a prediction whose group has fewer than 6 dependent nets is reported "not evaluable", not passed.
Reading: P1-P5 all pass => trained structure and its remnant fraction can be recovered from W alone exactly where theory says (above the bulk edge), and fails where it should.
Failure of P3 (extraction works even undecayed) would mean the spectral threshold is not the limit.

## Commitments
Failed predictions first; no threshold changes after seeing data; deviations labelled.
