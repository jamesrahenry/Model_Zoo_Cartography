# Preregistration: weight decay x task-size grid at ARC Phase-2 architecture

*Written: 2026-10-22 17:20 UTC, before any grid net is trained or analysed. This file is committed first; its
commit hash and the SHA-256 of this file are recorded in the first commit message and in every job spec.*

## Why

Pilot results (45 trained nets from `p2_w1024_d16_*`, 6 nets from a small weight-decay sweep, all in
`arc-whitebox-canary/phase2/harness/mzc_*_results.json`) were exploratory: 2-4 nets per cell, one task per
weight-decay setting. This grid is the confirmatory run. Nothing below is tuned to the grid's data.

## Design

Architecture 1024 x 16, bias-free ReLU, He-Gaussian init, GMM classification (separation 3.0, task seed 0,
whitened aggregate input), AdamW (Adam at wd = 0), lr 3e-4, 20 000 steps, batch 512, fp32, the batched trainer
(`train/train_mlp_batched.py`).
Grid: weight decay {0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0} x classes C {2, 5, 10, 25, 50} x seeds 0-5 = 210 nets.

Per net (`grid/analyze_net.py`): validation accuracy of the full net; accuracy after re-randomising the random
remnant alpha*W0 ("rerandom"); the aligned-slice search over k in {0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
with a random-subspace control; the weight decomposition W = alpha*W0 + Delta; Monte Carlo per-layer means
(4e6 samples of N(0,I)) for later chain-error comparison on seed-0 nets.

Inclusion: a net is analysed as "trained" only if its final validation accuracy is >= 2 x chance and its
outcome is not "stalled"; excluded nets are counted and reported per cell, never dropped silently.

## Predictions (each stated so it can fail)

**P1. Dependence on the specific random draw falls monotonically with weight decay.**
Statistic: for each C, the Spearman correlation across seeds and weight-decay settings between wd (the 7 values;
wd = 0 ranked lowest) and the accuracy after re-randomising the remnant, `rerandom_acc`.
Pass: Spearman >= 0.8 for at least 4 of the 5 values of C. Fail otherwise.
Expected shape from the pilot: `rerandom_acc` rises from ~0.15 (wd <= 0.003) through ~0.4 (wd 0.03) to ~0.85 (wd 0.1)
and ~0.89 (wd 0.3) at C = 10. We expect wd = 1.0 either to stay near-inert or to fail to train; either is reported.

**P2. The load-bearing slice is small and its size grows with task size.**
Define k_crit as the smallest k in the list at which the aligned slice re-randomised restores accuracy to
>= 0.95 x the full net's accuracy.
Pass, jointly: (a) among nets with wd <= 0.03 and `rerandom_acc` < 0.5 x full, Spearman(C, k_crit) > 0 over all
such nets; (b) the median of k_crit / C lies in [1, 8]; (c) in the random-subspace control, the median k needed
for the same recovery is >= 8 x the aligned k_crit.
Fail if any of (a), (b), (c) fails; each is reported separately.

**P3. Energy share does not predict dependence.**
Statistic: Spearman between the layer-8 random share of weight energy, 1/(1 + delta_rel^2), and `rerandom_acc`
across all included nets, versus the same statistic for wd alone.
Pass: |Spearman(energy share, rerandom_acc)| < |Spearman(wd, rerandom_acc)| and the energy share is within 0.05 of
0.92 for all wd <= 0.1 (as in the pilot). Fail otherwise.

**P4. The chain's error becomes coherent as weight decay rises (seed-0 nets only, run on the laptop CPU).**
Chain: the frozen 331937 estimator, unchanged. Statistic: per weight-decay setting, the held-out coherent share at
layer 15 from the leave-one-net-out per-layer fit e ~ a*m + b*1 (`phase2/harness/mzc_trained_vs_init_anatomy.py`),
pooled over C within a setting; nets on which the chain returns non-finite values are excluded and counted.
Pass: Spearman(wd, coherent share) >= 0.7 over the 7 settings, and the share at wd >= 0.1 exceeds 0.5.
Fail otherwise. The pilot suggests the share is ~0.08 at wd 0 and ~0.88 at wd 0.3.

**P5. The chain returns non-finite values on low-code-rank nets.**
Pass: the fraction of nets with non-finite chain output is larger at C <= 3 (wd >= 0.003) than at C >= 10, in the
seed-0 chain runs; C = 2 is run in the grid for this purpose.

## Analysis commitments

- Intervals: 95% bootstrap over nets within a cell; reported with n.
- No fitting of any threshold to grid data. The only thresholds are those written above.
- Results are reported whether or not they pass. Failed predictions are listed first.
- The pilot nets are not pooled into the grid statistics.
