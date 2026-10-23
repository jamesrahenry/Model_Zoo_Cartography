# Preregistration 2: does the remnant law generalise? Architecture, real data, optimiser dynamics

*Written: 2026-10-23 16:55 UTC, before any run of this second grid is trained or analysed. First grid: `PREREG.md` (b385b89), all five
predictions passed. This file is committed before the jobs are queued; its sha256 and commit are in the job specs.*

## Why

The first grid fixed one architecture (1024 x 16), one task family (Gaussian mixtures) and one recipe (AdamW, lr 3e-4,
20 000 steps). Its surviving-amplitude values matched a closed form: AdamW's decoupled decay shrinks the init copy by
exp(-lr * wd * t) (observed 0.55 at wd 0.1, 0.83 at 0.03, 0.17 at 0.3; predicted 0.55, 0.84, 0.165). If that is the mechanism,
effective decay (lr * wd * steps) and not wd alone should control dependence on the random draw, and the law should survive
changes of architecture and of task. Nothing below is tuned to this grid's data.

## Design (all: bias-free ReLU, He-Gaussian init, batched trainer, AdamW / Adam at wd = 0, batch 512, fp32, 6 seeds per cell)

- **B3 architecture**: GMM, C = 10, wd {0, 0.1, 0.3}, lr 3e-4, 20 000 steps; (width, depth) in {(512,16), (1024,8), (1024,24)}.
- **B2 real data**: whitened MNIST and Fashion-MNIST (784 -> 512 seeded projection + ZCA), width 512, depth 16, wd {0, 0.1, 0.3}, lr 3e-4, 20 000 steps.
- **B4 optimiser dynamics** at 1024 x 16, GMM, C = 10, wd {0, 0.03, 0.3}: (lr 1e-3, 20 000 steps), (lr 1e-4, 20 000), (lr 3e-4, 5 000), (lr 3e-4, 60 000).

Per net, the same analysis as the first grid (`grid/analyze_net.py`): decomposition, random-remnant ablation, aligned-slice search with a
random-subspace control, Monte Carlo truth at 4e6 samples. Real-data validation accuracy uses the held-out test split.
Inclusion: a net is included unless its outcome is "stalled" (final accuracy < chance + 0.10). Excluded nets are counted per cell.
Effective-decay prediction: alpha_pred = exp(-lr * wd * (steps - 250)) (the 250 accounts for the 500-step linear warmup).

## Predictions

**Q1. Surviving amplitude follows the closed form.** For every net with wd > 0: |alpha(layer 8) - alpha_pred| <= 0.03.
Pass: at least 90% of such nets. (For wd = 0, alpha >= 0.97.)

**Q2. Dependence on the draw falls with weight decay in every architecture and task.** For each of the five groups (three
architectures, MNIST, Fashion-MNIST): Spearman(wd, accuracy after re-randomising the remnant) >= 0.7 over its 18 nets.
Pass: at least 4 of 5 groups.

**Q3. Effective decay, not wd, controls dependence.** (a) Pooling the B4 nets with the first grid's C = 10 nets (reference), the
Spearman correlation between -ln(alpha_pred) and re-randomised accuracy is >= 0.85. (b) At wd = 0, re-randomised accuracy is
<= 0.5 x the full net's accuracy in every B4 row (no explicit decay: dependence stays high whatever the learning rate or length).
Pass: both.

**Q4. The load-bearing slice stays small.** Among nets whose re-randomised accuracy is < 0.5 x full, with k_crit the smallest k that
restores >= 0.95 x full accuracy: median k_crit <= 64 within each GMM group (B3 and B4 rows), and <= 128 within each real-data group;
and the random-subspace control needs >= 4 x the aligned median in every group. Pass: all groups.

**Q5. Chain error coherence follows effective decay (seed-0 nets, laptop, frozen 331937 chain).** Spearman(-ln alpha_pred, held-out
coherent share at layer 15), pooled over B3 and B4 settings with >= 3 finite chain outputs, >= 0.6. If the frozen chain cannot run
at a row's width or depth, that row is excluded and reported, not imputed.

## Commitments

Failed predictions are reported first. Intervals: 95% bootstrap over nets. No threshold is changed after seeing results; deviations
are labelled as such. The first grid's nets are used only as the Q3(a) reference.
