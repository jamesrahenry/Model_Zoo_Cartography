# Preregistration 3: decoupling decay from training time

*Written: 2026-10-24 14:59 UTC, before any grid-3 net is trained. Grid 1: PREREG.md (b385b89); grid 2: PREREG2.md (e239bf5). Grid 2 results used here are
the interim 8 of 9 rows; row 9 (60 000 steps) was still running when this was written.*

## Why
Grid 2's Q3 failed on the rows available: the surviving init amplitude alpha = exp(-lr*wd*(steps-250)) did not control how much the net depends
on its random draw. At alpha ~ 0.55, (lr 1e-4, wd 0.3) gave re-randomised accuracy 0.23 of full while (lr 3e-4, wd 0.1) gave ~0.83 of full.
An exploratory (not preregistered) look at 354 nets from both grids found that within 1024 x 16, C = 10 the relative update size
rho = ||Delta||_F / ||alpha W0||_F (mean over layers 3 .. depth-2) tracked dependence better than -ln(alpha) (Spearman 0.92 vs 0.74), while across
all nets the reverse held (0.46 vs 0.78). Hypothesis: dependence is set by how much the update has grown relative to what decay has shrunk, so
training time and decay act separately. This grid holds alpha fixed while varying training time.

## Design (1024 x 16, GMM, C = 10, bias-free ReLU, AdamW, lr 3e-4, batch 512, fp32, 8 fresh seeds 10-17 per cell)
Training time T = lr * (steps - 250): short T = 0.83 (3 000 steps), base T = 5.9 (20 000), long T = 17.9 (60 000).
Weight decay is chosen per T so nominal alpha is 0.55 or 0.17, plus alpha = 1 (wd = 0):
- steps 3 000:  wd {0, 0.72, 2.15};   steps 20 000: wd {0, 0.1, 0.3};   steps 60 000: wd {0, 0.033, 0.099}.
9 cells x 8 seeds = 72 nets. Per net: the same analysis as grids 1-2 (grid/analyze_net.py).
Inclusion: all nets except outcome "stalled" (final accuracy < chance + 0.10). Excluded nets counted per cell.
Reference set for H5: the grid-1 C = 10 nets (42) and the grid-2 B4 rows (lr 1e-3, lr 1e-4, 5 000 steps; and 60 000 steps if finished), 1024 x 16, C = 10.

## Predictions
**H1. Training time lowers dependence at fixed decay level.** Within each nominal-alpha group (0.55: wd 0.72/0.1/0.033; 0.17: wd 2.15/0.3/0.099),
Spearman(T, re-randomised accuracy ratio = rerandom/full) >= 0.7 over its 24 nets. Pass: both groups.
**H2. Decay lowers dependence at fixed training time.** Within each T group, Spearman(-ln alpha_pred, ratio) >= 0.7 over its 24 nets (alpha_pred from the actual
lr, wd, steps). Pass: at least 2 of 3 groups.
**H3. The relative update size beats alpha.** Across the 72 nets, Spearman(rho, ratio) exceeds Spearman(-ln alpha_pred, ratio) by >= 0.10.
**H4. Dependence is a one-dimensional function of rho.** Order the 9 cells by mean rho; the cell-mean ratios are non-decreasing in that order with at most
one adjacent inversion, each inversion smaller than 0.05. (wd = 0 cells are included.)
**H5. Out-of-sample prediction.** Fit an isotonic regression of the ratio on rho using the reference set only; apply it to the 72 grid-3 nets.
Pass: mean absolute error <= 0.15. Report the same for -ln alpha_pred as the comparator; the prediction is that rho's error is lower.

## Commitments
Failed predictions first; 95% bootstrap intervals over nets; no threshold changes after seeing data; deviations labelled. rho is computed as the
mean of ||Delta||_F / ||alpha W0||_F over layers 3 .. depth-2 from the per-net decomposition (the analysis already stores it).
