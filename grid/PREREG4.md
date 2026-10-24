# Preregistration 4: does the chain's coherent error follow decay (alpha) or update size (rho)?

*Written: 2026-10-24 21:25 UTC, before any grid-4 net is trained. Grids 1-3: PREREG.md, PREREG2.md, PREREG3.md. Grid 3's single-net exploratory look (n = 9) found the chain's
per-net error alignment tracked -ln(alpha) (Spearman 0.83) and not rho (0.35), the reverse of dependence on the random draw (rho 0.90 vs alpha 0.69). This grid
tests that with 8 chain runs per cell and held-out fits.*

## Design
Grid 3's nine cells (1024 x 16, GMM, C = 10, lr 3e-4; steps 3 000 / 20 000 / 60 000 with wd {0, 0.72, 2.15} / {0, 0.1, 0.3} / {0, 0.033, 0.099}), **fresh seeds 20-27**,
8 nets per cell = 72 nets. Every net's weights are saved (`--weights-all`), the frozen 331937 chain (flopscope 0.12.3, unchanged) is run on all of them on the laptop CPU,
against the Monte Carlo truth (4e6 samples) produced in the same job. Inclusion: not "stalled"; nets with non-finite chain output are excluded from the fits and counted.

## Statistics
Per cell: held-out coherent share at the last layer = 1 - (squared error after removing a per-layer fit e ~ a*m + b*1 learned on the OTHER 7 nets) / (squared error),
noise-debiased; computed only for cells with >= 6 finite chain outputs. Per cell also: mean -ln(alpha_pred) (alpha_pred = exp(-lr*wd*(steps-250))) and mean rho
(mean over layers 3..depth-2 of ||Delta||_F / ||alpha W0||_F).

## Predictions
**K1.** Across the nine cells, Spearman(coherent share, -ln alpha_pred) >= 0.7, and exceeds Spearman(coherent share, rho) by >= 0.2.
**K2.** At the three alpha = 1 cells (wd = 0), coherent share < 0.3 at every training length (rho reaches ~0.36 at 60 000 steps there).
**K3.** At the three alpha ~ 0.17 cells (wd 2.15 / 0.3 / 0.099), coherent share >= 0.5 at every training length.
**K4.** The chain returns non-finite values on at most 20% of nets in every cell.
Pass/fail as stated. If a cell has fewer than 6 finite outputs, its share is reported as not evaluable and the predictions that need it are reported as not evaluable (not passed).

## Commitments
Failed predictions first; 95% bootstrap intervals over nets for cell shares; no threshold changes after seeing results; deviations labelled.
