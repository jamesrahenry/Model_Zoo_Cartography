# A1 preregistration: is a low-rank aligned update, without any training, enough to break the chain coherently?

*Written: 2026-10-23 17:09 UTC, before any synthetic net is built or run.*

## Question
On trained nets (MZC grids) the frozen 331937 chain makes a coherent, same-signed shortfall (truth ~ 1.09-1.16 x prediction at
layer 15; held-out coherent share 0.86-0.88 at wd 0.3), while on random nets its error is incoherent (share ~ 0). Trained weights are a
shrunken copy of the init plus a low-rank update. Is such a structure *sufficient*, with no task, data or optimiser?

## Construction (1024 x 16, bias-free ReLU, He-Gaussian W0 per layer)
W_l = alpha * W0_l + Delta_l.  Aligned family: Delta_l = sqrt(2) * Q R_l Q^T, Q a fixed random n x r orthonormal basis shared by all
layers, R_l a random r x r orthogonal matrix (so Delta_l maps span(Q) into itself with gain sqrt(2), the He gain through a ReLU).
Misaligned control: Delta_l = sqrt(2) * U_l R_l V_l^T with independent random orthonormal U_l, V_l per layer (same spectrum, no shared subspace).
Settings (3 seeds each): S1 alpha 0.18, r 16, aligned | S2 alpha 0.18, r 16, misaligned | S3 alpha 0.18, r 64, aligned |
S4 alpha 0.18, r 4, aligned | S5 alpha 0.55, r 16, aligned | S0 alpha 1, no update (pure He, control).
(The relative update norm ||Delta||_F / ||alpha W0||_F = sqrt(r) / (alpha sqrt(n)) is 0.69 for S1; measured in wd = 0.3 nets: 0.6-0.8.)
Truth: Monte Carlo per-layer means, 4e6 samples of N(0,I). Chain: frozen 331937 estimator, flopscope 0.12.3, unchanged.
Statistics: relative final-layer error ||e||^2/||truth||^2 (noise-debiased); held-out coherent share at layer 15 (leave-one-net-out per-layer
fit e ~ a*m + b*1, pooled within a setting; same code as `mzc_trained_vs_init_anatomy.py`); fitted slope a. Nets on which the chain returns
non-finite values are excluded and counted.

## Predictions
**A1-P1.** S1's mean relative final-layer error >= 1e-3 (the trained nets sit at 1e-2; the pure-He control S0 sits at ~3e-8).
**A1-P2.** S1's held-out coherent share at layer 15 >= 0.5 and its fitted slope a in [-0.20, -0.05].
**A1-P3.** The misaligned control S2 has coherent share < 0.3 and mean relative error <= 0.1 x S1's.
**A1-P4.** Across S1, S3, S4 (rank 16, 64, 4) the relative error is monotone in r (either direction), reported with the direction.
A1-P1..P3 pass/fail as stated; P4 is descriptive but its direction is committed in the write-up either way.

## Reading
All of P1-P3 pass: an aligned low-rank update on a shrunken random remnant is sufficient for the coherent shortfall (training is not
needed for it). P1 and P3 pass but P2 fails: the update breaks the closure but not in the observed coherent way, so something else in
training shapes the sign and size. P3 fails: alignment is not the cause; any rank-r update does it.
