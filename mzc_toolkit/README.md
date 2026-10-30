# mzc_toolkit

Instruments for finding structure in trained **bias-free ReLU MLPs** (`x @ W`, every layer n x n), built for the ARC Phase 2 write-up.
numpy only. Every number below is a measurement from our preregistered tests, on 1024 x 16 nets trained as classifiers (Gaussian mixtures, AdamW);
nothing is claimed for other architectures, optimisers or for transformers. Test files, scripts and results live in the sibling `grid/` and `arc_chain/` folders.

| Stage | Needs | Call | What you get | Measured | Fails when |
|---|---|---|---|---|---|
| 0 | weights only | `triage(Ws)` | `alpha_hat` per layer; outlier counts; cross-layer `chain_score` and a detection flag | alpha_hat median error 0.004 (layers >= 2); chain score >= 10x chance in 92% of 51 trained nets, their inits <= 0.016 | layers 0-1 (rewritten by training); says that and roughly where, not what |
| 0b | weights only | `chained_subspaces(Ws)` | cross-layer subspaces from prefix/suffix layer products | rebuilding a net from the rank-16 slice touching them (rest re-randomised) kept 95% accuracy in 77% of 64 draw-dependent nets (29/43 with no decay), a random subspace needs ~1024; per-layer singular subspaces 5% | only 14% reach ARC-style fidelity; does not recover the update's own subspace (capture 0.47); GMM tasks only |
| 1 | weights + the net's own init or an early checkpoint | `decompose(Ws, W0s)`, `rebuild_with_slice` | alpha, the update Delta, relative size rho, the aligned slice | rho predicts dependence on the random draw (Spearman 0.90, held-out MAE 0.122); slice rank about 2.6 x classes | needs the reference; Fashion-MNIST needed a larger slice |
| 2 | running data | `fidelity_curve`, `accuracy` | what is necessary: accuracy and ARC-style fidelity of the rebuilt net | the only route to "necessary" | costs forward passes; N(0, I) inputs vs the task's distribution can disagree |

Recipe shortcut: for AdamW, alpha = exp(-lr * wd * (steps - 250)) (500 warmup steps), matching 162 nets to within 0.019, without looking at the weights.

## Command line
```
python -m mzc_toolkit triage NET.npz            # Stage 0 (NET.npz keys w0..w15)
python -m mzc_toolkit decompose NET.npz INIT.npz   # Stage 1 (INIT.npz keys w<l> or init_w<l>)
```

## What this does not do
Weights-only extraction preserves *accuracy* (`chained_subspaces`) but not ARC-style fidelity; our per-layer and shared-subspace attempts failed outright (see the `phase2/I1-I4` write-ups). Judged in ARC's metric, discarding the remnant is rarely free: in low-decay nets the random-looking part contributes.
