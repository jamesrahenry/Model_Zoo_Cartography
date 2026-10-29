# mzc_toolkit

Instruments for finding structure in trained **bias-free ReLU MLPs** (`x @ W`, every layer n x n), built for the ARC Phase 2 write-up.
numpy only. Every number below is a measurement from our preregistered tests, on 1024 x 16 nets trained as classifiers (Gaussian mixtures, AdamW);
nothing is claimed for other architectures, optimisers or for transformers. Test files, scripts and results live in the sibling `grid/` and `arc_chain/` folders.

| Stage | Needs | Call | What you get | Measured | Fails when |
|---|---|---|---|---|---|
| 0 | weights only | `triage(Ws)` | `alpha_hat` per layer; outlier counts; cross-layer `chain_score` and a detection flag | alpha_hat median error 0.004 (layers >= 2); chain score >= 10x chance in 92% of 51 trained nets, their inits <= 0.016 | layers 0-1 (rewritten by training); says that and roughly where, not what |
| 1 | weights + the net's own init or an early checkpoint | `decompose(Ws, W0s)`, `rebuild_with_slice` | alpha, the update Delta, relative size rho, the aligned slice | rho predicts dependence on the random draw (Spearman 0.90, held-out MAE 0.122); slice rank about 2.6 x classes | needs the reference; Fashion-MNIST needed a larger slice |
| 2 | running data | `fidelity_curve`, `accuracy` | what is necessary: accuracy and ARC-style fidelity of the rebuilt net | the only route to "necessary" | costs forward passes; N(0, I) inputs vs the task's distribution can disagree |

Recipe shortcut: for AdamW, alpha = exp(-lr * wd * (steps - 250)) (500 warmup steps), matching 162 nets to within 0.019, without looking at the weights.

## Command line
```
python -m mzc_toolkit triage NET.npz            # Stage 0 (NET.npz keys w0..w15)
python -m mzc_toolkit decompose NET.npz INIT.npz   # Stage 1 (INIT.npz keys w<l> or init_w<l>)
```

## What this does not do
No weights-only method here recovers the *load-bearing* structure with the function intact (our per-layer and shared-subspace attempts failed; see `phase2/I1-I3` write-ups). Judged in ARC's metric, discarding the remnant is rarely free: in low-decay nets the random-looking part contributes.
