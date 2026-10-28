# [Phase 2 write-up] What the cumulant chain does on trained weights: a preregistered study of ~570 trained 1024 × 16 nets (submission #331937)

*Written: 2026-10-28 19:10 UTC. Forum post text, paste as-is. One placeholder remains: `[LINK: estimator release]`.*

Submission ID: #331937 (jamesrahenry; adjusted 7.04e-9, final-layer MSE 2.18e-8 at 32.3 % of budget, 0/100 failures; regraded unchanged under flopscope 0.12.3 as #337164). The estimator is not the contribution of this write-up, and we say so in the first section. Everything else here is about what that estimator does when the weights are not random.

## 1. What this is, and what it is not

Our graded entry is a clean-room port of the factorized K = 3 cumulant chain (Wu et al., arXiv:2605.05179; ARC's `mlp_cumulant_propagation`, MIT), built without any participant's code, with a memoryless κ4 rider and a 47-term closure at the table λ. It reached the V18 band from scratch and no further. It contains no new estimator idea, and we did not chase the frontier after mid-September. The reproducibility release pins the archive (sha256 `8b6b1083…`), the grade and the regrades; the source is at `[LINK: estimator release]`. Section 6 says, in four sentences, what building it taught us.

After the board's method space had closed, we turned the estimator around. The challenge's stated motivation is white-box estimation for real networks, yet every estimator on this board, ours included, is calibrated on He-Gaussian weights and has only ever been scored on them. Our Phase 1 write-up (topic 18097, §8) named the missing piece: a population of trained networks at the same architecture with shared statistics. We built that population (Model Zoo Cartography, below) and ran the frozen #331937 chain on it, unchanged, against Monte Carlo truth on ARC's own input law.

**Every claim below was preregistered before the nets it concerns were trained**, with the prediction file committed first and its commit and SHA-256 recorded (table in §8). Failed predictions are listed first, as each preregistration commits us to. Nothing was re-thresholded after seeing data; deviations are labelled.

## 2. Setup

**Nets.** Bias-free ReLU MLPs, width 1024, depth 16, He-Gaussian init, the Phase 2 architecture, trained as classifiers: Gaussian-mixture tasks with C ∈ {2, 5, 10, 25, 50} classes on a whitened input, plus whitened MNIST and Fashion-MNIST at width 512. AdamW (Adam at wd = 0), batch 512, fp32, 500-step warmup; learning rate, steps and weight decay vary by grid. The initialisation of every net is stored. About 570 trained nets across four preregistered grids (210 + 162 + 72 + 72) plus a 45-net pilot and a 10-net sweep; 137 of them also serve the extraction tests of §4.5.

**Decomposition.** Per layer, W = α·W0 + Δ, with W0 the net's own init, α the least-squares surviving amplitude, Δ the trained update. "Re-randomising the remnant" replaces α·W0 by a fresh He draw at the same amplitude and leaves Δ alone; the drop in task accuracy measures how much the net depends on its specific random draw. ρ = ‖Δ‖_F / ‖α W0‖_F, averaged over layers 3 … 14, is the relative update size.

**Chain and truth.** The frozen #331937 estimator, flopscope 0.12.3, no change of any kind. Truth: Monte Carlo per-layer post-ReLU means, 4 × 10⁶ draws of N(0, I), ARC's input law (noise ≤ 4e-7 against trained-net errors ≥ 1e-3). Per-layer error e = pred − truth. **Held-out coherent share** at the last layer: fit e ≈ a·m + b·1 (m the predicted mean) on the other nets of the same setting, apply to the held-out net, report the fraction of ‖e‖² it removes, noise-debiased. On isotropic error this is ~0; on a same-signed multiplicative shortfall it is ~1.

## 3. Failed predictions first

| id | prediction | result |
|---|---|---|
| Grid 2, Q3 | the surviving amplitude α = exp(−lr·wd·(t − 250)) controls dependence on the random draw | **fail.** Spearman(−ln α, re-randomised accuracy) 0.70 against 0.85 required. At α ≈ 0.55, (lr 1e-4, 20k steps) leaves re-randomised accuracy at 0.23 of full while (lr 3e-4, wd 0.1) leaves 0.83. At wd = 0 and lr 1e-3, re-randomised accuracy is 0.70 of full, above the 0.5 bound |
| Grid 2, Q4 | load-bearing slice ≤ 128 directions per layer on real data | **fail on one of eight rows**: Fashion-MNIST needs 192 (random-subspace control 512); the seven other rows pass at 16–64 |
| Grid 2, Q5 | chain coherence tracks −ln α across architectures | **fail**: three evaluable settings, Spearman 0.5; the chain exhausts the FLOP budget at depth 24 and returns non-finite output on lr 1e-3 and some real-data nets |
| Grid 3, H4 | dependence is a one-dimensional function of ρ | **fail**: one adjacent inversion of 0.55. At equal ρ (0.32 vs 0.36) a decayed net keeps 0.94 of its accuracy under re-randomisation and an undecayed one 0.40 |
| Grid 4, K2 | undecayed nets give incoherent chain error at every training length | **fail**: undecayed nets trained 3 000 steps have coherent share 0.78 (CI 0.77–0.80); at 20 000 steps 0.006; at 60 000 steps 0.16. The 3 000-step failure is a real coherent error (relative error 5.5e-3, slope −0.07), not a small-error artefact |
| A1 | synthetic aligned update reproduces the coherent error | **not evaluable**: the synthetic nets left the trained activation regime (mean activation decays 2–9 orders over depth); 14 of 15 chain runs non-finite. Redesigned as A1b with a per-layer rescale, own preregistration |
| A1b, P1–P3 | same, in the right regime | **fail**: the aligned update produces coherent error (held-out share 0.70; misaligned control −0.002) but 200× too small (relative error 5.1e-5 vs ~1e-2 on trained nets) and with a positive slope where trained nets show −0.09 … −0.16 |
| I1, P1/P2/P4 | trained structure recoverable from each layer's weights alone | **fail**: α̂ within 0.03 on 85 % of layers (layers 0–1 are the misses); init-free extraction succeeds on 18 % of decayed dependent nets, 0 % of undecayed |
| I2, P2–P4 | a cross-layer shared subspace estimated from weights alone recovers the function | **fail**: shared subspace captured 0.45 (undecayed) / 0.77 (decayed) of the true one; function-preserving extraction 0 of 54 |
| Grid 1 | inclusion rule "accuracy ≥ 2 × chance" | **deviation**: unsatisfiable at C = 2 (chance 0.5), silently excluded 42 nets. Verdicts identical under the amended rule (exclude only "stalled"); both result files are published |

## 4. What held

### 4.1 The chain's error on trained weights is coherent; on random weights it is not

Pilot (45 trained nets, 14 paired inits, chain unchanged): on the 14 He initialisations the final-layer MSE is 2.1e-8, the error share along the prediction direction 0.012, held-out coherent share ≈ 0 from layer 8 on. On 30 nets trained with weight decay 0.3 the final-layer MSE is 8.6e-3, 0.89 of it along the prediction, held-out coherent share 0.88. The error is a same-signed multiplicative shortfall, truth ≈ 1.09–1.16 × prediction at the output, tight within a task family. A per-layer scale-and-shift fitted on the *other* task families cuts a held-out family's last-layer MSE to 0.07–0.27 ×. Grid 4 (72 fresh nets, every cell finite) reproduces it: 7 of 9 cells have coherent share 0.52–0.92; the two exceptions are the undecayed nets trained 20 000 and 60 000 steps (0.006 and 0.16; table in §4.3).

This is the same object as the "errors along the mean barely fade" clause of megan_park's survival law (18282, confirmed by dogus_ozel 18281 and AndreasHad04 18292). On random weights that direction carries ~1 % of the chain's error (our 10-20 measurement on 24 public nets: prediction-aligned share 0.011 at layer 15, held-out coherent share 0.000). On trained weights it carries nearly all of it.

### 4.2 How much of a trained net is still its initialisation

The init survives at amplitude α = exp(−lr·wd·(t − 250)): grid 2, Q1, 108 of 108 nets with decay within 0.03 of the closed form (max error 0.019), 54 of 54 undecayed nets at α ≥ 0.97, across widths 512 and 1024, depths 8, 16 and 24, GMM, MNIST and Fashion-MNIST. The update Δ is low-rank (participation ratio 8–29, about half its energy in the top 8 directions per layer). At wd 0.3 the random part still holds 0.60–0.67 of the weight energy in the middle layers; at wd 0, 0.91. So a closure that treats the weights as random can be mostly right in energy and still wrong coherently: the update sits in the signal directions.

### 4.3 What the chain's coherence tracks: decay, not update size

Grid 3 decoupled decay from training time (nine cells, α fixed at 1 / 0.55 / 0.17 by choosing wd per step count; 72 nets). For **dependence on the random draw**, ρ beats α: H1 (training time lowers dependence at fixed α) Spearman 0.83 and 0.78; H2 (decay lowers it at fixed time) 0.69 / 0.94 / 0.94; H3 Spearman(ρ) 0.90 vs Spearman(−ln α) 0.69; H5 out-of-sample isotonic fit on the earlier grids, MAE 0.122 for ρ vs 0.153 for α. But H4 failed (§3): decayed nets are more independent than undecayed ones at equal ρ, so it is not one variable.

For **the chain's coherence** the order reverses. Grid 4 (same nine cells, fresh seeds, chain on all 72 nets):

| steps \ nominal α | 1 (wd 0) | 0.55 | 0.17 |
|---|---|---|---|
| 3 000 | **0.78** (ρ 0.11) | 0.89 (ρ 0.12) | 0.83 (ρ 0.47) |
| 20 000 | 0.006 (ρ 0.29) | 0.68 (ρ 0.32) | 0.92 (ρ 0.66) |
| 60 000 | 0.16 (ρ 0.36) | 0.52 (ρ 0.39) | 0.88 (ρ 0.65) |

Cell entries: held-out coherent share at layer 15 (8 nets per cell, bootstrap CIs ±0.01–0.06). K1: Spearman(share, −ln α) 0.85 vs Spearman(share, ρ) 0.37. K3: every α ≈ 0.17 cell ≥ 0.83. K2 failed on the bold cell. Per-cell slopes a run from −0.07 (3 000 / wd 0) to −0.14 (20 000 / wd 0.3): the shortfall grows with decay. Reading, kept to what this shows: the function's dependence on its random draw and the chain's coherent error are governed by different variables, which is why we found no per-net relationship between them earlier (18 nets, |ρ| ≤ 0.49, n.s.). We do not have the law for the undecayed short-training cell.

### 4.4 The load-bearing part of the random remnant is small, but nothing is inert at the chain's resolution

Keeping only the slice of α·W0 that touches the rank-k input/output subspaces of the net's own Δ, and re-randomising the rest, restores ≥ 95 % of accuracy at k = 16–64 directions per layer (grid 1, median k_crit 16 / 16 / 64 / 128 for C = 5 / 10 / 25 / 50, Spearman with C 0.92); a random subspace of the same rank needs 4–32 × more in every grid 2 row that passed, and 2.7 × on Fashion-MNIST, the row that did not. That is accuracy. In ARC's currency (I3: ε = relative squared error of the rebuilt net's final-layer mean against the original, on N(0, I) inputs, compared with the chain's own relative error on the same net), 97 % of dependent nets lose more at their 95 %-accuracy rank than the chain's error; fidelity needs a median 64 × the rank accuracy does; and in the high-decay nets whose remnant is "inert" by accuracy (re-randomising costs ≤ 0.014), replacing the remnant costs ε = 0.68 against a chain error of 0.0135, so 100 % of them fail the inertness test, though k = 16 aligned directions bring ε to 0.0031.

### 4.5 Trained structure is not recoverable from the weights alone, at this resolution

Three weights-only extractors (I1 per-layer singular subspaces, I2 one subspace shared across layers, each with α̂ from the bulk spectrum). Detection works: I2-P1, 92 % of corpus nets have cross-layer subspace chaining ≥ 10 × chance, their own inits ≤ 0.016. Recovery does not: no weights-only method reaches the chain's error before keeping essentially everything (median k_fid 1024 for all three methods in I3-P3; at k = 512, INIT ε 0.013, I1 0.128, I2 0.295). The load-bearing slice is reachable from weights alone only at ~4 × the init-based rank and only in decayed nets.

### 4.6 A synthetic aligned update is enough for coherence, not for the size

A1b: W = 0.18·W0 + √2·Q R_l Qᵀ with one random 16-dimensional subspace Q shared across layers, per-layer rescaled to the He fixed point, no data, no optimiser. Held-out coherent share 0.70 (three seeds); the misaligned control with independent subspaces per layer, −0.002. So cross-layer alignment of the update is what makes the chain's error coherent, which is Artyom_Mazur's Proposition 1 (18249) seen from the other side: the next-layer error is a random projection of this layer's error only when the columns are independent of it. What the synthetic construction does not reproduce is the magnitude (200 × too small) or the sign of the slope, so training sets those by some other route.

## 5. What this says about the method family, kept narrow

1. The chain's random-init calibration is a property of the ensemble, not of the architecture. On trained weights the same code's error is an order of magnitude larger relative to the signal and almost entirely a multiplicative shortfall along the predicted mean, the one direction the survival law says the chain cannot forget.
2. The coherent error tracks how much of the initialisation decay has removed, not how much the task has moved the weights, with one unexplained regime (short undecayed training). A single per-layer scalar fitted on other tasks removes most of it at wd 0.3 and makes it worse at wd 0. Anyone extending these estimators to trained nets will need that scalar, and will need to know which regime they are in.
3. "Random" remains a good description of the bulk of a trained net's weights by energy, and a bad one at the resolution these estimators work at: 16 directions per layer out of 1024 carry what matters, and the chain cannot see which 16 without the init.
4. The chain is numerically fragile off-distribution: it returned non-finite output on every C = 2 net, on depth-24 nets within the budget, and on nets trained at lr 1e-3, while 0 of 14 initialisations failed. Any trained-weights benchmark for this family should report that count.

## 6. The graded submission, in four sentences

A correct-looking clean-room port was 65 × wrong for ten days because `dslices(gain)` used the unscaled A leg for carried slices, and every unit test exercised only the exact path; a further 10 × came from rank-truncating newborn hub slices that are closed-form. A fitted λ of 0.4, against the public 0.95, was absorbing 22 missing diagrams and a double-counted `wk431` term that partly cancelled; forcing λ to the reference value exposed both, for −22 % raw. We read the LuDoe / megan_park / yeongjunyoo exchange in 18251 (the inherited κ4 λ sits 2–5 × above the measured value and a truer value makes the chain worse) as the same phenomenon one level up: a fitted constant far from a reference is a symptom, not a parameter. Closed levers (CV sampling in the free window, Kerdock cubature, covariance-edge families, directional gate mixtures, learned correctors on the official panel) are tabulated in the release with their panels; where others closed the same doors (Susupro 18254, Nurlanggs 18252, kai_leal 18246) we point to them rather than repeat.

## 7. Limits

One chain, one input law (N(0, I), the tasks' whitened aggregate), one optimiser family. Mostly Gaussian-mixture tasks; two real datasets at width 512 only. Grid 1's P4 fits pool 3–4 nets per setting. ε in §4.4 is measured on N(0, I), not the task distribution. The chain's non-finite outputs are excluded and counted, never imputed. We have not tested whether any other team's estimator behaves the same way on these nets; the corpus is public so that someone can.

## 8. Reproducibility

Corpus, trainer, per-net analysis and the four grid preregistrations: https://github.com/jamesrahenry/Model_Zoo_Cartography (MIT, DOI 10.5281/zenodo.22017454). Chain-side analysis scripts, result files and the A1/A1b/I1–I3 preregistrations: `arc_chain/` in the same repository. Each preregistration was committed before any net it concerns was trained or any analysis run:

| file | commit | sha256 (prefix) | written (UTC) |
|---|---|---|---|
| grid/PREREG.md (grid 1, P1–P5) | b385b89 | 0a377f84625f | 2026-10-22 17:20 |
| grid/PREREG2.md (grid 2, Q1–Q5) | e239bf5 | 3d7b9bc1ca3f | 2026-10-23 16:55 |
| grid/PREREG3.md (grid 3, H1–H5) | 6f9e953 | fc7ea619b8f6 | 2026-10-24 14:59 |
| grid/PREREG4.md (grid 4, K1–K4) | 952cabe | 018e9b63e96d | 2026-10-24 21:25 |
| A1 / A1b | 0da83a9 / afc1652 | 32eba9f02567 / dad8e2344fae | 2026-10-23 17:09 / 17:30 |
| I1 / I2 / I3 | d7f23f6 / 89fc574 / 4adf130 | e95897da2613 / c49378981669 / 07b925c282bd | 2026-10-25 14:10 / 23:14, 2026-10-26 14:02 |

Grids were trained on a single H200; the chain runs are laptop CPU. Estimator release: `[LINK: estimator release]`.

## 9. Credits

Wu, Lecomte, Winer, Robinson, Hilton and Christiano (ARC) for the framework and reference code. 504aldo (18218) for the public V18/V29 line our port was checked against. megan_park (18282), dogus_ozel (18281) and AndreasHad04 (18292) for the fit-free survival law and its confirmations, which name the direction our trained-net error lives in. Artyom_Mazur (18249) for Proposition 1 and il.miscusi (18248) for the scale mode. LuDoe, megan_park and yeongjunyoo (18251 thread) for the κ4-λ cross-check. michael_anderson7 (18272) for crediting the Phase 1 cancellation account and for the noise-corrected oracle method. Susupro (18254) for the public write-up census. keenanpepper for the public corpus. AndreasHad04 for the flopscope billing reports. AIcrowd and mohanty for the rulings.

## 10. Use of LLMs

Claude Code (Anthropic; Claude Opus and Fable 5 models) wrote and ran the code and experiments and drafted this text under James Henry's direction. James set the research direction, chose the pivot from the leaderboard to the trained-weights question, rejected the accuracy-based extraction metric of I1/I2 (which led to I3), chose and approved submissions, and reviewed every claim. All preregistration files were written by the model and committed before the corresponding runs; the hashes above are the record.
