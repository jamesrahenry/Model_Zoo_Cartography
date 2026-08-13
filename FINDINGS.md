# MZC Findings

> **EXTERNALLY AUDITED 2026-08-13 — F2 and F3 are currently WRONG, do not cite.** An adversarial
> audit (`notes/2026-08-13_external-audit-f1-f7.md`) found F2's headline numbers are a stale
> 6-net pilot never rechecked against the full 21-family dataset already in the cited file (which
> contradicts it), and F3's "exactly C" / "flat" claims fail outside a narrow C=10-15 band and
> contradict F6's own ceiling claim in this same document. F1, F4, F5 hold up under independent,
> from-scratch reproduction but need wording/scoping fixes (MNIST/C32/C50 edge cases; a 10-17x
> range that doesn't match its own cited example; "exact chance" that's actually near-chance).
> F6/F7's core claims survive re-derivation but two headline numbers (C₅₀/width; "exact on
> synthetics") are stated with more precision/certainty than the evidence supports. Read the audit
> note before relying on any specific number below.

*Written: 2026-08-13 16:45 UTC. Covers program start (2026-08-10) through Phase A
and its analysis. Corpus: 401 trained nets / 22 families on
`james-ra-henry/MZC-Corpus` (private HF dataset); every number below is
reproducible from the committed JSONs + corpus. Running record: Hopper task
`t4b9971d`. Phase B Wave 1 (width/depth/wall sweeps, 448 nets) launches
2026-08-13 21:00 UTC.*

MZC trains populations of MLPs at the ARC White-Box Challenge Phase-1
architecture (depth 32, width 256, He-Gaussian `N(0, 2/fan_in)`, bias-free,
ReLU after every layer) and asks where training leaves a signature relative to
the analytic random-init null. Readout is a bias-free linear head outside the
censused stack. Tasks: Gaussian mixtures whose *aggregate* input distribution
is exactly `N(0, I)` (so the null's premise holds by construction; class count
C and separation are dials), plus whitened MNIST.

## F1. The input-rank law: L0 significant dims = C−1, exactly

Under the analytic MP floor (σ² = 2/fan_in, uncentered), the input weight
matrix of every converged net carries **exactly C−1 significant dimensions** —
the rank of the centered class-mean simplex. Exact at 20-seed statistics for
C ∈ {2,3,5,8,10,15,20,25} (values 1/2/4/7/9/14/19/24), 30.9 at C=32 (19/20
converged), **invariant across class separation 1.5–6.0** (including a task
whose Bayes accuracy is 0.505), and 9.4 on whitened MNIST. Partially-trained
nets show partial counts that keep accreting after accuracy plateaus (C=50:
35.7 at 20k steps → 45.3 at 60k while accuracy moved 0.56→0.69).
*Instrument: `census/run_census.py`; data: `census/*_weight_census.json`.*

## F2. Rank-collapse arrest: the skeleton becomes task-driven

The four-number state trajectory (q = PR/w of the propagated pre-activation
covariance, etc.; ARC's state-keyed descriptors) computed analytically from
weights alone: random nets' q decays smoothly 0.50 → 0.016 over 32 layers;
trained nets crash to ~0.10 at L0 and **hold flat (~0.04 ≈ (C+1)/w) to L31**,
+5σ above the 50-net random band at depth. Training both accelerates the
collapse at the input edge and arrests it at task rank.
*Instrument: `null_baseline/state_trajectory.py` (+ vendored
`chain_state_keyed.py`); data: `null_baseline/state_trajectories.json`.*

## F3. Weights carry the *where*; input sharpens the *what*

Pure-noise inputs through trained weights already show the full activation
signature (eff dim ≈ C+1, flat L1–L31, vs init nets collapsing 158 → 3). Task
inputs — identical in mean AND covariance to the noise by construction —
tighten it ~10% and make the census count exactly C significant dims at
L0–L24. The weight × input interaction contributes sharpening; location,
rank, and persistence are weight-determined.
*Instrument: `census/run_activation_census.py`; data:
`census/activation_census.json`.*

## F4. Quantitative prediction needs population-fitted corrections

The k=2 mean-field vacuum holds 3–7% eigenvalue error on random nets through
all 32 layers but degrades to 80%+ by L16 on trained weights. The ARC
state-keyed stabilizer refit on our converged population (sequential DAgger,
128 params) repairs the bulk **10–17× held-out across task size** (L7/L15/L23:
0.67/0.84/0.88 → 0.050/0.066/0.122) but stays weak at the edges (L0/L1, L31)
and **fails on partial learners** — corrections are population- and
regime-specific, exactly as ARC's write-up §8 predicted. All polish iterations
rejected: the sequential fit structure is load-bearing (independent
reconfirmation).
*Instruments: vendored `analytic_vacuum.py`, `null_baseline/refit_trained.py`;
data: `null_baseline/refit_trained_results.json`.*

## F5. Sharing is coordinate-bound; the depth code is one code up to rotation

Same-task twins overlap strongly in *input* coordinates (top-(C−1) subspaces
of ΔW₀: 0.86–0.93 across seeds, task-subspace alignment higher still; init
controls exactly at isotropic chance) — but sit at **exact k/d chance in raw
hidden-space activation eigenbases at every depth**, matching P4's
cross-family LLM result even with identical task and identical inputs. Fitted
orthogonal Procrustes (honest fit/test split) recovers it: **0.99 early / 0.90
at depth for twins**, graded by task overlap (cross-task 0.67, init 0.38).
PRH statement: representational convergence is real, rotation-hidden, and
task-graded; metrics must be rotation-invariant or input-anchored.
*Instruments: `census/directional_consistency.py`,
`census/eigenspace_overlap.py`, `census/procrustes_overlap.py`; notes:
`notes/2026-08-13_mzc-eigenspace-overlap-reply.md`.*

## F6. The expressivity wall: sharp, with a fixed mid-net code ceiling

At fixed budget (20k steps), convergence fraction vs C is a sharp transition:
1.00 through C=25 → 0.95 (32) → 0.30 (40) → 0.00 (50); logistic **C₅₀ = 38.2,
width 2.0**; seed variance grows 10× at the crossing; below the wall the
Bayes gap grows linearly (∝ C^0.97). The mid-net activation code saturates at
**~14 effective dims (~21 significant) for every C ≥ 25**, converged or not
(a converged C=25 net routes 25 classes through a ~14-dim code). Tripling the
budget at C=50 moves accuracy 0.56→0.69 (asymptoting) while L0 structure keeps
accreting — the wall is mid-network expressivity, not input learning or step
count. **Registered prediction (2026-08-13): C₅₀(w) = 2.7 × ceiling(w)** —
under test in Phase B Wave 1.
*Data: `census/transition_curve.json`, `census/c_sweep_summary.json`; note:
`notes/2026-08-13_wall-model-and-separation-axis.md`.*

## F7. Scale vs shape: weight decay annihilates the bulk; rank ≠ amplitude

The analytic MP floor is scale-anchored: at wd ≥ 0.3 (≥1.8 nats of decay)
converged nets read **zero** significant dims because the unused weight bulk
has decayed *below* the floor — not rescaled but **annihilated** (bulk scale
1.0 → 0.03 → 0.000 for wd 0/0.3/1.0; every layer goes near-low-rank, mid-net
eff dim 120 → 26–35). The census now reports two floors (fixed analytic +
robust median-MP estimate, exact on synthetics across scales 0.02–2.0) plus a
`bulk_regime` flag; in the depleted regime rank metrics, not MP counts, are
the structure measure. Rule: fixed floor for matched corpora, scaled floor +
regime flag for wild-caught (AdamW-trained) models. Corollary via the
separation axis: L0 structural **rank** is task geometry (C−1 always); L0
structural **amplitude** (spike mass, bulk depletion) follows training
economics, peaking at intermediate difficulty.
*Instrument: `census/manifold_detector.py::estimate_mp_variance`; note:
`notes/2026-08-13_wall-model-and-separation-axis.md`.*

## Infrastructure (for reviewers who want to re-run)

- **Corpus**: `james-ra-henry/MZC-Corpus` (HF, private) is the system of
  record; `train/corpus_io.py` re-downloads pruned nets on demand. Full
  provenance JSON per net (task, hyperparameters, seeds, outcome, trajectory,
  git commit).
- **Training**: `train/train_mlp.py` (sequential), `train/train_mlp_batched.py`
  (32 nets/process, init bit-identical per seed, validated 0.8961±0.0020 vs
  sequential 0.8965±0.0022; 64 s/net on a laptop RTX 500).
- **Vendored** (see `PROVENANCE.md`): AMC's manifold detector (forked: explicit/
  estimated MP variance, optional centering), ARC's analytic vacuum + state-keyed
  chain.
- Depth-32 bias-free ReLU MLPs at ARC's spec train without skip/norm: plain
  Adam, lr 3e-4, 500-step warmup. 369 Phase-A nets, zero failures, 31.1 h.
