# MZC Findings

> **Revision history.** First written 2026-08-13 16:45 UTC. Externally audited same day
> (`notes/2026-08-13_external-audit-f1-f7.md`: F2/F3 wrong as stated, F1/F4/F5 solid with
> scoping fixes, F6/F7 real cores with oversold precision). **Revised 2026-08-13 against that
> audit**: F2 rewritten from the current 21-family data (the original was an unrechecked 6-net
> pilot; the corrected finding is two-sided and stronger), F3's quantitative claims withdrawn
> pending census recalibration (root cause: self-calibrating MP floor, the same trap the weight
> census already fixes), all other precision/scoping fixes applied inline. Known-open items are
> marked ⚠ in place.

*Covers program start (2026-08-10) through Phase A and its analysis. Corpus: 401
trained nets / 22 families on `james-ra-henry/MZC-Corpus` (private HF dataset).
Running record: Hopper task `t4b9971d`. Phase B Wave 1 (width/depth/wall sweeps)
running as of this revision.*

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
the rank of the centered class-mean simplex. Verified per-net, not in
aggregate: all 240 individual converged nets across the C-sweep
(C ∈ {2,3,5,8,10,15,20,25,32}) and the separation sweep hit the integer C−1
with zero exceptions (C=32 converged-only mean is exactly 31.0; the previously
reported 30.9 blended in one partial net). **Invariant across class separation
1.5–6.0**, including a task whose Bayes accuracy is 0.505.

Scoping caveats (audit items): whitened-MNIST nets read 9.4 but are all
formally labeled `partial` — their Bayes ceiling is a proxy (0.985) that this
architecture cannot reach, so the label is an artifact; the law appears to
hold there but is technically out of the stated converged-nets scope. The
C=50 accretion observation (35.7 at 20k → 45.3 at 60k steps while accuracy
moved 0.56→0.69) rests on 3 seeds at 60k and only two budget checkpoints —
directionally solid, "asymptoting" not yet established.
*Instrument: `census/run_census.py`; data: `census/*_weight_census.json`.*

## F2. The terminal propagated rank is task-determined — in both directions
*(rewritten 2026-08-13 from the full 21-family data after the audit found the
original text was an unrechecked 6-net pilot contradicted by its own cited file)*

The propagated rank clock q (PR/w of the analytically-propagated
pre-activation covariance, from weights alone, no inference): random nets
decay smoothly to a depth-driven fixed point (q(L31) ≈ 0.016). Trained nets'
terminal rank instead tracks the **task code** — and deviates from the random
band in *both directions*, graded by how the task's code rank compares to the
random terminal rank:

- mid-C families (C=10–25): z(L31) = **+3.2 to +4.2** above the 50-net band
  (arrest above the fixed point); q(L31)·w ≈ 9.5–11, consistent with the
  activation code size including F6's ceiling;
- small-C and easy tasks (C=2–5; separation ≥ 4.5): z = **−0.4 to −2.0**,
  i.e. *below* the random band — the trained code needs less rank than depth
  alone would leave (q(L31)·w ≈ 1.2–3.7 ≈ C−1);
- whitened MNIST: decays monotonically below the band (z = −1.8), no plateau
  — its learned code is tighter than any GMM family's;
- partial learners (C=40/50 at 20k): z ≈ +0.1 to +1.0 — near the band, the
  q-clock's known insensitivity to uncommitted structure.

Median z across all 21 families: +2.3 (14/21 above the band at z>1). Not
"flat": several families show a shallow U (e.g. C=25: 0.105 → 0.038 at L20 →
0.044 at L31). The earlier `q ≈ (C+1)/w` formula holds only near C ≈ 8–15.
The corrected statement: **training replaces the depth-driven terminal rank
with a task-code-driven one**, above or below the random fixed point as the
task demands.

*Wave-1 cross-architecture update (2026-08-14, per-architecture null bands):*
the two-sided law generalizes. Narrow nets arrest hardest (w=64: z = +5.6 to
+11.2 — their task code is large relative to their random terminal rank);
w=128: +5.4 to +7.2; shallow nets sit far below their band (d=8: −7.5 — the
8-layer random band never collapses far, and the trained code undercuts it);
optimization-failed families (w=512, d≥48) read at/below band, consistent
with the q-clock measuring committed structure only.
*Instrument: `null_baseline/state_trajectory.py` (+ vendored
`chain_state_keyed.py`); data: `null_baseline/state_trajectories.json`.*

## F3. Weights carry the *where*; input sharpens the *what*
*(rewritten 2026-08-14 from the recalibrated census — init-anchored
exceedance floor, regenerated over all 36 families, 8 nets each)*

**What stands (algebraic + within-net):** task inputs match pure-noise inputs
in mean AND covariance by construction, so the paired noise-vs-task comparison
is confound-free; trained-weight activations under pure noise already carry
the qualitative structure signature, and task input tightens it modestly.
Location and persistence of activation structure are weight-determined.

**The recalibrated quantitative picture has two depth regimes:**

- **Early layers** (init activations still diffuse → the matched exceedance
  null is clean): anchored significant dims at L1 read **C−1 to C** for
  C ≤ 20 (2.0/3.0/5.0/7.4/9.0/14.0/19.0), compress at the ceiling (20.0 at
  C=25, 19.6 at C=32), and collapse with the wall (13.8 at C=40). The
  early-activation rank law mirrors F1's weight law.
- **Mid/deep layers**: the exceedance count reads ≈ 0 everywhere — and that
  is itself the finding: **init activations at depth are MORE concentrated
  than trained ones** (random rank collapse), so trained structure at depth
  is not "spikier than init" but *less collapsed* than init — the
  activation-side view of F2's arrest. The right deep statistic is the
  trained/init effective-dim ratio, not any exceedance count.

Instrument lesson (audit follow-through): the self-calibrating floor's counts
(the retracted "exactly C at L0–L24") conflated these regimes; the anchored
floor is valid exactly where init is diffuse, and inverts where init has
collapsed. Both counts + the anchor are now recorded per layer.
*Instrument: `census/run_activation_census.py`
(`significant_dims_anchored`); data: `census/activation_census.json`.*

## F4. Quantitative prediction needs population-fitted corrections

The k=2 mean-field vacuum holds 3–7% eigenvalue error on random nets through
all 32 layers but degrades to 80%+ by L16 on trained weights. The ARC
state-keyed stabilizer refit on our converged population (sequential DAgger,
128 params) repairs the bulk **~4–15× held-out across task size**, peaking
mid-depth (L7/L15: 0.67/0.84 → 0.050/0.066 ≈ 13×; L23: 0.88 → 0.122 ≈ 7×),
actively hurts at L0/L1, and **fails on partial learners** (refit is worse
than uncorrected on the C=50 transfer set) — corrections are population- and
regime-specific, exactly as ARC's write-up §8 predicted. All polish iterations
were rejected on held-out data (a verified internal self-consistency check;
the audit independently reproduced the full fit + gate + eval bit-for-bit).
Reproducibility (closed 2026-08-14): regenerated on an explicit, recorded
population (`--max-per-run 8`: 24 fit / 16 val / 8 transfer) — the bulk
repair reproduces (L7/L15/L23: 0.68/0.85/0.89 → 0.058/0.086/0.123, ≈ 7–12×)
and the edge failures reproduce (L0 worse than uncorrected). The original
3-seed-era numbers stand as a smaller-population instance of the same result.
*Instruments: vendored `analytic_vacuum.py`, `null_baseline/refit_trained.py`;
data: `null_baseline/refit_trained_results.json`.*

## F5. Sharing is coordinate-bound; the depth code is one code up to rotation

Same-task twins overlap strongly in *input* coordinates (top-(C−1) subspaces
of ΔW₀: 0.86–0.93 across seeds, task-subspace alignment higher still; init
controls at isotropic chance) — but sit **at or very near k/d chance in raw
hidden-space activation eigenbases at every depth** (exact at shallow layers;
a small consistent elevation of ~25% above chance at the deepest layers, which
init controls share), matching P4's cross-family LLM result even with
identical task and identical inputs. Fitted
orthogonal Procrustes (honest fit/test split) recovers it: **0.99 early / 0.90
at depth for twins**, graded by task overlap (cross-task 0.67, init 0.38).
PRH statement: representational convergence is real, rotation-hidden, and
task-graded; metrics must be rotation-invariant or input-anchored.
*Instruments: `census/directional_consistency.py`,
`census/eigenspace_overlap.py`, `census/procrustes_overlap.py`; notes:
`notes/2026-08-13_mzc-eigenspace-overlap-reply.md`.*

## F6. The expressivity wall: sharp, with a fixed mid-net code ceiling

At fixed budget (20k steps), convergence fraction vs C is a sharp transition:
1.00 through C=25 → 0.95 (32) → 0.30 (40) → 0.00 (50); seed variance grows
10× at the crossing; below the wall the Bayes gap grows linearly (∝ C^0.97,
robust to point-set choice per the audit's re-fits, 0.97–1.08). ⚠ The logistic
point estimate **C₅₀ ≈ 38, width ≈ 2** rests on only two informative binomial
points (C=32, 40 — everything else is saturated); Wilson-propagated intervals
are C₅₀ ∈ [35, 41] and width ∈ [1.2, 7.3], so "sharp" spans to "fairly
gradual" until B3's C=36/44 fill-ins land (running). The mid-net activation
code approaches a **~14-effective-dim (~21 significant) ceiling gradually**
(C=15/20 already at 86–97% of it; flat within 7% for C = 25–50), converged or
not — a converged C=25 net routes 25 classes through a ~14-dim code. Tripling
the budget at C=50 moves accuracy 0.56→0.69 while L0 structure keeps accreting
— the wall is mid-network expressivity, not input learning or step count.
**Registered prediction verdict (2026-08-14, Wave 1 complete).** With the
fill-ins the wall fit is **C₅₀ = 36.2 [35.2, 37.3], width 2.1 [1.5, 2.6]**
(committed fit + bootstrap; "sharp" survives the audit's CI concern). The
prediction C₅₀(w) = k × ceiling(w): at the two cleanly-trained widths the
measured ratios are **2.54 (w=128: C₅₀ ≈ 29.5 / ceiling 11.6)** and **2.53
(w=256: 36.2 / 14.3)** — the law holds with the constant revised 2.7 → ~2.5
(the 2.7 came from the pre-fill-in C₅₀). Ceiling scales sublinearly,
≈ w^0.5 (7.4 / 11.6 / 14.3 at w = 64/128/256). Standing falsifiable
prediction: **C₅₀(64) ≈ 2.5 × 7.4 ≈ 18.5** (its C=16 probe converged 32/32,
consistent; the crossing probe is queued). Two boundaries the law does NOT
cover: w=512 is optimization-confounded at fixed lr (bimodal at C=32 sliding
to total stall at C=64 — a *different failure mode* than the soft wall; needs
per-width lr scaling before its wall is measurable), and depth has its own
trainability frontier at fixed lr/budget (converges d ≤ 32, partial d=48,
stalled d=64 — at C=10, where d=32 is trivial). In every failure family the
L0 task subspace still accretes (alignment 0.38–0.89 of converged levels
while accuracy sits at chance): **learning proceeds at the input edge even
when the pipe fails.**
*Data: `census/transition_curve.json`, `census/c_sweep_summary.json`; note:
`notes/2026-08-13_wall-model-and-separation-axis.md`.*

## F7. Scale vs shape: weight decay annihilates the bulk; rank ≠ amplitude

The analytic MP floor is scale-anchored: at wd ≥ 0.3 (≥1.8 nats of decay)
converged nets read **zero** significant dims because the unused weight bulk
has decayed *below* the floor — not rescaled but **annihilated** (bulk mass
fraction 86% → 49% → 0.13% across wd 0/0.3/1.0 while the structural spikes
drop nowhere near proportionally; every layer goes near-low-rank, mid-net eff
dim 120 → 26–35; 640/640 layer-measurements read zero, audit-verified). The
census reports two floors (fixed analytic + robust median-MP estimate) plus a
`bulk_regime` flag. ⚠ The scaled estimator's validation is **1.00–1.11
(up to 11% error) on synthetics at scales 0.02–2.0** — previously overstated
as "exact" — and that validated range does not reach the annihilated regime
this finding is about: on real wd=1 matrices (bulk not a clean rescaled
Gaussian; near-zero dead rows) it is empirically broken, reporting 29–82
"significant" dims for true structural rank ~7–9. F7's conclusion routes
around it (in the depleted regime rank metrics, not MP counts, are the
structure measure — that is the rule), but the estimator must not be treated
as validated infrastructure in its motivating regime. Rule: fixed floor for
matched corpora; scaled floor + regime flag for wild-caught (AdamW-trained)
models *while the bulk is intact*; rank metrics once `bulk_regime` reads
depleted. Corollary via the
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
