# Task rank is imprinted in the input layer: a controlled cartography of training signatures in deep MLPs

*Draft v0.2 — 2026-08-17 20:18 UTC (v0.1: 2026-08-15). James Henry. Numbers reference FINDINGS.md
(F1–F7) and the committed analysis JSONs; corpus at
`james-ra-henry/MZC-Corpus` (flips public with this paper).*

## Abstract (draft)

Where does training leave a signature in a deep network, relative to the
statistical skeleton the same architecture has at initialization? We train a
controlled population of ~1,570 bias-free ReLU MLPs at a fixed He-Gaussian
specification (widths 64–512, depths 8–64) on classification tasks whose
aggregate input distribution matches the analytic null premise `N(0, I)`
exactly, with class count, class separation, real-data family, optimizer, and
weight decay as recorded axes. Against matched analytic and empirical nulls we
find: (1) an exact input-rank law — the input weight matrix of every converged
net carries precisely C−1 significant dimensions, the rank of the class-mean
simplex, invariant to width, class separation, and dataset (240/240 nets,
per-net); (2) a two-sided terminal-rank law — training replaces the
architecture's depth-driven rank collapse with a task-code-driven terminal
rank, above or below the random fixed point as the task demands; (3)
coordinate-bound sharing — same-task twins share nothing in raw hidden-space
eigenbases at any depth yet carry the same depth code up to rotation
(Procrustes-recovered overlap 0.90+ vs 0.035 chance, graded by task overlap);
(4) a sharp class-count wall (logistic width ≈ 2 classes) whose relationship
to the mid-network code ceiling we pre-registered as proportional, falsified
at small width, and resolved as a compute frontier logarithmic in width
(C₅₀ ≈ 28→39 across widths 64→512) and budget; and (5) functional inertness
of the random weight bulk — accuracy moves < 1% while weight decay sweeps the
bulk over three orders of magnitude of scale, implying that spectral censuses
which score bulk presence as structure measure optimizer hygiene, not
learning. We release the corpus (weights + full training provenance), the
audited instruments, and the negative results.

## 1. Motivation

Two research lines meet here. Moment-propagation estimators for random-weight
networks [ARC white-box line] provide an *analytic null*: what a given
architecture looks like with zero learned structure, computable from
specification alone. Label-free structure mining in trained models [AMC line]
finds low-dimensional, depth-persistent activation geometry but cannot say
which parts are learned rather than architectural. The missing object is a
*population of trained networks with shared statistics* — matched to the
analytic null in everything except training. We built it.

## 2. The corpus

- Architecture: ARC Phase-1 spec — square bias-free ReLU stacks, He-Gaussian
  `N(0, 2/fan_in)` init, ReLU after every layer; a bias-free linear head
  outside the censused stack carries the readout (its necessity and its
  gradient side-effects are themselves catalogued: a frozen-column no-head
  variant provides an in-net guaranteed-null control).
- Tasks: Gaussian mixtures constructed so the aggregate input distribution is
  exactly `N(0, I)` (class count C ∈ {2..64} and separation are dials; Bayes
  accuracy computed exactly per task), plus whitened MNIST and Fashion-MNIST
  (784→d seeded projection + ZCA; real higher moments, matched second
  moments).
- Axes: width {64, 128, 256, 512}, depth {8..64}, budget {20k, 60k, 200k},
  weight decay {0..1.0}, learning rate arms, two readout modes, 16–32 seeds
  per configuration; ~1,570 nets, 69 families, every net with full provenance
  (task, hyperparameters, seeds, outcome label vs exact Bayes, trajectory).
- Infrastructure: batched-stack trainer (32 nets/process, init bit-identical
  to the sequential reference, 64 s/net on a laptop GPU); upload-verify-prune
  corpus lifecycle with HF as system of record.

## 3. Instruments (each audited; §7)

Weight census with two MP floors (fixed analytic 2/fan_in; robust median-MP
estimate) + bulk-regime flag · analytic state-trajectory (q, ābar, āsd, r̄bar
via the Hermite moment chain, zero forward passes; per-architecture null
bands) · activation census with matched init-anchored exceedance floor ·
directional consistency (ΔW₀ subspaces vs task simplex) · raw and
Procrustes-recovered eigenspace overlap (honest fit/test split) ·
population-refit state-keyed stabilizer (with edge indicator dims) ·
transport-corrected feature tracking · shuffled-entry nulls for wild models.

## 4. Results

### 4.1 The input-rank law (exact)
L0 significant dims = C−1 under the fixed analytic floor: exact per-net for
all 240 converged nets across C ∈ {2..32} and separations 1.5–6.0 (including
a task with Bayes accuracy 0.505); 8.25–9.4 on real data; confirmed
independently by the scaled floor at C ≥ 15. Partially-trained nets show the
count still accreting after accuracy plateaus — at every failure boundary we
probed (class-count wall, width and depth optimization failures) the input
layer keeps learning the task geometry (task-subspace alignment 0.38–0.89 of
converged levels at chance accuracy). *Learning proceeds at the input edge
even when the pipe fails.*

### 4.2 The terminal rank is task-driven, two-sided
Random nets' propagated rank decays to a depth-driven fixed point. Trained
nets' terminal rank tracks the task code instead: +3 to +4σ above the
architecture's null band for mid-C, *below* it for small-C/easy tasks and
MNIST, at-band for uncommitted partial learners; strongest in narrow nets
(+11σ at w=64), inverted at shallow depth (−7.5σ at d=8). One law, signed by
the task's rank demand relative to the architecture's fixed point.

### 4.3 Sharing is coordinate-bound; the code is one code up to rotation
Same-task twins: L0 input-coordinate subspaces overlap 0.86–0.93 (init
controls exactly at isotropic chance); raw hidden-space eigenbases at or near
k/d chance at every depth; Procrustes-recovered overlap 0.99 early / 0.90
deep, graded by task overlap (cross-task 0.67, untrained 0.38). Chain-level
feature tracking fails in plain MLPs under both raw and linearly-transported
matching — continuity of the code is recoverable only through data-fitted
rotations. Consequence for representational-convergence claims: convergence
is real, rotation-hidden, and task-graded; metrics must be rotation-invariant
or input-anchored.

### 4.4 The class-count wall: pre-registered, falsified, then resolved as a
### logarithmic compute frontier

Convergence fraction vs C at fixed budget is a sharp logistic transition at
every width (logistic width ~1.5–2.2 classes; seed variance ×10 at the
crossing; Bayes gap linear in C below it). We pre-registered
C₅₀(w) = 2.7 × ceiling(w) — proportionality between the wall and the mid-net
activation-code ceiling — and **falsified it**: the constant held at
w=128/256 (2.54/2.53) and failed at w=64 (ratio 3.5). The resolution, from
per-width lr-tuned sweeps, is that the wall is not capacity-proportional at
all but a **compute frontier, approximately logarithmic in both width and
budget**: C₅₀ = 27.8 / 32.3 / 36.2 / 38.8 at w = 64/128/256/512 (~+3.7
classes per width doubling), sliding ~+10 classes per tripling of steps
(w=512, C=48: 0% → 88% converged at 3× budget). The ceiling grows comparably
slowly in width (8.1/11.6/14.3), which is why the falsified proportionality
appeared to hold at two widths. Ten-times budget arms show the slide
*continuing*: crossings reach ≈ 61 (w=512) and ≈ 50 (w=256) at 200k steps —
tasks unreachable at 3× budget converge at 10× — with mildly diminishing
log-returns (+10 then +4 classes per budget tripling at w=256) hinting at,
but not demonstrating, saturation beyond the tested decade. Every "hard
boundary" we encountered at
fixed hyperparameters — a w=512 stall, a depth-48/64 trainability frontier —
dissolved under lr and budget scaling (d=64: from total stall to 14/16
converged), while in every failure family the input layer's task-subspace
alignment kept accreting at chance-level accuracy. Fixed-hyperparameter
trainability boundaries must not be read as architectural limits.

### 4.5 The random bulk is functionally inert
Weight decay sweeps the bulk from 1.0× to 0.000× of its He scale while
accuracy moves < 0.008. The census's fixed floor fails abruptly at ~0.6–1.2
nats of decay while the depletion itself is smooth. The bulk carries no
function; a census that scores its presence as structure measures optimizer
hygiene. In the depleted regime rank metrics, not MP counts, are the
structure measure (the robust estimator provably breaks there — documented,
not patched over).

### 4.6 Quantitative skeleton prediction needs population-fitted corrections
A companion random-ensemble measurement (published separately as
`analytic_vs_sampling/` in the ARC replication repo) grounds the baseline:
the uncorrected chain's error is *depth-flat once the state trajectory
saturates* (constant to <2% across d=24→48 at w=256, 48 nets/cell) and beats
FLOP-matched Monte Carlo at a crossover depth that rises monotonically with
width — the analytic skeleton is a stable object to correct against at any
depth. On trained weights the picture changes: the k=2 mean-field chain
predicts random-net activation spectra to 3–7%
through 32 layers but degrades to 80%+ by L16 on trained weights. A
state-keyed correction (ARC's machinery, 160 params with edge indicators)
refit on the trained population repairs the bulk 7–12× held-out across task
size, fixes the edges, and — fit on mixed converged+partial populations —
serves both regimes. A 16-parameter constant-coefficient rung captures only
~1.4× (no hidden dominant scalar; the state-keying is load-bearing).

### 4.7 First contact with wild models
On pythia-70m/160m MLP blocks (no init anchor; scaled floor + entry-shuffled
empirical nulls): per-matrix structure profiles of 3–111 significant dims
against 0–12 shuffled-null counts, with depth-dependent profiles; the models'
own reinitializations read 0–1 significant dims everywhere — spectrally
indistinguishable from the shuffle null, validating entry-shuffling as a
universal wild-model baseline requiring no knowledge of the init scheme. The
instrument-validity rules learned on the controlled corpus (orientation,
regime flags, null choice) transferred directly, catching one artifact via
the shuffle null on first use.

## 5. Discussion (to expand)
Weights carry the *where* (location, rank, persistence of structure); the
weight×input interaction sharpens the *what*; quantitative *how much* needs
population calibration. Implications for weight-only model auditing, for PRH
mechanistically, and for reading structure in AdamW-trained wild models.

## 6. Limitations
Single architecture family (plain ReLU stacks); GMM tasks are linearly
separable by construction (real-data families partially address this); the
Bayes proxy mislabels real-data outcomes; wall-vs-ceiling relationship
unresolved; wild-model reinit baselines not yet run.

## 7. Audit trail
An external adversarial audit of the first findings digest found two findings
wrong as stated (stale pilot numbers; a self-calibrating floor) and five
oversold precisions; all were rewritten from current data, the falsified
pre-registration is reported as falsified, and every instrument's failure
modes are documented in-repo. A recurring statistical lesson enforced
throughout: per-net error quantities in this regime are heavy-tailed (both
Monte-Carlo and analytic-closure errors; max/median 10–22× at depth), so
small-sample mean-based comparisons flip sign under reseeding — medians with
bootstrap intervals are the reporting standard, with per-net records
committed. The corpus, instruments, per-net provenance, and audit note ship
with the paper.
