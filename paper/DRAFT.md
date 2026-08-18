# Task rank is imprinted in the input layer: a controlled cartography of training signatures in deep MLPs

*Draft v0.3 — 2026-08-18 02:34 UTC (v0.2: 2026-08-17; v0.1: 2026-08-15). James Henry.
Numbers reference FINDINGS.md (F1–F7) and the committed analysis JSONs; corpus at
`james-ra-henry/MZC-Corpus` (flips public with this paper). v0.3: question stated in §1,
§5 written in full, per-section meaning leads + term definitions in §4, F3 (noise-input
result) added as §4.3 — it carried §5's thesis but had no section; §4.3–4.7 renumbered
to §4.4–4.8.*

## Abstract (draft)

Where does training leave a signature in a deep network, relative to the
statistical skeleton the same architecture has at initialization? We train
~1,570 bias-free ReLU MLPs at a fixed He-Gaussian specification on
classification tasks whose aggregate input distribution matches the analytic
null premise `N(0, I)` exactly, and census the population against matched
analytic and empirical nulls. The headline is an exact law: the input weight
matrix of every converged net carries precisely C−1 significant dimensions —
the rank of the class-mean simplex — invariant to width, class separation,
and dataset, verified per-net with zero exceptions. Deeper in the network the
learned code is real but rotation-hidden: same-task twins share no
eigenvectors in raw coordinates at any depth yet carry the same code up to a
fitted rotation. And the random weight bulk that censuses often score as
structure is functionally inert: accuracy moves under 1% while weight decay
sweeps the bulk across three orders of magnitude of scale. We map where these
signatures break — a class-count frontier logarithmic in width and compute,
whose capacity-proportional form we pre-registered and falsified — and
release the corpus with full per-net training provenance, the audited
instruments, and the negative results.

## 1. Motivation

The question this paper answers: **can structure measured from weights alone —
before the network is ever run — predict where learned representation will
appear once inference starts?** Concretely: given a trained net and the
statistical skeleton its architecture has at initialization, which properties
of the activation geometry (location, rank, persistence, identity) are already
fixed by the weights, and which require input to materialize?

Two research lines meet here, and neither can answer this alone.
Moment-propagation estimators for random-weight networks provide an *analytic
null*: what a given architecture looks like with zero learned structure,
computable from the specification, no forward passes. The concrete instance is
the ARC White-Box Estimation Challenge [Wu et al., arXiv 2605.05179], which
scores exactly this skill on random MLPs under a FLOP budget — a tractable
proxy for the Alignment Research Center's larger estimation agenda: analytic
prediction of network behavior in regimes where sampling is uninformative
(rare and tail behaviors, anomaly detection). That agenda ultimately concerns
*trained* models, and the challenge write-up's "toward non-random networks"
section names the missing object — a population of trained networks with
shared statistics — and defers it as future work. Meanwhile, label-free
structure mining in trained models finds low-dimensional, depth-persistent
activation geometry, but with no matched untrained population to diff against
it cannot say which parts are learned rather than architectural.

We built the missing object: ~1,570 trained nets at the challenge's exact
Phase-1 architecture, matched to the analytic null in everything except
training. The answer, developed instrument-by-instrument in §4 and assembled
in §5, has three parts: weights alone carry the **where** (location, rank,
and persistence of structure are weight-determined); the weight×input
interaction sharpens the **what** (the code's identity is real but
rotation-hidden); and quantitative **how much** requires population-fitted
corrections.

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
*Meaning: the task's class geometry is readable off the input weight matrix
as an integer — per net, zero forward passes.* Terms: "significant dims"
counts eigenvalues of a weight matrix's uncentered Gram above the
Marchenko–Pastur (MP) bulk edge at the He init variance 2/fan_in — the
analytic bound on how large a purely random eigenvalue can be at this shape;
anything above it is structure the null cannot produce.

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
*Meaning: depth does not erase the input imprint — the rank the code carries
at the last layer is set by the task, not by depth — and it is computed from
weights alone by the same Hermite moment chain the ARC challenge scores on
random nets.* Terms: the "q-clock" is q(L) = PR/w, the participation ratio
(effective dimension) of the analytically propagated pre-activation
covariance at layer L, normalized by width; "null band" is the per-architecture
q(L) distribution over freshly initialized nets.

Random nets' propagated rank decays to a depth-driven fixed point. Trained
nets' terminal rank tracks the task code instead: +3 to +4σ above the
architecture's null band for mid-C, *below* it for small-C/easy tasks and
MNIST, at-band for uncommitted partial learners; strongest in narrow nets
(+11σ at w=64), inverted at shallow depth (−7.5σ at d=8). One law, signed by
the task's rank demand relative to the architecture's fixed point.

### 4.3 Noise-driven activations carry the structure; task input sharpens it
*Meaning: the activation structure is in the weights before any task-relevant
input arrives — inference reveals it rather than creates it.* The GMM
construction makes this comparison confound-free: task input and pure
`N(0, I)` noise match in mean and covariance exactly, so any activation
difference is higher-moment.

Trained-weight activations under pure noise already carry the qualitative
structure signature; task input tightens it modestly. Under the init-anchored
exceedance floor, the early-activation rank law mirrors the weight law:
L1 anchored significant dims read C−1 to C for C ≤ 20
(2.0/3.0/5.0/7.4/9.0/14.0/19.0), compress at the code ceiling (20.0 at C=25,
19.6 at C=32), and collapse with the wall (13.8 at C=40). At mid/deep layers
the exceedance count reads ≈ 0 everywhere — and that is itself the finding:
init activations at depth are *more* rank-collapsed than trained ones, so
trained structure at depth is not "spikier than init" but *less collapsed*
than init — the activation-side view of §4.2's arrest. The valid deep
statistic is the trained/init effective-dim ratio, not any exceedance count
(the anchored floor is clean exactly where init is diffuse and inverts where
init has collapsed; both counts plus the anchor are recorded per layer).

### 4.4 Sharing is coordinate-bound; the code is one code up to rotation
*Meaning: two nets that learned the same task carry the same code in
different coordinates — any comparison across models that is not
rotation-invariant or input-anchored will read false negatives at depth.*

Same-task twins: L0 input-coordinate subspaces overlap 0.86–0.93 (init
controls exactly at isotropic chance); raw hidden-space eigenbases at or near
k/d chance at every depth; Procrustes-recovered overlap 0.99 early / 0.90
deep, graded by task overlap (cross-task 0.67, untrained 0.38). Chain-level
feature tracking fails in plain MLPs under both raw and linearly-transported
matching — continuity of the code is recoverable only through data-fitted
rotations. Consequence for representational-convergence claims: convergence
is real, rotation-hidden, and task-graded; metrics must be rotation-invariant
or input-anchored.

### 4.5 The class-count wall: pre-registered, falsified, then resolved as a logarithmic compute frontier
*Meaning: the "wall" — the class count at which training stops converging at
fixed budget — is a compute frontier, not an architectural capacity limit;
our pre-registered capacity-proportional story about it was wrong and we say
so.* Terms: C₅₀ is the class count at which half the seeds converge; the
"ceiling" is the mid-net activation code's saturating effective dimension
(~14 at w=256 for every C ≥ 25, converged or not).

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

### 4.6 The random bulk is functionally inert
*Meaning: most of a trained net's weight mass is functionally inert init
residue — a census that scores its presence as structure measures optimizer
hygiene, not learning.*

Weight decay sweeps the bulk from 1.0× to 0.000× of its He scale while
accuracy moves < 0.008. The census's fixed floor fails abruptly at ~0.6–1.2
nats of decay while the depletion itself is smooth. The bulk carries no
function; a census that scores its presence as structure measures optimizer
hygiene. In the depleted regime rank metrics, not MP counts, are the
structure measure (the robust estimator provably breaks there — documented,
not patched over).

### 4.7 Quantitative skeleton prediction needs population-fitted corrections
*Meaning: the ARC challenge's analytic machinery survives contact with
trained networks as a correctable skeleton, but the correction is a
population object — exactly the obstruction the challenge write-up's "toward
non-random networks" section predicted, now measured. This is the section the
estimation agenda inherits: any program that wants analytic estimates on
real (trained) models gets this shape — analytic null + population-fitted
correction, with the correction's validity bounded by the training regime it
was fit on.*

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

### 4.8 First contact with wild models
*Meaning: the instrument-validity rules learned on the controlled corpus are
not corpus-specific — they transferred unchanged to models trained elsewhere
with unknown recipes, and caught an artifact on first use.*

On pythia-70m/160m MLP blocks (no init anchor; scaled floor + entry-shuffled
empirical nulls): per-matrix structure profiles of 3–111 significant dims
against 0–12 shuffled-null counts, with depth-dependent profiles; the models'
own reinitializations read 0–1 significant dims everywhere — spectrally
indistinguishable from the shuffle null, validating entry-shuffling as a
universal wild-model baseline requiring no knowledge of the init scheme. The
instrument-validity rules learned on the controlled corpus (orientation,
regime flags, null choice) transferred directly, catching one artifact via
the shuffle null on first use.

## 5. Discussion

### 5.1 The answer, assembled

Can pre-inference structure predict where learned representation will appear
once inference starts? The corpus answer has three parts, each carried by
independent instruments:

**Weights carry the *where*.** Location, rank, and persistence of activation
structure are weight-determined. The input layer holds the task's class
geometry as an exact integer (§4.1); analytic propagation of the weights —
zero forward passes — predicts the terminal rank of the activation code
(§4.2); and trained weights driven by pure noise already produce the full
qualitative structure signature, which task input tightens but does not
create (§4.3). To first order, a trained net's activation geometry is a
property of its weights that inference reveals rather than creates.

**The weight×input interaction sharpens the *what* — and hides it in
coordinates.** The identity of the learned code is real and shared across
same-task nets, but only up to rotation (§4.4): raw hidden-space eigenbases
overlap at chance at every depth while Procrustes-recovered overlap reads
0.90–0.99, graded by task similarity. So *where* structure lives is
predictable from weights; *which directions* carry it is not even well-posed
without an anchor — the input coordinates, or a fitted rotation.

**Quantitative *how much* needs population calibration.** The analytic chain
predicts random-net spectra to a few percent through 32 layers but degrades
to 80%+ on trained weights; a 160-parameter state-keyed correction refit on
the trained population repairs the bulk ~7–12× held-out — and fails on
populations from other training regimes (§4.7). There is no free
trained-network estimator; there is a cheap, audited recipe for building one
per population.

### 5.2 For analytic estimation on real models

The ARC White-Box Estimation Challenge scores analytic prediction on random
nets as a tractable proxy for the estimation problems that matter in
reality — rare and tail behaviors, anomaly detection, properties of models in
regimes where sampling cannot reach. Those problems live on trained models.
This corpus is the controlled step between the two, and it returns the
estimation agenda concrete news:

- **The random-weight chain's best trained-network role is the null, not the
  predictor.** Every training signature in this paper is a measured deviation
  from the analytic skeleton (§4.1, §4.2, §4.6), computable from a weights
  file at specification cost. That is an anomaly-detection primitive: does
  this net deviate from its architecture's null the way its claimed training
  says it should?
- **Corrections are buildable, cheap, and population-bound** (§4.7): ~10²
  parameters and a modest matched population suffice, and the failure mode on
  out-of-regime populations (partial learners) is detectable by held-out
  gating, not silent.
- **Regime flags are mandatory instrumentation.** Every instrument we fielded
  has a validity boundary that training economics can cross: the weight-decay
  knee (§4.6), the anchored floor's inversion at depth (§4.3), the scaled
  estimator's break in the depleted regime. The transferable discipline is
  the paired-null habit — fixed analytic null where the init is known,
  entry-shuffled empirical null where it is not (§4.8) — plus a per-matrix
  flag saying which regime the reading came from.

### 5.3 For weight-space auditing and representational convergence

For weight-only model auditing: task-relevant rank is readable from weights
alone, but amplitude is not learning (§4.6) — an auditor that scores spectral
mass measures optimizer hygiene, and the structure measure must switch from
MP counts to rank metrics once the bulk is depleted. For the
representational-convergence literature: convergence across same-task models
is real, task-graded, and strictly rotation-hidden (§4.4) — cross-model
comparisons made in raw coordinates at depth measure nothing, and metrics
must be rotation-invariant or input-anchored to see the shared code at all.

## 6. Limitations
Single architecture family (plain ReLU stacks); GMM tasks are linearly
separable by construction (real-data families partially address this); the
Bayes proxy mislabels real-data outcomes; the wall is characterized as a
compute frontier (§4.5) but its mechanism — and its relation to the code
ceiling — is unresolved; correction scope across optimizers and objectives is
unmeasured (AdamW wild models are first-contact only, §4.8). The next
falsifiable step mirrors this paper's method: pre-register what the census
should read on a wild model given its claimed training, then check.

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
