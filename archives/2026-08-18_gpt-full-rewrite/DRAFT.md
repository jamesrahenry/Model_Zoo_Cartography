# Input-layer task rank in controlled deep MLP populations

## Abstract

Where does training leave a signature in a deep network, relative to the
statistical skeleton the same architecture has at initialization? We train
1,569 bias-free ReLU MLPs with He-Gaussian initialization on classification
tasks whose aggregate input mean and covariance equal those of the analytic
null, and compare the resulting population with analytic and empirical nulls.
Within the evaluated converged Gaussian-mixture runs (C <= 32), the input
weight matrix has C−1 eigenvalues above the specified Marchenko--Pastur edge,
matching the rank of the centered class-mean simplex. In deeper layers, raw
activation-eigenspace overlap for same-task networks is near its isotropic
baseline, whereas fit/test Procrustes alignment is high. In a weight-decay
sweep, validation accuracy changes by less than 0.008 while the scale of the
weight bulk changes by three orders of magnitude. We also characterize a
class-count convergence transition whose location increases with width and
training budget over the evaluated range. The release will include per-network
training provenance, analysis code, and the full corpus.

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
the ARC White-Box Estimation Challenge [ARC Challenge 2026; backbone paper Wu
et al. 2026], which scores exactly this skill on random MLPs under a FLOP budget — a tractable
proxy for the Alignment Research Center's larger estimation agenda: analytic
prediction of network behavior in regimes where sampling is uninformative
(rare and tail behaviors, anomaly detection). That agenda ultimately concerns
*trained* models, and the challenge write-up's "toward non-random networks"
section names the missing object — a population of trained networks with
shared statistics — and defers it as future work. Meanwhile, label-free
structure mining in trained models finds low-dimensional, depth-persistent
activation geometry, but with no matched untrained population to diff against
it cannot say which parts are learned rather than architectural.

We construct a population of 1,569 trained networks based on the challenge's
Phase-1 architecture. The GMM inputs match the null's mean and covariance but,
as mixtures, not its full Gaussian distribution. We evaluate three questions:
whether weight-only measurements identify the location and effective dimension
of activation structure; how those measurements change when inputs are applied;
and whether a population-specific correction improves analytic predictions.

## 2. The corpus

- Architecture: ARC Phase-1 spec — square bias-free ReLU stacks, He-Gaussian
  `N(0, 2/fan_in)` init, ReLU after every layer; a bias-free linear head
  outside the censused stack carries the readout (its necessity and its
  gradient side-effects are themselves catalogued: a frozen-column no-head
  variant provides an in-net guaranteed-null control).
- Tasks: Gaussian mixtures whitened to have aggregate mean zero and covariance
  I (their higher moments remain non-Gaussian; class count C ∈ {2..64} and
  separation vary), plus whitened MNIST and Fashion-MNIST (784→d seeded
  projection + ZCA; matched second moments but different higher moments).
- Axes: width {64, 128, 256, 512}, depth {8..64}, budget {20k, 60k, 200k},
  weight decay {0..1.0}, learning rate arms, two readout modes, 16–32 seeds
  per configuration; 1,569 nets, 69 families, every net with full provenance
  (task, hyperparameters, seeds, outcome label vs exact Bayes, trajectory).
- Infrastructure: batched-stack trainer (32 nets/process, init bit-identical
  to the sequential reference, 64 s/net on a laptop GPU); upload-verify-prune
  corpus lifecycle with HF as system of record.

## 3. Instruments

Eight instruments, each audited (§8), each running from committed code
(`census/`, `null_baseline/`; vendored provenance in `PROVENANCE.md`). The
first two read weights only — zero forward passes; the next three read
activations on controlled inputs; the last three are correction, tracking,
and wild-model machinery.

**3.1 Weight census** (`census/manifold_detector.py`, vendored from the AMC
line). Per-layer eigenspectrum census of the weight matrices themselves.
For a layer's uncentered second-moment spectrum {λᵢ}, the Marchenko–Pastur
bulk edge is λ₊ = σ²(1 + √γ)², γ the matrix aspect ratio [Marchenko & Pastur
1967]; **significant dims** = #{λᵢ > λ₊}. Two floors are always reported:
the *fixed analytic* floor σ² = 2/fan_in, exact because the He init scheme is
known by construction (the matched-corpus instrument, §4.1); and a *robust
estimate* σ̂² = median(λ)/m(γ), m the MP median factor, for models whose init
is unknown (§5.5). Alongside: effective dimension via participation ratio
PR = (Σλᵢ)²/Σλᵢ², and a `bulk_regime` flag (intact/depleted) that gates which
reading is meaningful — under strong weight decay the bulk falls *below* any
floor and MP counting is undefined (§4.6).

**3.2 Analytic state trajectory — the q-clock**
(`null_baseline/state_trajectory.py` + the vendored Hermite moment chain
[ARC Challenge 2026; Wu et al. 2026]). Propagates (μ, Σ) = (0, I) through
the trained weights with the uncorrected k=2 Hermite chain — an analytic
computation on the weights, no data. Per-layer descriptors: **q =
PR(Σ_pre)/w** (the rank-collapse clock: 1 isotropic → 0 collapsed), ā/āsd
(mean/std of the per-neuron ReLU operating point μ/σ), r̄ (mean |ρ|
off-diagonal correlation). The null is a *band*: the same descriptors over
50 fresh He inits per architecture; trained readings are z-scores against
that band (§4.2).

**3.3 Activation census with init-anchored floor**
(`census/run_activation_census.py`). The eigenspectrum census on post-ReLU
activations, under matched task and pure-noise inputs (identical mean and
covariance by construction, §4.3). The floor is *anchored to init*: per
layer, the init net's maximum activation-eigenvalue fraction is the
threshold, and `significant_dims_anchored` counts trained eigenvalue
fractions exceeding it — a matched empirical exceedance null. Validity is
regime-bound and recorded per layer: clean where init activations are
diffuse (early layers), inverted where init has rank-collapsed below trained
(depth) — there the trained/init effective-dim ratio is the statistic (§4.3).

**3.4 Directional consistency** (`census/directional_consistency.py`). Does
training move the input layer in the same direction across seeds, and toward
the task? Uᵢ = top-(C−1) left singular vectors of ΔW₀ = W₀ᵗʳᵃⁱⁿᵉᵈ − W₀ⁱⁿⁱᵗ;
subspace overlap(A, B) = ‖AᵀB‖²_F / k — mean cos² of the principal angles —
against the exactly-known task subspace T = orthonormal basis of the centered
class means (a GMM-construction privilege), and pairwise across seeds.
Isotropic chance is k/d (§4.1, §4.5).

**3.5 Eigenspace overlap, raw and Procrustes-recovered**
(`census/eigenspace_overlap.py`, `census/procrustes_overlap.py`). Per layer:
top-k eigenvectors of each net's activation covariance on a shared input
sample; pairwise overlap ‖UᵢᵀUⱼ‖²_F / k against k/d chance — the *raw*
(coordinate-bound) reading. The *recovered* variant splits the shared sample
into fit/test halves, fits the orthogonal Procrustes rotation R = UVᵀ from
SVD(A_fitᵀB_fit) [Schönemann 1966] on the fit half only, and measures overlap
between test-half eigenbases after rotation (§4.4).

**3.6 Population-refit state-keyed stabilizer**
(`null_baseline/refit_trained.py` + vendored `chain_state_keyed.py` [ARC
Challenge 2026]). Repairs §3.2's chain on trained weights. The chain's 16
correction slots become functions of measured state rather than layer index:
c_j = θ_j · g(state), the basis g built from (q, ā, āsd, r̄) — 128
parameters; two indicator dims [is_first, is_last] extend it to 160 and fix
the edges. θ is fit by DAgger-style iteration [Ross et al. 2011]: roll
trajectories with the current θ, regress residual-to-truth at every layer,
repeat. Candidate polish iterations are selected using a validation population;
the C=50 population is reserved for a transfer evaluation (§4.7).

**3.7 Transport-corrected feature tracking** (`census/feature_tracker.py`,
vendored from the AMC line). Matches activation eigenfeatures layer-to-layer,
raw and after linear transport through the layer map. In this corpus it is
chiefly a calibrated negative result: chain-level feature continuity in plain
MLPs is not recoverable under raw or transported matching — only through
data-fitted rotations (§4.4) — which bounds what tracking can claim
elsewhere.

**3.8 Entry-shuffled empirical null for wild models**
(`census/wild_census.py`). For each wild weight matrix, an entry-shuffled
copy — identical marginal entry distribution and scale, structure destroyed —
is censused as that matrix's own null, requiring no knowledge of the init
scheme. In the preliminary Pythia analysis, model reinitializations read 0–1
significant dimensions and were close to the shuffled reference (§5.5).

## 4. Results

### 4.1 Input-layer C−1 relationship
"Significant dims"
counts eigenvalues of a weight matrix's uncentered Gram above the
Marchenko–Pastur (MP) bulk edge at the He init variance 2/fan_in — the
specified analytic-null threshold used by this instrument.

For all 240 converged GMM runs in the C ∈ {2..32} class-count and separation
sweeps, L0 significant dims equaled C−1 under the fixed analytic floor. This
includes a task with estimated Bayes accuracy 0.505 and separations 1.5–6.0.
The corresponding means were 9.4 for MNIST and 8.25 for Fashion-MNIST; these
real-data observations are approximate rather than integer identities.
Partially trained networks retained nonzero task-subspace alignment (0.38–0.89
of converged-family levels) even when validation accuracy remained near chance.

![Fig 1 — input-layer C−1 relationship](figures/fig1_input_rank_law.png)
*Figure 1. Input-layer significant dimensions under the specified analytic MP
floor. Left: in the evaluated converged GMM runs, L0 counts equal C−1; partial
and stalled runs scatter below the line. Right: the C=10 GMM result is stable
over separations 1.5–6.0. MNIST and Fashion-MNIST means are 9.4 and 8.25.*

### 4.2 Terminal effective dimension varies with task
The normalized covariance participation ratio is q(L) = PR/w, the participation ratio
(effective dimension) of the analytically propagated pre-activation
covariance at layer L, normalized by width; "null band" is the per-architecture
q(L) distribution over freshly initialized nets.

Random networks' propagated effective dimension decays toward a depth-dependent
fixed point. Relative to the 50-network initialization reference, trained
families lie above that reference for mid-range C, below it for small-C/easy
tasks and MNIST, and near it for partially trained families. The quoted
standardized differences are descriptive comparisons to the initialization
reference, not confidence intervals for the trained-family medians.

![Fig 2 — normalized covariance participation ratio](figures/fig2_qclock.png)
*Figure 2. Family-median q trajectories (w=256, d=32) against a 50-network
initialization reference band. Mid-range C families lie above the terminal
reference, whereas C=2, MNIST, and the partially trained C=50 family lie at or
below it. Computed from weights under the assumed input model.*

### 4.3 Activations under noise and task input
The GMM construction matches task input and pure `N(0, I)` noise in mean and
covariance. They differ in higher moments, so this comparison isolates neither
all task-related information nor causality, but it tests the contribution of
the fixed weights under a matched-second-moment input control.

Under the init-anchored exceedance floor, trained-weight activations under pure
noise show a similar early-layer pattern to task input. In early layers:
L1 anchored significant dims read C−1 to C for C ≤ 20
(2.0/3.0/5.0/7.4/9.0/14.0/19.0). Values at 20 are right-censored because the
implementation stores only the top 20 eigenvalues, so the 20.0 reading at
C=25 cannot establish a ceiling. At mid/deep layers
the exceedance count reads ≈ 0 everywhere — and that is itself the finding:
init activations at depth are *more* rank-collapsed than trained ones, so
trained structure at depth is not "spikier than init" but *less collapsed*
than init — the activation-side view of §4.2's arrest. The valid deep
statistic is the trained/init effective-dim ratio, not any exceedance count
(the anchored floor is clean exactly where init is diffuse and inverts where
init has collapsed; both counts plus the anchor are recorded per layer).

![Fig 3 — activation measurements under matched-second-moment inputs](figures/fig3_weights_carry_where.png)
*Figure 3. Left: L1 anchored significant dims vs C under pure-noise (open)
and task (filled) input. Right: activation effective dimension versus depth
for the same weights before and after training under noise input. At depth,
initialization activations have lower effective dimension than trained
activations.*

### 4.4 Cross-network alignment depends on the comparison basis

For same-task twins, L0 input-coordinate subspaces overlap 0.86–0.93, whereas
initialization controls are near the isotropic baseline. Raw hidden-space
eigenbasis overlap is near that baseline, with a small elevation at depth.
In the eight-network C=10 experiment, a Procrustes map fitted on one half of a
shared task-input sample gives 0.99 early-layer and about 0.90 deep-layer
overlap on the other half. The cross-task comparison uses noise input and is
therefore not a matched test of task dependence. Feature tracking did not
recover chain-level continuity under either raw or linearly transported
matching in these MLPs.

![Fig 4 — raw and Procrustes-aligned eigenspaces](figures/fig4_rotation_hidden.png)
*Figure 4. Left: raw activation-eigenspace overlap for same-task twins and
initialization controls. Right: a Procrustes map fitted on one half of the
task-input sample is evaluated on the other half; same-task twins reach 0.99
in early layers and about 0.90 at depth. The cross-task series uses noise input
and is not directly comparable as a task-dependence control.*

### 4.5 Class-count convergence transition
C₅₀ denotes the class count at which half of the evaluated seeds converge.

At a fixed budget, convergence fraction decreases over a narrow C range in the
evaluated sweeps. The preregistered model C₅₀(w) = 2.7 times a mid-layer
effective-dimension summary was not supported at w=64 (ratio 3.5 versus
2.54 and 2.53 at w=128 and w=256). In learning-rate-tuned sweeps at 20k steps,
estimated C₅₀ values were 27.8, 32.3, 36.2, and 38.8 for widths 64, 128, 256,
and 512. The 200k-step arms gave approximately 50 at w=256 and 61 at w=512.
These sparse measurements are consistent with sublinear growth in width and
budget over the tested range; they do not identify a logarithmic functional
form or establish an asymptote. Fixed-hyperparameter stalls improved under
learning-rate and budget changes, so the observed transition should not be
interpreted as a hard architectural capacity limit.

![Fig 5 — class-count convergence transition](figures/fig5_wall.png)
*Figure 5. Left: convergence fraction vs C at w=256/20k steps, with the
descriptive logistic fit and bootstrap band. Right: estimated C₅₀ versus width
at 20k and 200k steps. The width and budget series are sparse and should not
be read as a fitted functional law.*

### 4.6 Weight-bulk scale is weakly associated with validation accuracy

In the evaluated weight-decay sweep, the bulk scale changes from 1.0 to near
zero times its He scale while mean validation accuracy changes by less than
0.008. This result does not establish functional equivalence outside the
measured validation distribution. The fixed-floor MP count changes abruptly
between weight decay 0.1 and 0.2 while bulk depletion is smooth. In this
depleted regime, the MP-count interpretation is invalid; rank-based summaries
should be used instead.

![Fig 6 — bulk scale and validation accuracy](figures/fig6_bulk_inert.png)
*Figure 6. The weight-decay sweep at C=10, converged nets only. Accuracy is
nearly constant (top) while the L0 bulk scale changes by three orders of
magnitude (middle). The fixed-floor census reads C−1 until the bulk crosses
the floor between weight decay 0.1 and 0.2, then reads zero; this is a
measurement-regime transition, not evidence that the low-rank structure
disappeared.*

### 4.7 Population-specific correction of analytic predictions

A companion random-ensemble measurement (reported separately in
`analytic_vs_sampling/` in the ARC replication repo) grounds the baseline:
the uncorrected chain's error is *depth-flat once the state trajectory
saturates* (constant to <2% across d=24→48 at w=256, 48 nets/cell) and beats
FLOP-matched Monte Carlo at a crossover depth that rises monotonically with
width — the analytic skeleton is a stable object to correct against at any
depth. On trained weights the picture changes: the k=2 mean-field chain
predicts random-net activation spectra to 3–7%
through 32 layers but degrades to 80%+ by L16 on trained weights. A
state-keyed correction (160 parameters with edge indicators) reduces error on
the validation population by about 7–12-fold in mid layers. The validation set
also selected the number of polish iterations, so this is a validation result,
not an untouched held-out estimate. Transfer to the C=50 population is mixed
and worsens errors in some layers; the correction is therefore not established
as regime-general. A 16-parameter constant-coefficient comparison achieves
about 1.4-fold improvement in the evaluated configuration.

![Fig 7 — population-fitted correction](figures/fig7_refit.png)
*Figure 7. Per-layer spectrum error on the C=10 validation population. The
uncorrected chain degrades at mid depth; the state-keyed refit reduces mid-layer
error by approximately 7–12-fold. The same validation population selected the
polish iterations, so these are validation rather than held-out estimates.*

## 5. Discussion

### 5.1 Interpretation within the matched corpus

The controlled GMM experiments show that several weight-side measurements are
associated with activation-side measurements. In particular, the input-layer
MP count tracks the class-mean simplex rank within the evaluated converged GMM
runs, and the propagated effective-dimension descriptor differs systematically
from its initialization reference across task families. Noise-input and
task-input activations have similar early-layer anchored counts, but this does
not show that weights alone determine all activation geometry: the inputs share
only their first two moments, and the paper does not test every possible
location statistic.

The alignment experiment further shows that raw covariance eigenspaces and
Procrustes-aligned eigenspaces answer different questions. For the evaluated
C=10 twin population, raw overlap is near the isotropic baseline while
fit/test Procrustes alignment is high. This result is limited to the selected
architecture, task, metric, and input protocol; it does not establish a
general equivalence class of learned representations.

The analytic chain remains useful as a baseline, but the fitted correction is
validated only on the population used for model selection and has mixed C=50
transfer behavior. An independent test population is required before making
claims about out-of-population predictive performance.

### 5.2 Reading weights versus running the model

Inference is the end product — a model matters only when it runs — so it is
worth being precise about what this paper claims can be known before it runs.
The instruments operate at three levels of access:

1. **Weights alone, analytically** (§3.1, §3.2; results §4.1, §4.2): no
    empirical input samples or network execution, conditional on the assumed
    input model. In the matched GMM corpus, the input mean and covariance are
    known, but the mixture does not have the null's full Gaussian law.
2. **Weights executed on structureless input** (§3.3; §4.3): mechanically
   inference, epistemically still a weights reading — the input carries no
   task information, so whatever structure appears is attributable to the
   weights.
3. **Weights executed on task input** (§4.3, §4.4): inference proper. Every
   reading is now a property of the composition (weights ∘ input
   distribution); attributing structure to learning requires an
   input-matched control, which this corpus has by construction and wild
   models never grant.

The measurements provide evidence that selected location and effective-
dimension summaries can be read at level 1, while levels 2 and 3 assess their
behavior under controlled input distributions. The study does not establish
that every level-3 location statistic is fixed by level-1 measurements.

Each measurement regime has a limitation. In §4.6, weight-bulk scale is not
closely associated with validation accuracy in the evaluated sweep. In §4.4,
raw eigenspace overlap does not capture the Procrustes-aligned result for the
selected twin population. These observations motivate reporting both
weight-side and activation-side measurements rather than treating either as a
complete account of network behavior.

The propagated q descriptor connects a weight-only calculation with an
activation-side effective-dimension comparison under noise input. Its use
requires an explicit input-model assumption; it is not an input-distribution-
free prediction.

### 5.3 For analytic estimation on real models

The random-weight chain provides a defined baseline for the matched setting,
but it is not an accurate trained-network predictor without an additional
population-specific fit. The observed correction requires about 10^2
parameters and has mixed transfer to partially trained C=50 networks. The
results also motivate explicit validity flags: the fixed MP count is not
interpretable after bulk depletion, and the activation anchored count inverts
when initialization activations are more collapsed than trained activations.
The entry-shuffled reference is preliminary outside the matched corpus.

### 5.4 For weight-space auditing and representational convergence

For weight-only auditing, this corpus shows that spectral amplitude can vary
substantially with weight decay while validation accuracy is nearly unchanged.
MP counts should not be interpreted in the depleted-bulk regime. For
representational comparisons, raw and Procrustes-aligned covariance eigenspace
metrics produce different results in the C=10 twin experiment. Whether either
pattern transfers to other architectures, optimizers, or objectives remains an
open empirical question.

### 5.5 Outlook: wild models

Everything above is a matched-corpus result. A preliminary analysis of
Pythia-70m/160m MLP blocks used scaled and entry-shuffled references because
their initialization distribution was unavailable. For these matrices, the
entry-shuffled reference gave 0--12 significant dimensions and model
reinitializations gave 0--1. This limited sanity check does not validate
entry-shuffling for arbitrary architectures or training procedures. A
separately controlled external-model study is required.

## 6. Related work

**Analytic propagation on random networks.** The signal-propagation /
mean-field line [Poole et al. 2016; Schoenholz et al. 2017] studies random
deep nets in expectation over both weights and inputs, and the NNGP/NTK line
[Neal 1996; Lee et al. 2018; Jacot et al. 2018] takes the infinite-width
limit. The regime here is different and less studied: *one finite network
with its actual weights*, statistics over random inputs — the setting
formalized by the heuristic-arguments program [Christiano et al. 2022] and
operationalized by the ARC White-Box Estimation Challenge and its backbone
paper [Wu et al. 2026], with rare-behavior estimation as the motivating
application [Wu & Hilton 2024]. We inherit that machinery unchanged and
invert its role: on trained networks the propagated skeleton is the null, and
deviation from it is the measurement (§4.2, §4.7).

**Training signatures in weight matrices.** The closest antecedent to the
weight census is Thamm, Staats & Rosenow [2022], who compare trained-network
weight spectra against random-matrix predictions and find the bulk still
conforming to Marchenko–Pastur after training, with learned information
confined to the largest singular values — RMT comparison as a way to *locate*
learning. Martin & Mahoney read training signatures in wild models' spectra —
heavy-tailed self-regularization, quality prediction with no access to data
[Martin & Mahoney 2021; Martin, Peng & Mahoney 2021]. The Neural Feature
Ansatz [Radhakrishnan et al. 2024] supplies the mechanism by which task
structure enters weight Grams: WᵀW tracks the average gradient outer product.
The matched corpus adds an initialization-anchored MP floor
(σ² = 2/fan_in) and a controlled population in which the GMM input-layer count
equals C−1 for the evaluated converged runs. The weight-decay sweep further
shows that bulk scale can change substantially without a comparable change in
validation accuracy (§4.6). The fixed-floor count therefore has a
regime-dependent interpretation once the bulk is depleted.

**Simplex geometry and neural collapse.** Neural collapse [Papyan et al.
2020] finds last-layer features of converged classifiers collapsing to the
C−1-dimensional simplex ETF — an *output-side, activation-space,
terminal-phase* phenomenon — and has since been traced into intermediate
layers [Rangamani et al. 2023; Parker et al. 2023], where collapsed layers
carry low-rank weights. Relatedly, SGD with weight decay provably biases
weight rank down [Galanti et al. 2022; Zangrando et al. 2024]. The input-rank
law (§4.1) is the input-edge, weight-space counterpart: the class-mean
simplex rank is reflected in W₀ under the analytic floor and is detectable in
some nonconverged runs (§4.1, §4.5). Section 4.6 shows that the fixed-floor
count changes regime as the decayed bulk crosses the specified threshold while
validation accuracy remains nearly unchanged.

**Representational similarity and convergence.** The alignment literature
[Li et al. 2016; Raghu et al. 2017 (SVCCA); Kornblith et al. 2019 (CKA)] and
the Platonic Representation Hypothesis [Huh et al. 2024] argue trained
representations converge; permutation/rotation symmetry work [Ainsworth et
al. 2023] explains why raw coordinates cannot show it. Our controlled twin
population sharpens both claims into a measurement (§4.4): with task, data,
architecture, and init distribution all matched, raw eigenspace overlap is near
the isotropic baseline while a fitted rotation recovers 0.90–0.99 in the
selected C=10 experiment. Anchor-based "relative representations" [Moschella et al. 2023]
operationalize the same prescription from the engineering side: comparisons
made relative to fixed anchor samples are invariant to latent isometries,
which is §4.4's input-anchored reading as a design principle.

**Model zoos and weight-space learning.** Populations of trained nets exist
as datasets for meta-learning — model zoos [Schürholt et al. 2022], accuracy
prediction from weights [Unterthiner et al. 2020]. Those populations vary
hyperparameters to *cover* behavior space; ours is constructed around a
specified initialization distribution with GMM inputs matched in mean and
covariance. We did not identify prior work constructing a trained-model
population explicitly for comparison with a computable analytic null.

## 7. Limitations
The study uses one architecture family: plain bias-free ReLU stacks. The GMM
tasks are linearly separable by construction, and the real-data outcome labels
use an empirical performance proxy. The mechanism of the class-count
transition is unresolved. The correction has not been tested across optimizers
or objectives, and the Pythia analysis is preliminary rather than a transfer
study.

## 8. Reproducibility and reporting
The release includes the corpus, per-network provenance, analysis code, and
result summaries. Per-network error quantities can be heavy-tailed; reports
therefore use medians and bootstrap intervals where available. The paper also
identifies statistics whose interpretation is regime-dependent, including the
MP count after bulk depletion and the activation anchored count when it reaches
the stored-eigenspectrum limit.

## 9. References

- ARC White-Box Estimation Challenge (2026). Competition organized by the
  Alignment Research Center on AIcrowd.
  https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026.
  Warm-up June 10–17; Phase 1 June 18–July 31; Phase 2 August 1–September 19,
  2026. Accessed 2026-08-18; archived at
  https://web.archive.org/web/20260818152858/https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026.
- Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging
  Models modulo Permutation Symmetries. ICLR 2023 (oral). arXiv:2209.04836.
- Christiano, P., Neyman, E., & Xu, M. (2022). Formalizing the presumption of
  independence. arXiv:2211.06738.
- Galanti, T., Siegel, Z. S., Gupte, A., & Poggio, T. (2022). SGD and
  Weight Decay Secretly Minimize the Rank of Your Neural Network.
  arXiv:2206.05794.
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). Position: The Platonic
  Representation Hypothesis. ICML 2024. arXiv:2405.07987.
- Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel:
  Convergence and Generalization in Neural Networks. NeurIPS 31, 8571–8580.
  arXiv:1806.07572.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of
  Neural Network Representations Revisited. ICML 2019, PMLR 97, 3519–3529.
- Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., &
  Sohl-Dickstein, J. (2018). Deep Neural Networks as Gaussian Processes.
  ICLR 2018. arXiv:1711.00165.
- Li, Y., Yosinski, J., Clune, J., Lipson, H., & Hopcroft, J. (2016).
  Convergent Learning: Do different neural networks learn the same
  representations? ICLR 2016. arXiv:1511.07543.
- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for
  some sets of random matrices. Matematicheskii Sbornik 72(114)(4), 507–536.
  English translation: Mathematics of the USSR-Sbornik 1(4), 457–483.
- Martin, C. H., & Mahoney, M. W. (2021). Implicit Self-Regularization in
  Deep Neural Networks: Evidence from Random Matrix Theory and Implications
  for Learning. Journal of Machine Learning Research 22(165), 1–73.
  arXiv:1810.01075.
- Martin, C. H., Peng, T. S., & Mahoney, M. W. (2021). Predicting trends in
  the quality of state-of-the-art neural networks without access to training
  or testing data. Nature Communications 12, 4122. arXiv:2002.06716.
- Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., &
  Rodolà, E. (2023). Relative representations enable zero-shot latent space
  communication. ICLR 2023 (oral). arXiv:2209.15430.
- Neal, R. M. (1996). Bayesian Learning for Neural Networks. Lecture Notes
  in Statistics 118. Springer.
- Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of Neural
  Collapse during the terminal phase of deep learning training. PNAS
  117(40), 24652–24663. arXiv:2008.08186.
- Parker, L., Onal, E., Stengel, A., & Intrater, J. (2023). Neural Collapse
  in the Intermediate Hidden Layers of Classification Neural Networks.
  arXiv:2308.02760.
- Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., & Ganguli, S.
  (2016). Exponential expressivity in deep neural networks through transient
  chaos. NeurIPS 29 (NIPS 2016). arXiv:1606.05340.
- Radhakrishnan, A., Beaglehole, D., Pandit, P., & Belkin, M. (2024).
  Mechanism of feature learning in deep fully connected networks and kernel
  machines that recursively learn features. arXiv:2212.13881. Published as
  "Mechanism for feature learning in neural networks and
  backpropagation-free machine learning models," Science (2024),
  doi:10.1126/science.adi5639.
- Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA:
  Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics
  and Interpretability. NeurIPS 30 (NIPS 2017). arXiv:1706.05806.
- Rangamani, A., Lindegaard, M., Galanti, T., & Poggio, T. A. (2023).
  Feature learning in deep classifiers through Intermediate Neural Collapse.
  ICML 2023, PMLR 202, 28729–28745.
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A Reduction of Imitation
  Learning and Structured Prediction to No-Regret Online Learning. AISTATS
  2011. arXiv:1011.0686.
- Schoenholz, S. S., Gilmer, J., Ganguli, S., & Sohl-Dickstein, J. (2017).
  Deep Information Propagation. ICLR 2017. arXiv:1611.01232.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal
  Procrustes problem. Psychometrika 31(1), 1–10.
- Schürholt, K., Taskiran, D., Knyazev, B., Giró-i-Nieto, X., & Borth, D.
  (2022). Model Zoos: A Dataset of Diverse Populations of Neural Network
  Models. NeurIPS 2022, Datasets and Benchmarks Track. arXiv:2209.14764.
- Thamm, M., Staats, M., & Rosenow, B. (2022). Random matrix analysis of
  deep neural network weight matrices. Physical Review E 106, 054124.
  arXiv:2203.14661.
- Unterthiner, T., Keysers, D., Gelly, S., Bousquet, O., & Tolstikhin, I.
  (2020). Predicting Neural Network Accuracy from Weights. arXiv:2002.11448.
- Wu, G., & Hilton, J. (2024). Estimating the Probabilities of Rare Outputs
  in Language Models. arXiv:2410.13211.
- Wu, W., Lecomte, V., Winer, M., Robinson, G., Hilton, J., & Christiano, P.
  (2026). Estimating the expected output of wide random MLPs more
  efficiently than sampling. arXiv:2605.05179.
- Zangrando, E., Deidda, P., Brugiapaglia, S., Guglielmi, N., & Tudisco, F.
  (2024). Provable Emergence of Deep Neural Collapse and Low-Rank Bias in
  L²-Regularized Nonlinear Networks. arXiv:2402.03991.
