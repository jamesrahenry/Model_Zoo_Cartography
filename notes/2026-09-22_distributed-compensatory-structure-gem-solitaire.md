# Distributed, compensatory structure: GEM (concepts, LLMs) and Solitaire
# (rank, RL-trained MLP) converge on the same shape of finding

*Written: 2026-09-22 18:09 UTC*

## The observation

Two independent lines, different programs, different methodology, different
model class, arrived at the same shape of result within days of each other:

**GEM** (`../papers/gem/preprint.md`, Henry 2026b) tracks a concept's causal
footprint across the Concept Allocation Zones it passes through in real
trained transformers, then ablates at each site. Conclusion: no single layer
is sufficient. The deepest GEM is typically the best single ablation site
(40/42 atlases) but ablating it alone never approaches complete suppression,
and — the sharper claim — **ablating a shallower GEM measurably changes what's
recoverable at a deeper one in 37/42 atlases** (preprint.md:696). Their own
framing (preprint.md:629, 670): "every layer makes some contribution... what
CAZ/GEM identify are the layers where that contribution is large enough to be
geometrically organized... It is not a complete account."

**Solitaire** (`../../Solitaire/agent/structure_metrics.py`, this thread,
2026-09): a from-scratch DQN with LayerNorm added specifically to prevent
net.0's effective-rank collapse. It worked exactly as targeted — net.0 stayed
near its init rank (304 vs. baseline's 2.6 effective dims at 400k steps) —
but the readout layer collapsed *harder* than baseline in exchange (3.3 vs.
62.8 effective dims), and win rate was consistently worse throughout training
(22-29% vs. baseline's 32-38%). See run comparison: `023_...` (baseline),
`025_..._layernorm` (intervention).

Stripped of domain: **intervene at one layer → the thing being measured
doesn't disappear, it relocates to another layer.** GEM shows this for causal
concept-suppression in trained LLMs via ablation. Solitaire shows it for
structural rank in a tiny from-scratch RL net via a normalization change.
Different measurement (causal intervention vs. spectral/rank), different
training regime (pretrained-then-probed vs. from-scratch RL), different scale
by orders of magnitude — same shape of result.

## The hypothesis this suggests

Depth-stacked networks distribute whatever computational/representational
work a task demands across layers, and an intervention at one site —
ablation, normalization, anything that changes what that site can carry —
does not remove that work. It redistributes it. This is a stronger and more
specific claim than "structure is distributed across layers" (which MZC's own
F2/F5 already established in different ways — task-code-driven terminal rank,
rotation-hidden but real cross-net sharing). It is specifically about
**compensation under intervention**: perturb layer A, and layer B's structure
changes as a *consequence*, not independently.

Neither GEM nor the Solitaire run can test this hypothesis cleanly on its
own. GEM has no matched null model — it studies real pretrained LLMs, so
"how would this concept's structure have looked without the ablation" is
counterfactual, not measured. Solitaire has no controlled population —
one seed, one architecture, one intervention, no matched-baseline rigor.

## Why MZC is the right vehicle for this

MZC already has exactly the missing piece: a matched, controlled population
with a known random-init baseline, and instruments built to separate real
structure from artifacts (`census/run_census.py`'s two-floor MP census,
`census/directional_consistency.py`, `census/procrustes_overlap.py`). A
clean test of the compensation hypothesis inside MZC's own architecture:

- Train matched pairs of nets — one arm unconstrained (baseline), one arm
  with a targeted intervention at a single layer (candidate interventions,
  cheapest first): (a) freeze a layer's weights after some step count
  (matching LayerNorm's "prevent this layer's collapse" effect by a more
  surgical mechanism), (b) add LayerNorm at one specific depth only, (c)
  clamp a layer's effective rank directly via a low-rank reparameterization.
- Run the existing weight census on *every* layer, not just the intervened
  one, for both arms.
- The hypothesis predicts: layers *other than* the intervened one should show
  a measurably different collapse trajectory between arms — not just "the
  intervened layer changed" (trivially true by construction) but "some other
  layer's rank/timing changed as a consequence." A null result (only the
  intervened layer differs, everything else matches baseline) would be a
  real falsification, symmetric with how F6's w=64 crossover-law and the
  C50(1024) lr-tuning result were both honest falsifications of their own
  pre-registered predictions.
- This is the first MZC investigation with a *cross-layer* dependent
  variable by design — every finding so far (F1-F7) characterizes a layer
  (or the same layer across nets); this would explicitly measure whether
  layers move *together*.

Not scoped or scheduled yet — this note exists to register the hypothesis
and the observation that motivated it before it's lost, per this project's
pre-registration discipline (F6 already has one on record; do the same
here before any run happens, not after).

## Open questions before designing the experiment

1. ~~What counts as "layer B's structure changed as a consequence of A's
   intervention" vs. "layer B just varies run to run anyway"? Needs a
   seed-variance baseline (matched-seed unconstrained pairs) to know what
   noise floor a real compensation signal has to clear.~~ **Resolved
   2026-09-22 (James): this has an ARC shape — build a null, z-score
   deviations against it, exactly F2's existing q-clock convention
   (`z(L31) = +3.2 to +4.2` against the random-init null band), just pointed
   at a different reference population. ARC's own analytic null (zero
   forward pass, random-init statistics) answers "is there structure at
   all" — not the question here. What's needed is a **trained-baseline
   null**, not a trained-vs-random one: K≥8-16 seeds, unconstrained, per
   non-intervened layer gives an empirical mean±std under pure seed noise
   at fixed recipe. Z-score the intervention arm's same-layer, same-seed-
   count measurements against that band. A layer whose z-score sits inside
   the baseline's normal spread is noise; a layer clearly outside it is
   real compensation. This is also why the single-seed Solitaire
   observation (one run each side) can't actually answer its own question —
   there's no null band to check against at n=1; the multi-seed baseline
   arm is not a nicety, it's the whole mechanism that makes this testable
   rather than anecdotal.
2/3. ~~Is the right dependent variable effective_dim or something more
   causal? GEM's compensation is about a specific concept's causal
   footprint; MZC's would be about aggregate rank — same shape, not the
   same claim, until tested.~~ **Resolved 2026-09-22 (James: "trying to
   answer this is why we're here" — pushed to actually design it rather
   than leave both parked): a two-phase design answers both at once,
   gated so the expensive phase only runs if the cheap one earns it.**

   **Phase 1 — cheap, aggregate, gate.** Train matched baseline/intervention
   arms (K≥8-16 seeds each, train-time intervention: freeze a layer,
   LayerNorm at one depth, or a low-rank reparameterization — per the
   candidate list above). Run the existing weight census on every layer.
   Z-score every non-intervened layer's intervention-arm effective_dim
   against the baseline arm's own null band (item 1, above). This tests the
   *weak* claim: does intervening on A shift B's aggregate rank beyond
   normal seed variance? Cheap because it's exactly the census instrument
   already built. If nothing clears the null band anywhere but the
   intervened layer itself, the hypothesis is dead and phase 2 doesn't
   need to happen — a real, well-earned falsification, not a shrug.

   **Phase 2 — only if phase 1 clears the gate — causal, concept-specific,
   the actual GEM parallel.** MZC's classifiers have something GEM's real
   LLMs don't: an *exactly known* concept direction — F1's class-mean
   simplex (rank C−1, verified per-net with zero exceptions for C≥8). That
   makes the causal test cleaner here than in GEM, not just an analogy:
   post-hoc ablate the class-simplex subspace at layer A (already-trained
   phase-1 models, both arms) and measure recoverability at layer B —
   direct restatement of GEM's own ablate-and-measure-recoverability
   method (preprint.md:696), with concept identity controlled instead of
   discovered. Prediction: if phase 1's redistribution is real and
   functional (not just cosmetic rank movement), ablating the concept
   subspace at A should hurt task accuracy *less* in the intervention arm
   than in baseline — because B picked up some of A's causal load — mirroring
   GEM's cross-layer recoverability change (37/42 atlases) but with a fully
   known, controlled concept instead of a discovered one.

   This also directly answers the old item 3: phase 1 tests the *aggregate*
   version of the claim (comparable to Solitaire's rank measurements), phase
   2 tests the *concept-causal* version (comparable to GEM's actual method)
   — on the same trained models, in sequence, rather than picking one
   dependent variable and hoping it's the right one.

## Phase 1 result (2026-09-22): gate cleared, but localized, not distributed

Ran `train/compensation_phase1.py` (C=10, sep=3.0, wd=0, width=256, depth=32,
16 seeds/arm, ~0.5h wall) and `census/compensation_analysis.py`
(`census/compensation_phase1_results.json`).

**Freezing L0 is a strong intervention, not a no-op**: baseline converges
16/16 to 89.2-90.1% (matches Bayes ~90%); freeze_l0 converges 0/16, capping
at 62-64%. Expected — L0 does the C−1 task-extraction (F1); frozen at
random init, it can't, so every downstream layer is stuck working from an
uninformative projection of the input.

**Z-scores against the baseline arm's own null band** (16 seeds each,
`census_weights`'s effective_dim):

| layer | baseline eff_dim | freeze_l0 eff_dim | z |
|---|---|---|---|
| 0 (frozen) | 26.8 ± 0.5 | 127.3 ± 0.4 | 188 (trivial, excluded) |
| 1 | 100.1 ± 1.0 | 83.1 ± 1.4 | **−17.4** |
| 2 | 119.6 ± 0.8 | 112.2 ± 1.0 | **−8.9** |
| 3–31 | ~120 throughout | ~119–121 throughout | all \|z\| < 2 (max 1.97 at L29) |

**Gate cleared** — layers 1 and 2 show large, unambiguous shifts (z=−17,
−9; nothing borderline about them). But the shape is not what either GEM
or the loose reading of the Solitaire result would suggest: **compensation
here is local, not distributed** — it's absorbed within one to two layers
of the intervention and every one of the remaining 29 layers sits
statistically indistinguishable from baseline's own seed-to-seed spread.
Layer 3 onward converges to nearly the *same* effective_dim (~119-121)
regardless of whether L0 gave the network a useful task-projection or a
useless frozen-random one — echoing F6's fixed mid-net code ceiling (a
similar-sized downstream code forming close to independent of upstream
difficulty), now with a new data point: independent of upstream
*function*, not just upstream *difficulty*.

This refines the hypothesis rather than confirming or denying the loose
version of it: not "structure is distributed across depth when you
intervene," but "an intervention's disruption propagates a short, bounded
distance and the deep bulk of the network is unaffected." Whether that's
architecture-specific (bias-free ReLU stack, this exact width/depth) or
general is untested — the natural next question, cheaper than phase 2:
does the same 1-2-layer localization hold if the intervention is on a
*deeper* layer instead of L0?

Phase 2 (concept-ablation recoverability) is warranted on this evidence,
scoped to layers 0-2 where the actual shift lives, not all 32 layers.

## Full-layer sweep (2026-09-23): not L0-specific pattern, L0-specific *zone*

James's call: don't sample, freeze all 32 layers (`train/compensation_full_layer_sweep.py`,
16 seeds each, ~8.0h wall, 512 nets, zero errors) and z-score every arm
against the same baseline (`census/compensation_full_analysis.py`).
Frozen=0's row reproduced Phase 1 almost exactly from an independently
trained cohort (z=-17.35/-8.87 vs Phase 1's -17.4/-8.9, same seeds/config)
-- a real reproducibility check, not just consistency by construction.

**Only layers 0-3 show ANY compensatory effect. Layers 4-31 — 28 of 32,
87.5% of the depth — show zero shifted layers (radius 0) when frozen.**
Full picture for the input-edge zone:

| frozen | shifted (\|z\|>3) | radius |
|---|---|---|
| 0 | 1, 2 | 2 |
| 1 | 0, 2 | 1 |
| 2 | 1 | 1 |
| 3 | 2 | 1 |
| 4–31 | *(none)* | 0 |

Two things sharpen the picture beyond Phase 1 alone:

- **This isn't "compensation is local wherever you intervene."** It's
  narrower: only the input edge (L0-L3) participates in cross-layer
  compensation *at all*. The other 28 layers are not "locally compensating
  and then it stops" — freezing them produces no detectable effect
  anywhere, full stop. That's closer to *modularity* than to *bounded
  redistribution*.
- **The interaction is asymmetric and decays fast even within the zone.**
  L0→L1 is huge (z=-17.4); L1's effect on its two neighbors is more even
  (-8.75, -8.9); L2 only pulls backward toward the input (L1: -8.9) and not
  forward (L3: +0.92, noise); L3's pull on L2 is barely over threshold
  (-3.17) and nothing reaches L4. The zone doesn't have a sharp edge so
  much as strength that runs out by ~3 layers in.

Reading against F1/F6: this lines up with L0-L3 being where the C-1
task-extraction actually happens (not instantaneously at L0 alone — takes
a few layers to complete, matching the wd=0 arm's own observation that L0's
collapse is accompanied by continued eff-dim movement in L1-L2 before the
bulk stabilizes). Once that's done, the remaining ~28 layers process an
already-extracted signal and don't appear to need to renegotiate with each
other when one of them is disabled -- consistent with, and now more
specific than, F6's fixed mid-net code ceiling: not just similar-sized
regardless of task difficulty, but *structurally uninvolved in cross-layer
compensation* regardless of which one of them is disabled.

**Revised claim, weaker than the GEM-motivated original hypothesis**:
depth-stacked bias-free ReLU MLPs are not generally "distributed and
compensatory" through their whole depth. What compensates is specifically
the task-extraction zone at the input edge; the processing bulk behind it
is modular with respect to single-layer disablement, at least in this
architecture and at this measurement's sensitivity (16 seeds, \|z\|>3).
Whether GEM's transformers show the equivalent zone-boundedness (i.e., is
37/42 cross-layer recoverability itself concentrated near where a concept
enters/exits its CAZ, rather than uniform across depth) is now the sharper
question to ask back at GEM, not something this note can answer.

Phase 2 should stay scoped to layers 0-3 (where there's something to test)
-- testing concept-ablation recoverability on layers 4-31 would very
likely just reproduce their independence, not inform the compensation
question.

Data: `census/compensation_full_layer_results.json` (full 32x32 z-matrix).

Related: `../arc-whitebox-canary` FINDINGS-equivalent work is untouched by
this; this is purely MZC + GEM + the Solitaire side investigation.
