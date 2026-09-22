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

Related: `../arc-whitebox-canary` FINDINGS-equivalent work is untouched by
this; this is purely MZC + GEM + the Solitaire side investigation.
