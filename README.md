# Model Zoo Cartography (MZC)

*Started 2026-08-10.*

Where does training leave a signature in a deep MLP, relative to the statistical
skeleton the same architecture has before any training at all?

## Origin

Two sibling Rosetta programs motivate this one but are otherwise independent of it:

- **[ARC White-Box Estimation Challenge](../arc-whitebox-canary/)** — a moment-propagation
  estimator for **random**-weight (He-Gaussian, bias-free ReLU) depth-32/width-256 MLPs.
  The challenge nets are never trained on anything; the estimator only has to predict
  their output statistics. The write-up's own §8 ("toward non-random networks") flags
  the natural extension: random-weight moment propagation gives an *analytic null
  model* — what a net at this architecture looks like with zero learned structure — and
  asks what changes once the weights are no longer random. It names "a population of
  trained networks with shared statistics" as the main obstruction to answering that,
  and calls it out as future work.
- **[Activation Manifold Cartography](../Activation_Manifold_Cartography/)** — label-free
  structure-hunting in *activations* of existing trained models (eigenspectrum census,
  Marchenko-Pastur noise floor, feature tracking). AMC's tooling has never been pointed
  at weights directly, or at a population where the *only* difference between members is
  training — everything else (architecture, init distribution) held fixed.

MZC is the missing middle: build the population ARC's write-up says is needed, using
AMC's census tooling (on both weights and activations) to ask where training deviates
from the random-init baseline ARC already knows how to compute analytically.

If this finds a real, structurally-detectable signature, ARC's Phase 2 (per its own
write-up) inherits a usable trained-network null-model check for free. That is a nice
outcome but not the goal — the goal is the AMC question: can pre-inference structure
predict where concepts will form once inference starts.

## Plan

1. **`train/`** — train many depth-32/width-256, He-Gaussian-init, bias-free-ReLU MLPs
   (ARC's exact Phase-1 architecture) as classifiers. No skip connections, no
   normalization layers — matching ARC's spec means accepting the vanishing/exploding
   risk that comes with it. If plain classifiers don't converge at this depth, the
   fallback is scaling task/width up until they do, not changing the architecture (the
   corpus needs to stay analytically comparable to ARC's null model, or the comparison
   is meaningless).
2. **`corpus/`** — trained weights plus full training provenance per net: task, dataset,
   hyperparameters, seed, convergence outcome (converged / stalled / diverged), final
   accuracy. Keeping "what it was trained on" alongside the weights is the point — a
   corpus of weights with no record of what produced them can't support the
   deviation-from-null analysis this program exists for.
3. **`null_baseline/`** — ARC's analytic random-init moment-propagation baseline,
   adapted as the reference skeleton to diff the trained corpus against.
4. **`census/`** — AMC's manifold-census tooling (eigenspectrum, participation ratio,
   Marchenko-Pastur threshold, feature tracking), run on weight matrices directly
   (no forward pass) and on activations (post-inference), for every corpus member.
5. Compare: does weight-space structure predict activation-space structure? Does
   deviation from the null baseline correlate with anything task-relevant? Does it
   resemble the CAZ relational-merge/diverge patterns AMC found in real LLMs?

**Later iteration (not in scope yet):** pull small pretrained models directly off
Hugging Face and run the same census against them, once the self-trained,
architecture-matched corpus has established what a clean signal looks like. HF models
won't match ARC's null architecture, so they're a second, looser test of generality —
not a substitute for the matched corpus above.

## Status

Scaffolding only, as of 2026-08-10. No training runs yet.
