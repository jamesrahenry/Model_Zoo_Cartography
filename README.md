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

## Vendored code

MZC is internally reliant: the census tooling (AMC's `manifold_detector.py` /
`feature_tracker.py`) and the analytic null machinery (ARC's `analytic_vacuum.py`,
stabilized moment chain) are vendored into `census/` and `null_baseline/` and
updated **here**. See `PROVENANCE.md` for sources, commits, and the list of
pending adaptations.

## Status

*Updated: 2026-08-18 20:45 UTC. All planned local phases complete; paper in external review.*

Corpus: **1,569 trained nets, 69 families** — widths 64–512, depths 8–64,
GMM C ∈ {2..72} with separation and weight-decay sweeps, whitened MNIST and
Fashion-MNIST, budgets 20k–200k steps, lr arms, two readout modes, 16–32
seeds per config. Weights + provenance: HF dataset
`james-ra-henry/MZC-Corpus` (private; flips public with the paper); local
corpus is a prune-and-refetch cache (`train/corpus_io.py`). Paper draft:
`paper/DRAFT.md` (v0.12 — one-thesis scope structure, seven figures from
committed data via `paper/make_figures.py`, all references verified, three
external review passes absorbed). Public spin-offs: the analytic-vs-sampling crossover
map and the trained-network state-keyed refit
(`trained_refit/`), both in the ARC replication repo. Key results — consolidated
digest with instruments and data pointers in **[FINDINGS.md](FINDINGS.md)**
(running record: Hopper task t4b9971d):

- **L0 rank law, 20-seed exact**: input-layer significant dims = C−1 through
  C=32, at every separation (1.5–6.0), on GMM and MNIST alike. **Scale
  caveat**: meaningful weight decay (≥0.3) shrinks weights below the
  fixed-σ² MP floor and the counter reads 0 while the structure demonstrably
  remains — the census needs a scale-normalized variant for decayed nets.
- **Expressivity wall, measured**: convergence fraction 1.00 through C=25 →
  0.95 (C=32) → 0.30 (C=40) → 0.00 (C=50); seed variance grows 10× at the
  crossing. Mid-net activation code saturates at ~14 effective dims (~21 sig
  dims) for every C ≥ 25 regardless of convergence.
- **Rank-collapse arrest**: trained nets crash to task rank at L0 and hold
  flat to L31; random nets decay smoothly to the fixed point.
- **Weights carry the "where"**: noise-input activations on trained weights
  show the full structure signature; task input sharpens, not creates.
- **Directional consistency**: same-task nets learn the same L0 subspace
  (init controls at isotropic chance); detects partial learning rank misses.
- **Sharing is coordinate-bound** (answering P4's cross-pollination question):
  same-task twins overlap at exact k/d chance in raw activation eigenspace at
  every depth — subspace sharing exists only where nets share coordinates
  (input space / L0). See `notes/2026-08-13_mzc-eigenspace-overlap-reply.md`.
- **Refit chain**: 128 state-keyed parameters repair mean-field's mid-net
  failure on trained weights ~7–12× on validation (which also gates polish —
  see FINDINGS F4); edge indicator dims fix L0/L1; partial-learner transfer
  remains the honest open failure.

## Data

All heavy data lives on the HF dataset `james-ra-henry/MZC-Corpus` (private;
flips public with the paper) — this git repo holds code, docs, and the small
summary JSONs only:

- `corpus/<family>/` — trained weights + full per-net provenance
  (`train/corpus_io.py` re-downloads pruned nets on demand).
- `analysis/census/`, `analysis/null_baseline/` — the per-net analysis JSONs
  (weight/activation censuses, state trajectories; moved out of git
  2026-08-18, history purged). Restore into the working tree with:

  ```bash
  hf download james-ra-henry/MZC-Corpus --repo-type dataset \
      --include "analysis/*" --local-dir .
  rsync -a analysis/ ./ && rm -rf analysis/
  ```

## Testing

`tests/` covers the pure-numpy/torch instrumentation (census, null baseline,
task construction, architecture) with synthetic data and no GPU required —
see `tests/README.md`. `pip install -r tests/requirements.txt && pytest`.

## License

MIT — see `LICENSE`. Vendored code (`PROVENANCE.md`) is compatibly licensed
(all source repos are authored by jamesrahenry; `arc-whitebox-replication` is
additionally MIT).
