# ARC → MZC handoff: measured state descriptors, saturation, and atypicality

*Written: 2026-08-10 18:44 UTC — claude:chain-depth-generic (ARC-side session)*

Three findings from the ARC depth-generic stabilizer redesign
(`arc-whitebox-canary/moment_chain/`, commit `b6e75e1`) that bear directly on MZC's
plan. Full numbers in the three `*_results.json` files next to the scripts.

## 1. The null baseline's "statistical skeleton" is four numbers per layer, zero forward passes

The ARC estimator's per-layer corrections are now keyed to a measured state vector
computed from the analytically propagated (mean, covariance) — no inference, weights only:

- `q = PR(Σ_pre)/w = tr(Σ)² / (w·‖Σ‖²_F)` — participation-ratio fraction, the
  rank-collapse clock. No eigendecomposition needed (trace and Frobenius norm only).
- `abar, asd` — mean/std of `a = μ/σ` (the ReLU operating point).
- `rbar` — mean |off-diagonal correlation| (correlation buildup).

These four, per layer, replaced a 512-number depth-32-specific lookup table at parity
(held-out 4.74e-6 vs 5.21e-6) with 128 parameters and no layer index anywhere. For MZC:
this trajectory **is** a candidate `null_baseline/` fingerprint — for a random He net it
is computable analytically from architecture + weights, and `chain_state_keyed.py`
(`step_np`, `state_basis`, `roll_chain`) is a working numpy implementation. Suggested
first corpus measurement: the (q, abar, asd, rbar) depth-trajectory for every trained
net vs the random-init ensemble band, *before* any eigenvector census — it's far cheaper
and already known to carry signal (see §3).

## 2. Deep plain ReLU MLPs saturate: the state stops moving past ~L30

E2 (`depth_extrap_eval.py`): coefficients fitted at depth ≤ 32 extrapolate to depths 48
and 64 with no loss — and so does naive last-row clamping — because the state trajectory
reaches a fixed-point regime (rank fully collapsed, slow drift). Depth-64 estimation
error is *lower* than depth-32 (median 2.6e-6 vs 4.7e-6): the saturated regime is
effectively low-dimensional and easier to predict.

MZC implications: (a) for the random null, depth beyond ~32 adds little new ensemble
structure — corpus depth diversity budget is better spent below saturation; (b)
training-induced structure has a natural signature to look for: **deviation from, or
displacement of, the saturation trajectory** — e.g., does training arrest the rank
collapse (hold q higher at depth), or accelerate it? That's a single scalar-per-layer
question answerable on day one of the corpus.

## 3. Atypicality is real, measurable, and is (part of) the hard tail

E3 (`tail_robustness_eval.py`, 200 unseen challenge nets): replacing "coefficients for
layer l" with "coefficients for the measured state" improves estimation 1.06× in the
body but **1.20× in the p90+ tail**, corr(log difficulty, log improvement) = +0.31.
I.e., even among *random* nets, individual state trajectories deviate from
ensemble-typical enough to matter, and the hard-to-estimate tail is substantially the
atypical-trajectory population.

MZC implications: the "statistical skeleton" varies meaningfully net-to-net even at
random init — so the null is a *band*, not a curve; use the random ensemble's trajectory
distribution, not its mean. And the working hypothesis sharpens: if random-init
atypicality already predicts estimation difficulty, then training-induced structure
should register as atypicality that is *consistent across nets trained on the same task*
— directionally aligned deviation, where random atypicality is isotropic. That's a
testable corpus-level statement.

## Practicalities

- Challenge weight convention (verified against challenge nets): `W ~ N(0, 2/fan_in)`,
  i.e. std `sqrt(2/256) ≈ 0.0884` at width 256. `depth_extrap_eval.py:gen_net` generates
  matching nets; `mc_truth` is a cheap batched MC ground-truth pattern (float32 compute,
  float64 accumulate).
- One methods warning that transfers: corrections/probes fitted on a population must be
  fitted on the *corrected system's own trajectory* (sequential/DAgger), not on the bare
  system's — the pooled naive fit diverged within two iterations (writeup §5's 16:1
  bias-amplification asymmetry, independently reconfirmed here). Any MZC scheme that
  calibrates a detector against rolled-forward null statistics inherits this.
