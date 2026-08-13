# External audit of FINDINGS.md (F1-F7): 2 broken, 3 solid, 2 mixed

*Written: 2026-08-13 by claude:p4-review (Rosetta_Program session). Requested by James after
an unrelated P4 conversation surfaced that the ARC/MZC cross-pollination note
(`2026-08-11_p4-eigengap-crosspollination.md`) had been constructed same-day, narrative-first,
and didn't hold up to scrutiny (see that P4 session for the ARC-side follow-up). James then asked
for the same adversarial standard applied to `FINDINGS.md`'s own seven findings, since that
document is explicitly written "for cross-program review" and is the thing most likely to get
cited elsewhere. Six parallel audits (one per F1-F4, F6, F7) plus a manual check of F5, each told
to actively try to break the claim, not confirm it. Full agent transcripts are not preserved here;
this note is the synthesis. Read this before trusting any specific number in `FINDINGS.md` — two
of the seven findings are currently wrong as stated, contradicted by data already sitting in the
files they cite.**

## Scorecard

| Finding | Verdict | Action needed |
|---|---|---|
| F1 (input-rank law) | Solid core, oversold edge cases | Scope MNIST/C=50 claims correctly (below) |
| F2 (rank-collapse arrest) | **Broken** | Rewrite from current 21-family data, or retract |
| F3 (weights carry the "where") | **Broken** | Fix census null-calibration, then rewrite |
| F4 (stabilizer refit) | Solid (reproduced from scratch) | Fix "10-17x" range; note reproducibility gap |
| F5 (Procrustes recovers same-task twins) | Solid | One wording precision fix |
| F6 (expressivity wall) | Mixed — real core, hollow precision | Caveat C50/width numbers until B3 lands |
| F7 (weight decay annihilates bulk) | Mixed — real core, false instrument claim | Fix "exact on synthetics"; disclose estimator failure |

The general shape: **the phenomena being studied are mostly real.** F1, F4, F5 survive genuinely
adversarial, from-scratch reproduction (not just a code read). Even F6 and F7's core claims hold
up under independent re-derivation. But `FINDINGS.md`'s prose consistently states more precision
and certainty than the specific cited numbers support, and in two cases (F2, F3) the actual
claims are simply wrong — contradicted by more complete data that was already sitting in the same
JSON files being cited, generated *before* `FINDINGS.md` was written but never checked against.

---

## F2 — broken, needs a rewrite from current data

**Claim:** trained nets' rank-collapse "holds flat... +5σ above the 50-net random band at depth,"
matching `q ≈ (C+1)/w`.

**Problem:** this is the untouched output of the *original 6-net, single-task (C=10) pilot* from
`3cf1ea4` (2026-08-10). `state_trajectories.json` was regenerated in `e1dce38` (2026-08-13
06:15 UTC) to n_null=50 and 21 run families — the file `FINDINGS.md` now cites — but the F2 text
was never recomputed against it. Recomputed:

- **0 of 21 families reach +5σ** (max is +4.24σ, `sweep_c25_head`).
- **6 of 21 families (29%) sit at or below the null band**, not above it — including
  `mnist_d32`, the only real-data (non-synthetic-Gaussian) family in the corpus, which shows *no
  plateau at all*, decaying monotonically past the null band all the way to L31.
- `q ≈ (C+1)/w` only fits near C=10 (the original pilot's task); the ratio of actual q(L31) to
  predicted (C+1)/w degrades from ~0.9-1.0 at C=10 to 0.15-0.43 for C≥15.
- "Hold flat to L31" is contradicted by `sweep_c25_head`'s own trajectory: 0.105 (L2) → 0.0376
  (L20, minimum) → 0.0437 (L31), a U-shape with a ~16% rebound, not a plateau.

**What's real:** the qualitative mechanism (rank-collapse decays faster at input, partially
arrests somewhere) has directional support in most families, and the instrument itself (`q` via
the analytic Hermite moment chain, `chain_state_keyed.py`) is legitimate and leak-free — purely
deterministic from weights, no sampling, no data dependence. The number the instrument is
currently reporting is what's wrong, not the measurement method.

**Fix:** recompute F2 against `state_trajectories.json` as it currently stands (all 21 families),
report the actual z-score distribution (likely "modestly elevated in most families, absent in
some, strongly negative in real-data/MNIST" rather than a single "+5σ" headline), and either drop
the `(C+1)/w` formula or restrict its claimed range to where it actually holds (roughly C=8-15).

## F3 — broken, root cause identified

**Claim:** activation census counts "exactly C significant dims at L0–L24"; signature is "flat
L1–L31, eff dim ≈ C+1."

**Problem:** computed directly from `census/activation_census.json`:

- Exact match (`sig_dims == C`) only actually holds at **C=10 and C=15**. At C=50 the mean
  deviation is −25 dims. **At L0 specifically, exact match is 0% for every single C value
  tested**, despite L0 being inside F3's own stated range.
- The high-C failure is the *same* expressivity ceiling **F6 already documents in this same
  file** ("mid-net activation code saturates at ~14 effective dims for every C≥25"). F3's
  "exactly C" directly contradicts F6's ceiling claim, internally, within one document.
- "Flat L1-L31" fails for C≥20 (C=25: 35.3 → 13.5 (L16) → 25.3 (L31), a U-shape) and fails
  outright for the only real-data condition (`mnist_d32`: continuous decay through depth, no
  early plateau, final floor nowhere near C+1).

**Root cause, not just a data mismatch:** `run_activation_census.py` calls the census
(`manifold_detector.py`) without passing an explicit `noise_variance` null, so the significance
threshold self-calibrates per layer from each matrix's own scale. `manifold_detector.py`'s own
docstring warns this is fine within one matrix but "NOT comparable across matrices of different
scale," and `PROVENANCE.md` already documents a case (0 vs. 26 "significant dims" for identical
underlying structure at different amplitude) motivating the explicit-null parameter that F1's
*weight* census deliberately uses for exactly this reason. The activation census never got the
same fix.

**Fix:** pass an explicit analytic or empirically-anchored noise floor into the activation census
(mirroring what `run_census.py` already does for weights), regenerate `activation_census.json`,
then rewrite F3 against the corrected numbers. Until that's done, F3's specific dimension-count
claims should not be cited or relied on — the qualitative "weights determine where/rank/
persistence, task input sharpens it" claim is fine (proven algebraically for the GMM family's
mean/covariance match, and the paired noise-vs-task comparison is genuinely within-net, no
cross-net confound) but the "exactly C" / "flat" specifics are not.

## F1 — solid core, three edge-case numbers need rescoping

**Claim:** L0 significant dims = C−1 exactly, for every converged net, invariant to separation,
including on MNIST and at C=32/50.

**Core claim: genuinely exact.** Checked at the raw per-net level (not aggregates): all 240
individual nets across the C-sweep and 4-way separation-sweep hit the integer C−1 exactly, zero
exceptions. The significance threshold (`manifold_detector.py`'s MP edge, `σ²=2/fan_in` fixed by
init spec) has no dependence on C anywhere in the code — not circular.

**Three things to fix in the write-up, not the science:**
1. MNIST's "9.4" figure is drawn from nets that are all formally `outcome: "partial"`, not
   converged, despite F1 explicitly scoping the law to converged nets. Root cause: MNIST's
   hardcoded Bayes-accuracy proxy (0.985, borrowed from a different architecture) doesn't match
   what this project's bias-free depth-32 net can actually reach (~0.92-0.95), so every MNIST
   net gets labeled partial regardless of whether the rank law holds for it. State this caveat
   explicitly rather than reporting 9.4 as if it were in-scope.
2. C=32's "30.9" is a floating-point rounding artifact of blending 19 converged nets (exactly 31)
   with 1 partial net (30). The converged-only mean is a clean 31.0 — report that instead; it's a
   *stronger* statement of the same law, not a weaker one.
3. C=50's "35.7 → 45.3, asymptoting" 60k-step comparison rests on 3 seeds, not 20 (undisclosed),
   and "asymptoting" isn't really supported by a 13-point accuracy jump (0.56→0.69) between the
   only two step-count checkpoints on record. Either get an intermediate checkpoint or soften the
   "asymptoting" language.

## F4 — solid, reproduced from scratch; one arithmetic fix needed

**Claim:** ARC's state-keyed stabilizer refit repairs the eigenspectrum bulk "10-17x held-out"
(citing L7/L15/L23: 0.67/0.84/0.88 → 0.050/0.066/0.122).

**Independently reproduced bit-for-bit**, not just read: downloaded the corpus, reconstructed the
exact original 9-fit/6-val/3-transfer net population (via net-provenance timestamp forensics,
since the current code path no longer reproduces this population by default — see below), and
re-ran the full sequential-DAgger fit + polish-accept/reject + held-out eval from scratch. Output
matched `refit_trained_results.json` to 4 decimal places. This included verifying the one place
val data touches the pipeline (the polish-loop's accept/reject gate) never actually accepted a
candidate (`ACCEPT_LOG [False, False, False]`, confirmed independently) — so "held-out" is
genuine, not a misnomer, and "fails on partial learners" is a real shown degradation (refit gets
*worse* than uncorrected on `transfer_c50`), not hand-waving.

**Two fixes needed:**
1. **Arithmetic error in the headline itself**: L23's own cited pair (0.8797 → 0.1220) is
   0.8797/0.1220 = **7.2x**, below the claimed 10-17x floor. The true per-layer ratio peaks at
   ~15x mid-depth and decays toward the output, dropping below 1x (correction actively hurts) at
   L0/L1. Either widen the stated range to ~4-15x or drop L23 from the illustrative triple.
2. **Reproducibility gap in current code**: `refit_trained.py`'s `load_trained()` used to glob a
   local `corpus/` directory that, at the time results were generated, happened to contain only
   the first 3 nets per run. A later refactor (`21b34e6`, same day) switched it to `net_paths()`,
   which now auto-downloads the full 20-net-per-run corpus from HF. Running the script today
   would silently use a much larger, different population and would not reproduce the committed
   numbers. Either pin the original net IDs explicitly in the script, or regenerate the results
   against the full population and update the committed JSON + FINDINGS.md together.

Minor: the "independent reconfirmation" language for the polish-rejection check overstates
"independent" — it's an internal self-consistency ablation within the same run, not external
replication. Accurate to call it a real, verified computation; not accurate to call it independent
confirmation.

## F5 — solid, one wording fix

**Claim:** raw activation-eigenspace overlap for same-task twins sits at "exact k/d chance... at
every depth," but a fitted, honestly-held-out Procrustes rotation recovers it (0.99 early / 0.90
at depth).

Checked `procrustes_overlap.py` directly for the one failure mode that matters most given how
central this claim is: **the fit/test split is honest** — R is fit on one half of shared-input
activations, recovered overlap is evaluated only on the other half, which R never saw. No
leakage. Init-twin controls are properly matched (same nets' own init weights). The recovered-vs-
init gap (0.90-0.99 vs 0.07-0.35) is large and clean.

**One fix:** "exact k/d chance... at every depth" is not quite right — raw overlap is genuinely at
chance (0.0352 vs 0.035) at shallow layers but drifts to ~0.044 at the deepest layers, about 25%
above chance. Small, but real and consistent across layers, not noise. State it as "at or very
near chance, with a small consistent elevation at depth" rather than "exact... at every depth."

## F6 — real core, hollow precision on the headline numbers

**Claim:** sharp convergence wall, logistic fit C₅₀=38.2, width=2.0; ~14-dim mid-net code
ceiling; Bayes-gap grows ∝ C^0.97 (r²=0.94); registered prediction C₅₀(w)=2.7×ceiling(w) for
Phase B.

**Real and robust:** the power-law exponent survives re-fitting under several different point-set
choices (0.969-1.075 across subsets, all consistent with "almost exactly linear" as the text
itself hedges — not an inflated claim). The ~21-significant-dims ceiling is genuinely flat
(<7% spread) across C=25-50. **The "registered prediction" is genuinely pre-registered** — git
history confirms no Phase B data existed when it was committed, and it's stated with real
falsification criteria. This was specifically checked given the project's history with this
exact failure mode, and it's clean.

**Hollow:** "C₅₀=38.2, width=2.0" is not a real fit — no fitting code exists anywhere in the repo.
It's the exact algebraic solution through **two** n=20 binomial points (C=32: 19/20; C=40: 6/20),
since every other tested C value is fully saturated (0 or 20 of 20). Propagating ordinary Wilson
CIs through those same two points gives C₅₀ ∈ [35,41] and **width ∈ [1.2, 7.3]** — a 6x range
spanning "sharp" to "fairly gradual." The 2.7x prediction inherits this softness (it's a ratio of
two soft numbers). The project's own Phase B plan already includes the C=36/44 fill-in points
that would fix this — recommend not citing C₅₀/width with 3-sig-fig confidence until B3 lands,
and reporting a bootstrap or Wilson-propagated CI alongside the point estimate once it does.

Also: the "~14 eff dim ceiling for every C≥25" framing implies a sharp onset at C=25 that the
data doesn't show — C=15 and C=20 are already at 86-97% of the ceiling value before C=25. The
saturation is gradual, not onset-at-25; worth not conflating with the genuinely sharp
convergence/accuracy wall, which *is* sharply located.

Small: `notes/2026-08-13_wall-model-and-separation-axis.md`'s self-declared "Written: 16:05 UTC"
header postdates its own commit timestamp (15:34 UTC) by 31 minutes — worth a general reminder to
generate "Written:" timestamps from the actual commit time, not aspirationally.

## F7 — real core, false instrument-validation claim

**Claim:** weight decay annihilates (not rescales) the weight bulk below the MP floor; a "robust
median-MP estimate" is "exact on synthetics across scales 0.02-2.0" and provides two-floor +
regime-flag infrastructure for handling this.

**Real, independently re-derived from raw eigenvalues in the committed JSONs:** the "annihilated,
not rescaled" distinction is genuinely demonstrated, not asserted — bulk mass fraction collapses
86%→49%→0.13% across wd 0/0.3/1 while the structural spike only drops ~8x then ~1.2x, nowhere
near proportional to the ~50x/~1000x drop in total variance. The zero-significant-dims result at
wd≥0.3 is total (640/640 data points, not cherry-picked). The separation-axis "amplitude peaks at
intermediate difficulty" corollary reproduces closely with tight error bars.

**False:** "exact on synthetics" rewrites a commit-message-only note (`8a922c2`) that actually
said "est/truth 1.00-1.11" — up to 11% error, not exact — and there is no test script, notebook,
or data file anywhere in the repo that reproduces this claim, which directly contradicts
`FINDINGS.md`'s own stated policy that "every number below is reproducible from the committed
JSONs + corpus." Worse: the validated range (0.02-2.0 scale) doesn't reach the actual regime this
finding is about — the real wd=1 corpus sits one to two orders of magnitude below that floor, and
in that regime the "robust" estimator is empirically broken: for a task with true structural rank
~7-9, it reports 29-82 "significant dimensions." This was isolated to a real mechanism (the
estimator assumes a clean rescaled-Gaussian bulk; the real annihilated bulk isn't shaped that way,
likely from near-zero/dead-unit weight rows under extreme decay) — not just a discrepancy.

**Why the finding survives anyway:** F7's own stated fallback — "use rank metrics, not MP counts,
in the depleted regime" — happens to route around the broken instrument entirely. The headline
conclusion doesn't depend on the scaled estimator working. But the estimator itself is currently
described as validated infrastructure when it demonstrably isn't in its intended use case.

**Fix:** correct "exact" to "1.00-1.11 (up to 11% error)" and cite where that's reproducible from
(or add the missing test script if it's meant to be a real, standing check); extend the validated
scale range to actually cover the depleted regime or explicitly scope the claim to say it doesn't;
add the empirical failure mode (near-degenerate real bulk ≠ clean rescaled-Gaussian bulk model) to
`manifold_detector.py`'s docstring alongside the existing "&gt;half the spectrum is structure"
caveat, since it's a different failure mode than the one currently documented there.

---

## For whoever picks this up

None of this touches the underlying corpus or training infrastructure — no bug was found in
`train/`, and F2/F3's problems are in analysis/write-up, not data collection. The fastest path to
a trustworthy `FINDINGS.md`: fix F3's census null-calibration first (it's the root cause behind
part of F3 and arguably explains some of F6's "gradual not sharp" onset framing too), regenerate
`activation_census.json`, then rewrite F2 and F3 from current data rather than patching the prose
in place. F1/F4/F5's fixes are pure wording/scoping — no recompute needed. F6/F7 just need honest
uncertainty language until B3 lands and the estimator gets its scale range extended, respectively.
