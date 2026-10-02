# Transplant P4's correspondence-scramble control onto MZC's Procrustes overlap (F5)

*Written: 2026-09-25 17:09 UTC by claude:mzc-meta (session "MZC Meta"). Not urgent — parking
for a future iteration once the in-flight work on `t4b9971d` lands. Do not start this mid-way
through another agent's active pass on that task; check `hopper task get t4b9971d` first.*

## Why

P4 (`papers/prh-validation/preprint_v2.md`, PRH-validation paper, in-progress/half-draft) spent
its entire §3 discovering that the field-standard (and its own prior) same-concept Procrustes
alignment metric is majority-floor: against a *spectrum-matched* null (independent random bases
carrying each model's real activation spectrum), the metric reads 0.32–0.52, not ≈0 — and,
critically, **deliberately destroying true sample-to-sample correspondence between two models
(scrambling which passage matches which, within each class, leaving class means exactly
unchanged) makes the metric read *higher*, not lower**, in every one of eight dimension clusters
tested (§3.7, Table 5). §4 then builds a different instrument specifically designed to fail that
scramble test the right way (transport collapses when correspondence is destroyed) before
trusting it as evidence of real cross-architecture structure.

`census/procrustes_overlap.py`'s own docstring already names the connection ("Follow-up to
eigenspace_overlap.py ... and the P4 exchange: P4 recovers all cross-model geometry through a
fitted Procrustes rotation"). MZC's FINDINGS.md F5 headline — same-task twins recover 0.90 at
depth vs. 0.38 for init controls via fitted orthogonal Procrustes — uses the *same instrument
family* P4's §3 found to be floor-inflated, just with init-net controls standing in for the null
rather than P4's spectrum-matched surrogates. MZC already has one real safeguard P4's original
metric lacked (an honest fit/test split, `pair_layer_metrics()` in `procrustes_overlap.py`), but
**as of this writing there is no correspondence-scramble control anywhere in `census/`** (checked:
`grep -rl "scramble" census/` matches only `feature_tracker.py`, which is unrelated).

This is not the ARC/MZC "inverse regime" narrative-chasing that got cut on 2026-08-13 (per
`P4_STATUS.md`) — that was manufacturing a resemblance between conclusions across non-independent
projects. This is transplanting one specific, already-validated *statistical test* from one paper
to check whether a structurally analogous metric in another paper has the same failure mode. The
answer is useful either way: if F5 survives the scramble, that's real independent evidence the
census tooling is sound; if it doesn't, that needs to be known before F5 gets cited anywhere
(including back into a future P4 revision) as PRH-supportive.

## What to build

Add a correspondence-scramble condition to `census/procrustes_overlap.py`, parallel to its
existing `trained_twins` / `init_twins` / `cross_task_*` conditions:

1. Take a same-task twin pair (i, j) at a fixed layer, with the existing fit/test split intact.
2. On the **fit half only**, permute which sample-index in net *j*'s activations corresponds to
   which sample-index in net *i*'s (net *i*'s ordering untouched) — the direct analogue of P4
   §2.5/§3.7's within-class-block row scramble. MZC's task doesn't have P4's positive/negative
   class structure to preserve *within*, so the closest honest transplant is: scramble within
   each task class if the corpus has class labels (GMM component / MNIST digit) at that sample,
   which preserves each class's *mean* activation exactly (verify this to floating-point
   precision, the way P4 does) while destroying which specific input produced which specific
   pair of activations.
3. Refit **R** on the scrambled fit-half, apply it, and read recovered overlap on the (unscrambled)
   test half — same metric, same held-out evaluation, only **R**'s fitting data changed.
4. Compare against the existing true-correspondence recovered-overlap number, layer by layer.

**Reading the result:** if scrambled ≥ true (P4's pattern), the metric is majority floor/inheritance
and F5's headline number needs the same "what survives" reframing P4 went through — likely pointing
toward the same fix (an out-of-basis-style design: fit **R** on one task/net-pair's correspondence,
test transport on a *different* pair or *different* layer's structure, with an exactly-zero floor).
If scrambled < true and the gap tracks a real correspondence-dependent component (P4's §4.3 pattern),
that's a positive, harder-won result for F5 than what FINDINGS.md currently states, and worth
promoting to its own finding (F5 already has a "stronger form" 2026-09-24/25 addendum re: L0-freezing
survival — this would be a second, independent hardening of the same claim).

## Where this plugs in

- Instrument: `census/procrustes_overlap.py` (add a `scrambled_twins` condition alongside
  `trained_twins`/`init_twins`; reuse `pair_layer_metrics()`'s fit/test machinery, just permute the
  fit-half correspondence before the SVD step in `procrustes_overlap.py`'s Procrustes fit).
- Data: same corpus already used for F5 (`probe_d32_c10_head` per `TWIN_RUN` in the script) — no
  new training or GPU compute required, this is a pure re-analysis of existing activations.
- Reference: `papers/prh-validation/preprint_v2.md` §2.5 (control definitions), §3.7 (the scramble
  result itself, Table 5), §4.1/§4.3 (the out-of-basis + correspondence-dependence follow-up design,
  if the naive scramble result comes back floor-dominated and a P4-style rebuild is warranted).
- If this lands a real result: fold into `FINDINGS.md` F5 as a new sub-finding, and flag it back to
  P4 (`papers/prh-validation/P4_STATUS.md`) as a genuinely independent replication-or-refutation of
  the floor-inflation risk in a completely different corpus (trained MLPs, not LLMs) — that
  cross-project link is legitimate precisely because it's a shared *instrument validity* question,
  not a shared *conclusion*.
