# MZC next steps — full outline

*Written: 2026-08-13 15:24 UTC by claude:model-zoo-cartography. State at writing:
Phase A complete (401 nets / 22 families on MZC-Corpus), batched trainer validated
(64s/net), scale-normalized census, Procrustes/sharing story closed, AMC Q10 answered.
Hopper t4b9971d carries the running record.*

## Phase B — corpus expansion (batched trainer ready; each item is a config block)

- **B1. Width sweep** — w ∈ {64, 128, 512}, C-sweeps scaled to width, 20–32 seeds.
  Tests: C−1 law across width (should hold — simplex rank is a task property);
  does the mid-net code ceiling (~14 at w=256) scale with width — the wall theory's
  main prediction; null bands per width (vacuum + state chain are width-generic).
  Note: w=512 at B=32 is VRAM-tight on the local 4 GB — use B=16 or the A10.
- **B2. Depth sweep** — d ∈ {8, 16, 48, 64} at C=10. Tests: q-plateau persistence
  past the random fixed point (ARC E2 says the *random* trajectory saturates ~L30;
  does the trained plateau just continue?); does the wall move with depth.
- **B3. Wall microscopy** — C ∈ {36, 40, 44} × budgets {20k, 60k, 200k}. Is the
  transition sharp in C at fixed budget; does the crossing point drift with budget
  (S2's model makes a prediction here first).
- **B4. Second real-data family** — fashion-MNIST (same 784→256+ZCA pipeline; one
  flag in mnist_task). Optional: grayscale CIFAR.
- **B5. Weight-decay onset curve** — wd ∈ {0.03, 0.1, 0.3}: where does
  `bulk_regime` flip from intact to depleted (we have the endpoints, not the knee).

Scale: ~1,500–2,500 nets ≈ 30–50 local GPU-hours batched ≈ 3–5 overnights, or ~1 day
on an A10. **Decision point: local overnights vs A10 rental** (A10 only clearly wins
if the 200k-budget arm of B3 matters soon).

## M — instruments & methods debt

- **M1. Edge-aware stabilizer** — add edge-specific correction slots to the
  state-keyed refit (L0/L1/L31 are where it still fails). Feeds ARC (vacuum v2).
- **M2. Partial-learner refit populations** — mixed-population fit or per-regime
  coefficients; decide before B3 mass-produces near-wall nets.
- **M3. Feature tracker on the corpus** — vendored, never run. Cross-layer feature
  births/deaths, trained vs random; connects MZC's rank language to AMC's feature
  language.
- **M4. Rectangular-matrix MP + head/embedding handling** — prerequisite for any
  HF-model census (X3).
- **M5. Batched-trainer completeness** — MNIST sampler on GPU + AdamW/wd support
  (both small); needed before B4/B5 run batched.

## S — science on the existing corpus (no training required)

- **S1. Separation anomaly** — L0 eff dim is non-monotone in separation
  (33/28/26/36/57) while sig dims stay exactly 9. Decompose with the two-floor
  census (scale vs shape axes now separable).
- **S2. Wall transition model** — fit the gap-to-Bayes curve + variance blowup vs C;
  produce the quantitative prediction B3 then tests (candidate form: wall where
  C exceeds ~2.5× the mid-net code ceiling).
- **S3. Directional-consistency distributions** — publish 20-seed bands (currently
  pairwise means), incl. the C=50 20k-vs-60k accretion curve.
- **S4. Within-corpus atypicality** — per-net state-trajectory deviation vs training
  difficulty; the internal analogue of ARC's E3 tail result.

## X — cross-program deliverables

- **X1. Vacuum v2** (with ARC) — per-architecture-family stabilizer fits on reinit
  nets → per-layer instrument error bars = minimum detectable structure. Builds on M1.
- **X2. AMC Q12 direct test** — run census + state-trajectory instruments on ARC's
  actual hard-tail challenge nets (tooling exists; needs the HF atlas subset).
- **X3. HF pretrained models** (gated on M4) — census real small models
  (pythia-70m/160m MLP blocks first) with scaled floor + regime flag; the LLM half
  of AMC Q10.

## W — write-up & publication

- **W1. Corpus paper draft** — "task rank is imprinted in the input layer": the
  three laws, the wall, the instruments, the corpus. arXiv target.
- **W2. Public flip** — MZC-Corpus dataset card + reproduction script polish;
  flip private→public with W1.

## O — ops

- **O1. Disk** — prune ~/.cache/huggingface (~27 GB, re-downloadable) before Phase B.
- **O2. phase_b.py** — adapt the Phase A orchestrator to the batched trainer
  (per-config: batched train → census → directional → upload-prune).
- **O3. A10 + Prefect/MLflow** — only if the B config list outgrows local overnights.

## Recommended sequencing

1. **Wave 1 (now):** O1 → M5 → O2 → launch B1+B2+B3-core as overnights; S1+S2 during
   the day (they shape B3's config choices).
2. **Wave 2:** M1+M2 (stabilizer edges) alongside B4+B5 overnight; S3+S4.
3. **Wave 3:** X2, then M4 → X3 (HF models — first contact with wild-caught nets).
4. **Wave 4:** W1 draft + W2 public flip; X1 lands with ARC whenever their side is ready.
