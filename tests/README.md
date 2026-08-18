# Test suite

Unit tests for the pure-numpy/torch instrumentation and infrastructure code —
the modules `FINDINGS.md`'s seven findings and the `notes/2026-08-13_external-audit-f1-f7.md`
audit sit on top of. Written after the fact ("better late than never"); scoped
to what can run without the corpus (private HF dataset) or a GPU.

## Run

```
pip install -r tests/requirements.txt
pytest
```

No GPU required anywhere in this suite. Every training/network test uses
tiny widths/depths/step counts on CPU — that's adequate to check the
architecture and math are correct, but is not a substitute for running the
real corpus-scale sweeps (`train/phase_a.py`, `train/weekend_sweep.py`, etc.),
which do want a GPU for throughput, not correctness.

## What's covered

| File | Covers | Notes |
|---|---|---|
| `test_manifold_detector.py` | `census/manifold_detector.py` | MP noise floor (fixed + self-calibrated), participation ratio, planted-rank recovery, concept coverage, the `center`/`noise_variance` adaptations from `PROVENANCE.md` |
| `test_analytic_vacuum.py` | `null_baseline/analytic_vacuum.py` | ReLU Hermite/Mehler moment math; end-to-end prediction checked against Monte Carlo at the exact config in `null_baseline/stage1_results.json` |
| `test_chain_state_keyed.py` | `null_baseline/chain_state_keyed.py` | `step_np`, `state_basis` (the q rank-collapse clock), `corrections`/`zero_theta`, the committed `port_coefs.b64.txt`; cross-checks its Hermite coefficients against `analytic_vacuum.py`'s independent implementation of the same math |
| `test_gmm_task.py` | `train/gmm_task.py` | The load-bearing claim that the GMM family's aggregate input matches the null in mean and covariance exactly, by construction (the test asserts moments — as a mixture, the full distribution is not Gaussian; see draft v0.10) |
| `test_activation_census.py` | `census/run_activation_census.py` | `forward_census` pinned field-by-field to a hand-computed 2-layer network (q_pre, dead units, centered spectrum, PR, self-calibrated MP count, anchored counts) — §4.3's floor and §4.2's activation-side corroboration share this path; also regression-documents the top-20 right-censoring of anchored counts (FINDINGS F3 correction, 2026-08-18) |
| `test_subspace_overlap.py` | `census/directional_consistency.py`, `census/eigenspace_overlap.py`, `census/procrustes_overlap.py` | Subspace-overlap and orthogonal-Procrustes-recovery helpers behind F5, using synthetic data (chance-level overlap of random subspaces, exact recovery of a planted rotation) |
| `test_train_mlp.py` | `train/train_mlp.py` (`ArcMLP`), `train/train_mlp_batched.py` (`init_stacked`) | Architecture spec (bias-free, He-Gaussian, ReLU-every-layer), forward-convention consistency with the null baseline, and the "batched trainer init is bit-identical to sequential" claim from `FINDINGS.md` |
| `test_corpus_io.py` | `train/corpus_io.py` | Local filesystem logic only (empty-run errors, existing-file lookup) — the actual HF upload/download paths are not covered |

## What's intentionally NOT covered

- **Anything that needs the corpus** (`corpus/`) or the private HF dataset
  `james-ra-henry/MZC-Corpus`: the `main()` entry points of
  `census/*.py` and `null_baseline/chain_port_refit*.py`, `train/phase_a.py`,
  `train/phase_b.py`, `train/weekend_sweep.py`. These are corpus-scale
  orchestration scripts, not units — a mocked-HF integration test is possible
  later but wasn't attempted here.
- `train/mnist_task.py` — needs `torchvision` + a real MNIST/Fashion-MNIST
  download on first use; not mocked.
- `census/feature_tracker.py`, `census/wild_census.py`,
  `census/transition_curve.py`, `census/sweep_summary.py`,
  `census/q12_tail_test.py` — post-hoc analysis/reporting scripts over
  already-computed census JSON, lower risk than the instruments that produce
  the JSON in the first place; left for a follow-up pass.
- `null_baseline/chain_port_refit.py` / `chain_port_refit_freshval.py` — same
  math family as `chain_state_keyed.py` (already covered) but still coupled
  to the HF parquet atlas at import time for their fit loop.

## GPU

**No test here needs a GPU.** If you want to extend this suite toward
actually retraining or re-running the corpus-scale sweeps (as opposed to the
tiny architecture/math smoke tests above), that's a different, much heavier
job — flag it and run it on a GPU box, not in this suite.
