# Vendored Code Provenance

*Written: 2026-08-10 13:05 UTC*

MZC is internally reliant: all reused code is vendored here byte-verbatim and
updated **here** from now on. The copies below are the canonical MZC versions;
the source repos are provenance only. If code migrates back out later, fine —
but MZC must build and run from this repo alone.

All source repos are authored by jamesrahenry, so there is no license friction;
`arc-whitebox-replication` is additionally MIT-licensed.

| File | Source | Commit | Notes |
|---|---|---|---|
| `census/manifold_detector.py` | `~/Source/rosetta_analysis/caz/` | `0606a36` | AMC spectral census: eigenspectrum, participation ratio, MP threshold. numpy-only. Entry: `layer_manifold_census`. |
| `census/feature_tracker.py` | `~/Source/rosetta_analysis/caz/` | `0606a36` | Cross-layer feature tracking over census PC directions (`store_directions=True` required upstream). numpy-only. |
| `null_baseline/analytic_vacuum.py` | `arc-whitebox-canary/amc_vacuum/` | `56bf71b` | Zero-forward-pass weights → activation-cov-spectrum null (k=2 Mehler mean-field). numpy-only. Entry: `predict_activation_cov_spectrum`. |
| `null_baseline/stage1_validate.py` | `arc-whitebox-canary/amc_vacuum/` | `56bf71b` | MC validation harness; also holds the canonical pure-numpy `build_mlp` (He-Gaussian `N(0, 2/fan_in)`, matches ARC `local_engine.build_mlp`). |
| `null_baseline/stage1_results.json` | `arc-whitebox-canary/amc_vacuum/` | `56bf71b` | Committed validation evidence for `analytic_vacuum.py` (eig relerr ~0.008 at L0 growing to ~0.11 by L2 at width 64 / depth 8). |
| `null_baseline/chain_port_refit_freshval.py` | `arc-whitebox-replication/moment_chain/` | `a4112f1` | Stabilized moment chain, eval-only: `step_np` = per-layer (mean, cov) propagation; `run_baseline` = uncorrected pure-analytic chain. Loads nets from HF parquet cache (to be replaced with local generation). |
| `null_baseline/chain_port_refit.py` | `arc-whitebox-replication/moment_chain/` | `a4112f1` | Same math + the sequential per-layer coefficient fitting loop. Needed to RE-FIT stabilizer coefficients for new populations — the frozen coefficients are only valid for random He-Gaussian 256×32 nets (WRITEUP §8). Side effect: writes `port_coefs.b64.txt` at import. |
| `null_baseline/port_coefs.b64.txt` | `arc-whitebox-replication/moment_chain/` | `a4112f1` | Frozen stabilizer coefficients (512 float64, base64), fitted on 40 random atlas nets at 256×32. Random-ensemble null ONLY. |

## Known adaptations pending (tracked, not yet applied — files are verbatim)

1. `manifold_detector.py`: plumb an explicit noise variance through to
   `_mp_upper_edge` (it accepts `variance` but `_layer_census` never passes it;
   currently self-calibrates σ² from the observed mean eigenvalue — not
   comparable across corpus members). For He-init weights the analytic null is
   `σ² = 2/fan_in`.
2. `manifold_detector.py`: make column mean-centering optional (baked in at the
   spectrum step; for trained weight matrices the mean-row direction may itself
   be learned signature).
3. `chain_port_refit*.py`: drop the HF parquet atlas dependency (hardcoded
   `~/.cache/huggingface/...` globs) in favor of local `build_mlp` /
   corpus-loaded nets; parameterize the hardcoded `256` / `range(32)`.
4. `stage1_validate.py`: imports `analytic_vacuum` as a bare sibling module —
   fine while everything runs from `null_baseline/`, revisit if MZC grows a
   package structure.
