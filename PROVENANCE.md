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
| `null_baseline/chain_state_keyed.py` | `arc-whitebox-canary/moment_chain/` | `b6e75e1` | State-keyed stabilizer (depth-generic successor to the `_LC` table): `step_np` = Hermite (mean, cov) propagation, `state_basis` = (q, abar, asd, rbar) descriptors. Fitting/eval paths still use the HF atlas; `step_np`/`state_basis` are pure numpy and used by `state_trajectory.py`. |
| `null_baseline/chain_port_refit_freshval.py` | `arc-whitebox-replication/moment_chain/` | `a4112f1` | Stabilized moment chain, eval-only: `step_np` = per-layer (mean, cov) propagation; `run_baseline` = uncorrected pure-analytic chain. Loads nets from HF parquet cache (to be replaced with local generation). |
| `null_baseline/chain_port_refit.py` | `arc-whitebox-replication/moment_chain/` | `a4112f1` | Same math + the sequential per-layer coefficient fitting loop. Needed to RE-FIT stabilizer coefficients for new populations — the frozen coefficients are only valid for random He-Gaussian 256×32 nets (WRITEUP §8). Side effect: writes `port_coefs.b64.txt` at import. |
| `null_baseline/port_coefs.b64.txt` | `arc-whitebox-replication/moment_chain/` | `a4112f1` | Frozen stabilizer coefficients (512 float64, base64), fitted on 40 random atlas nets at 256×32. Random-ensemble null ONLY. |

## Adaptations applied in MZC (files no longer verbatim)

*Updated: 2026-08-10 13:35 UTC*

- `manifold_detector.py`: added `noise_variance` (explicit null σ² for the MP
  threshold, plumbed through to `_mp_upper_edge`; for He-init weights pass
  `2/fan_in`) and `center` (column mean-centering now optional; the uncentered
  path uses the raw second-moment matrix since `np.cov` re-centers) to
  `_layer_census` and `layer_manifold_census`. Defaults preserve original
  behavior exactly (regression-checked). Motivating check: a uniformly ×2
  weight matrix shows 0 significant dims under self-calibrated MP but 26 under
  the analytic null.

## Known adaptations pending (files still verbatim)

> **Successor note (2026-08-10):** `arc-whitebox-canary` commit `b6e75e1` adds a
> *state-keyed* stabilizer (`moment_chain/chain_state_keyed.py`, 128 params keyed
> to measured chain state instead of the 512-entry layer-index `_LC` table) —
> depth-generic (validated at depths 48/64) and better on "atypical state at a
> typical layer" nets, which is exactly the trained-net regime (our trained L0
> occupies a state random nets only reach ~L30). When MZC needs the stabilized
> chain on trained weights, vendor that instead of parameterizing item 3 below.

3. `chain_port_refit*.py`: drop the HF parquet atlas dependency (hardcoded
   `~/.cache/huggingface/...` globs) in favor of local `build_mlp` /
   corpus-loaded nets; parameterize the hardcoded `256` / `range(32)`.
4. `stage1_validate.py`: imports `analytic_vacuum` as a bare sibling module —
   fine while everything runs from `null_baseline/`, revisit if MZC grows a
   package structure.
