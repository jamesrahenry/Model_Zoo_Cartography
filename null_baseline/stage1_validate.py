"""AMC Plan B Stage 1 — validate predict_activation_cov_spectrum vs Monte Carlo.

PURE NUMPY, CPU. Builds random He-Gaussian ReLU MLPs, pushes N(0,I) inputs through
(numpy, ~10k samples), compares the EMPIRICAL per-layer activation covariance
eigenspectrum (centered, as the AMC census computes it) to the ANALYTIC prediction.

Reports, per layer:
  - relative error of the top-k eigenvalues (||pred - mc|| / ||mc||)
  - relative error of the participation ratio (effective dim)
  - relative error of total variance (trace)

Run: ../.venv/bin/python stage1_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_vacuum import predict_activation_cov_spectrum, _participation_ratio


def build_mlp(width, depth, seed=0):
    """He-initialized N(0, 2/width) square weights — matches local_engine.build_mlp.
    forward convention: x = relu(x @ W) for W in weights."""
    rng = np.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    return [rng.standard_normal((width, width)).astype(np.float64) * scale
            for _ in range(depth)]


def mc_layer_cov_spectra(weights, n_samples, seed, n_top=None):
    """Push n_samples N(0,I) inputs through relu(x@W); per layer return centered-cov
    eigenspectrum + participation ratio + trace. Matches caz census (centered SVD)."""
    width = weights[0].shape[1]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, width)).astype(np.float64)
    eig, effdim, trace = [], [], []
    for W in weights:
        x = np.maximum(x @ W, 0.0)  # post-ReLU activation a_l
        xc = x - x.mean(axis=0)
        # centered covariance eigenvalues via SVD (s^2/(n-1)), matches manifold_detector
        s = np.linalg.svd(xc, compute_uv=False)
        ev = (s * s) / (n_samples - 1)
        ev = np.sort(ev)[::-1]
        eig.append(ev if n_top is None else ev[:n_top])
        effdim.append(_participation_ratio(ev))
        trace.append(float(ev.sum()))
    return dict(eig=eig, eff_dim=effdim, trace=trace)


def rel_err(pred, mc):
    pred = np.asarray(pred, dtype=np.float64)
    mc = np.asarray(mc, dtype=np.float64)
    k = min(len(pred), len(mc))
    pred, mc = pred[:k], mc[:k]
    denom = np.linalg.norm(mc)
    return float(np.linalg.norm(pred - mc) / denom) if denom > 0 else float("nan")


def main():
    configs = [
        (64, 8, 20000),
        (128, 8, 20000),
        (256, 8, 10000),
        (256, 12, 10000),
    ]
    n_seeds = 3
    TOPK = 20  # compare the top-20 eigenvalues (what the empirical census stores)

    print("AMC Plan B Stage 1 — analytic vacuum vs Monte Carlo (pure ReLU MLP)")
    print("=" * 88)
    grand = {"eig": [], "effdim": [], "trace": []}
    for width, depth, nsamp in configs:
        print(f"\n## width={width} depth={depth} n_samples={nsamp}  (mean over {n_seeds} seeds)")
        print(f"  {'layer':>5} {'eig_relerr(top%d)'%TOPK:>18} "
              f"{'effdim_relerr':>14} {'trace_relerr':>13} "
              f"{'pred_eff':>9} {'mc_eff':>8}")
        per_layer = {}
        for seed in range(n_seeds):
            W = build_mlp(width, depth, seed=seed)
            pred = predict_activation_cov_spectrum(W, n_top=None)
            mc = mc_layer_cov_spectra(W, nsamp, seed=1000 + seed, n_top=None)
            for l in range(depth):
                e = rel_err(pred["eig_centered"][l][:TOPK], mc["eig"][l][:TOPK])
                ed = abs(pred["eff_dim_centered"][l] - mc["eff_dim"][l]) / max(mc["eff_dim"][l], 1e-9)
                tr_pred = float(np.sum(pred["eig_centered"][l]))
                tr = abs(tr_pred - mc["trace"][l]) / max(mc["trace"][l], 1e-9)
                per_layer.setdefault(l, {"eig": [], "ed": [], "tr": [],
                                         "pe": [], "me": []})
                per_layer[l]["eig"].append(e)
                per_layer[l]["ed"].append(ed)
                per_layer[l]["tr"].append(tr)
                per_layer[l]["pe"].append(pred["eff_dim_centered"][l])
                per_layer[l]["me"].append(mc["eff_dim"][l])
        for l in range(depth):
            d = per_layer[l]
            me = np.mean(d["eig"]); med = np.mean(d["ed"]); mtr = np.mean(d["tr"])
            print(f"  {l:>5} {me:>18.4e} {med:>14.4e} {mtr:>13.4e} "
                  f"{np.mean(d['pe']):>9.2f} {np.mean(d['me']):>8.2f}")
            grand["eig"].append(me); grand["effdim"].append(med); grand["trace"].append(mtr)

    print("\n" + "=" * 88)
    print("GRAND SUMMARY (mean over all layers/configs/seeds):")
    print(f"  top-{TOPK} eigenvalue rel-err : {np.mean(grand['eig']):.4e}  "
          f"(max layer {np.max(grand['eig']):.4e})")
    print(f"  effective-dim rel-err     : {np.mean(grand['effdim']):.4e}  "
          f"(max {np.max(grand['effdim']):.4e})")
    print(f"  trace (total var) rel-err : {np.mean(grand['trace']):.4e}  "
          f"(max {np.max(grand['trace']):.4e})")
    print("\nInterpretation: <~1e-2 = analytic predictor matches MC; the residual is "
          "MC sampling noise + the Gaussian-mean-field truncation (higher cumulants).")


if __name__ == "__main__":
    main()
