"""How much of a trained MZC net is still its random initialisation?  W = alpha*W0 + Delta, per layer.

Written 2026-10-20.  Weight-space only (no inference).  Each corpus .npz stores w<i> (trained) and init_w<i>.
Per layer: alpha = <W,W0>/<W0,W0>; delta_rel = ||Delta||_F / ||alpha W0||_F; Delta's spectral structure:
participation ratio of sigma^2 (effective rank) and the energy in its top-8 singular directions; and the
fraction of trained-W energy above the Marchenko-Pastur edge of the random part (outlier singular values).
usage: mzc_weight_decomposition.py OUT.json NPZ [NPZ...]
"""
import json, os, sys
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
out, files = sys.argv[1], sys.argv[2:]
res = {}
for f in files:
    z = np.load(f); name = os.path.relpath(f).split("corpus/")[-1]
    rows = []
    for l in range(16):
        W = z[f"w{l}"].astype(np.float64); W0 = z[f"init_w{l}"].astype(np.float64)
        alpha = float((W * W0).sum() / (W0 * W0).sum())
        D = W - alpha * W0
        s = np.linalg.svd(D, compute_uv=False); e = s ** 2
        sW = np.linalg.svd(W, compute_uv=False)
        sig = float(np.sqrt((alpha * W0).var()))                      # entry scale of the surviving random part
        edge = sig * 2 * np.sqrt(1024)                                # MP edge for a 1024x1024 iid matrix
        rows.append(dict(alpha=alpha, delta_rel=float(np.linalg.norm(D) / np.linalg.norm(alpha * W0)),
                         delta_pr=float(e.sum() ** 2 / (e ** 2).sum()), delta_top8=float(e[:8].sum() / e.sum()),
                         n_outliers=int((sW > 1.02 * edge).sum()), outlier_energy=float((sW[sW > 1.02 * edge] ** 2).sum() / (sW ** 2).sum())))
    res[name] = rows
json.dump(res, open(out, "w"), indent=1)
for name, rows in res.items():
    print(name, " ".join(f"L{l}:a={r['alpha']:.2f},d={r['delta_rel']:.2f},pr={r['delta_pr']:.0f},o={r['n_outliers']}" for l, r in enumerate(rows) if l in (0, 7, 15)))
