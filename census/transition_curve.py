"""
transition_curve.py — Convergence fraction vs task size C at fixed budget.

Reads only corpus provenance JSONs (no weights). For each GMM family at
sep=3.0 / 20k steps / lr 3e-4, reports the fraction of seeds that converged
(within 0.03 of Bayes) and the mean accuracy gap to Bayes — the empirical
crossing profile of the depth-32 expressivity wall.

Usage: python census/transition_curve.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"


def fit_wall(C: np.ndarray, conv_frac: np.ndarray, n: np.ndarray,
             n_boot: int = 2000, seed: int = 0) -> dict:
    """Two-parameter logistic MLE for convergence fraction vs C, with
    parametric-bootstrap CIs (resample per-C binomial counts, refit)."""
    from scipy.optimize import minimize

    def fit_once(k_obs):
        def nll(p):
            c50, s = p
            q = 1 / (1 + np.exp((C - c50) / max(s, 1e-3)))
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return -np.sum(k_obs * np.log(q) + (n - k_obs) * np.log(1 - q))
        r = minimize(nll, [np.median(C), 3.0], method="Nelder-Mead")
        return r.x

    k = np.round(conv_frac * n)
    c50, s = fit_once(k)
    rng = np.random.default_rng(seed)
    boots = []
    p_hat = np.clip(k / n, 1e-6, 1 - 1e-6)
    for _ in range(n_boot):
        boots.append(fit_once(rng.binomial(n.astype(int), p_hat)))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    informative = int(np.sum((k > 0) & (k < n)))
    return {"c50": float(c50), "width": float(s),
            "c50_ci": [float(lo[0]), float(hi[0])],
            "width_ci": [float(lo[1]), float(hi[1])],
            "n_informative": informative, "n_boot": n_boot}


def main() -> None:
    rows = []
    for run_dir in sorted(CORPUS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        provs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("net_*.json"))]
        if not provs:
            continue
        t = provs[0]["task"]
        tr = provs[0]["training"]
        arch = provs[0]["architecture"]
        # the C-transition slice: GMM, sep 3.0, 20k steps, no weight decay,
        # STANDARD architecture (width/depth sweeps are separate curves)
        if (t["family"] != "gmm" or t.get("separation") != 3.0
                or tr["steps"] != 20000 or tr.get("weight_decay", 0) > 0
                or arch["width"] != 256 or arch["depth"] != 32):
            continue
        accs = np.array([p["final_val_acc"] for p in provs])
        conv = np.mean([p["outcome"] == "converged" for p in provs])
        rows.append({"run_id": run_dir.name, "C": t["n_classes"], "n": len(provs),
                     "converged_frac": round(float(conv), 3),
                     "bayes": t["bayes_accuracy"],
                     "acc_mean": round(float(accs.mean()), 4),
                     "gap_to_bayes_mean": round(float(t["bayes_accuracy"] - accs.mean()), 4),
                     "gap_to_bayes_std": round(float(accs.std()), 4)})

    rows.sort(key=lambda r: r["C"])
    print(f"{'C':>4} {'n':>4} {'conv_frac':>10} {'acc_mean':>9} {'bayes':>7} "
          f"{'gap':>7} {'gap_std':>8}")
    for r in rows:
        print(f"{r['C']:>4} {r['n']:>4} {r['converged_frac']:>10.2f} "
              f"{r['acc_mean']:>9.4f} {r['bayes']:>7.4f} "
              f"{r['gap_to_bayes_mean']:>7.4f} {r['gap_to_bayes_std']:>8.4f}")

    # Logistic wall fit with parametric-bootstrap CIs (audit item: the fit
    # must live in the repo, with uncertainty — the point estimate rests on
    # however many informative (non-saturated) binomial points exist).
    fit = fit_wall(np.array([r["C"] for r in rows]),
                   np.array([r["converged_frac"] for r in rows]),
                   np.array([r["n"] for r in rows]))
    print(f"\nwall fit: C50 = {fit['c50']:.1f} [{fit['c50_ci'][0]:.1f}, "
          f"{fit['c50_ci'][1]:.1f}], width = {fit['width']:.1f} "
          f"[{fit['width_ci'][0]:.1f}, {fit['width_ci'][1]:.1f}] "
          f"({fit['n_informative']} informative points, "
          f"{fit['n_boot']} bootstrap draws)")

    out = CENSUS_DIR / "transition_curve.json"
    out.write_text(json.dumps(
        {"slice": "gmm, sep=3.0, 20k steps, no wd",
         "rows": rows, "wall_fit": fit,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
