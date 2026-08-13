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
        # the C-transition slice: GMM, sep 3.0, 20k steps, no weight decay
        if (t["family"] != "gmm" or t.get("separation") != 3.0
                or tr["steps"] != 20000 or tr.get("weight_decay", 0) > 0):
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

    out = CENSUS_DIR / "transition_curve.json"
    out.write_text(json.dumps(
        {"slice": "gmm, sep=3.0, 20k steps, no wd",
         "rows": rows,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
