"""
directional_consistency.py — Handoff §3's sharpened hypothesis, tested at L0.

Claim under test: training-induced structure is atypicality that is CONSISTENT
across nets trained on the same task (directionally aligned deviation), where
random-init atypicality is isotropic.

L0 is the layer where this is crisp: every net shares input coordinates, and
the task's true class-mean subspace is known exactly (the GMM construction).
For each family (same task, 3 init seeds):

  learned subspace  U_i = top-(C-1) left singular vectors of dW0 = W0_t - W0_i
                    (left = input-space directions; forward is x @ W)
  seed consistency  overlap(U_i, U_j) = ||U_i^T U_j||_F^2 / (C-1), pairwise
  task alignment    overlap(U_i, T) with T = orthonormal basis of the centered
                    class means (rank C-1)
  init control      same overlap on top-(C-1) directions of the INIT matrices
  chance            (C-1)/256 — expected overlap of random (C-1)-dim subspaces

Usage: python census/directional_consistency.py --run-id <id> [--run-id ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from gmm_task import make_gmm_task

CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"


def top_left_svs(M: np.ndarray, k: int) -> np.ndarray:
    U, _, _ = np.linalg.svd(np.asarray(M, dtype=np.float64))
    return U[:, :k]  # [dim, k], orthonormal columns


def overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean cos^2 of principal angles = ||A^T B||_F^2 / k."""
    return float(np.sum((A.T @ B) ** 2) / A.shape[1])


def main() -> None:
    rows = []
    for run_id in args.run_ids:
        run_dir = CORPUS_DIR / run_id
        provs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("net_*.json"))]
        C = provs[0]["task"]["n_classes"]
        k = max(C - 1, 1)
        t = provs[0]["task"]
        if t["family"] == "gmm":
            task = make_gmm_task(dim=256, n_classes=C, separation=t["separation"],
                                 seed=t["task_seed"])
            means = task.means
        else:  # mnist_whitened: class means of the whitened train split
            from mnist_task import make_mnist_task
            task = make_mnist_task(dim=256, seed=t["projection_seed"])
            means = np.stack([task.x_train[task.y_train == c].mean(axis=0)
                              for c in range(C)]).astype(np.float64)
        means_c = means - means.mean(axis=0)
        T = np.linalg.qr(means_c.T)[0][:, :k]  # [256, k] task subspace basis

        learned, inits = [], []
        for npz_path in sorted(run_dir.glob("net_*.npz")):
            d = np.load(npz_path)
            dW0 = d["w0"].astype(np.float64) - d["init_w0"].astype(np.float64)
            learned.append(top_left_svs(dW0, k))
            inits.append(top_left_svs(d["init_w0"], k))

        n = len(learned)
        pair_learned = [overlap(learned[i], learned[j])
                        for i in range(n) for j in range(i + 1, n)]
        pair_init = [overlap(inits[i], inits[j])
                     for i in range(n) for j in range(i + 1, n)]
        task_learned = [overlap(U, T) for U in learned]
        task_init = [overlap(U, T) for U in inits]

        rows.append({
            "run_id": run_id, "C": C, "k": k,
            "outcomes": ",".join(sorted({p["outcome"] for p in provs})),
            "seed_overlap_learned": float(np.mean(pair_learned)),
            "seed_overlap_init": float(np.mean(pair_init)),
            "task_overlap_learned": float(np.mean(task_learned)),
            "task_overlap_init": float(np.mean(task_init)),
            "chance": k / 256,
        })

    print(f"{'run':<22} {'C':>4} {'k':>4} {'outcome':>10} "
          f"{'seed(dW)':>9} {'seed(init)':>11} {'task(dW)':>9} "
          f"{'task(init)':>11} {'chance':>7}")
    for r in rows:
        print(f"{r['run_id']:<22} {r['C']:>4} {r['k']:>4} {r['outcomes']:>10} "
              f"{r['seed_overlap_learned']:>9.4f} {r['seed_overlap_init']:>11.4f} "
              f"{r['task_overlap_learned']:>9.4f} {r['task_overlap_init']:>11.4f} "
              f"{r['chance']:>7.4f}")

    out_path = CENSUS_DIR / "directional_consistency.json"
    out_path.write_text(json.dumps(
        {"rows": rows,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", dest="run_ids", action="append", required=True)
    args = p.parse_args()
    main()
