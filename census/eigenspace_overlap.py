"""
eigenspace_overlap.py — Does same-task subspace sharing survive at depth in
activation-covariance eigenspace? (P4 cross-pollination question,
notes/2026-08-11_p4-eigengap-crosspollination.md §2.)

P4 found cross-family LLM top-k activation-eigenspace overlap at exact k/d
chance — all shared geometry was Procrustes-recoverable only. MZC's twins
share a *task* (and at L0, a weight subspace). This measures, per family and
per layer, the pairwise top-k activation-covariance eigenvector overlap
across seeds, in raw neuron coordinates:

  - same fixed task-input sample X for every seed in a family
  - per net, per layer: top-k eigenvectors of the post-ReLU activation
    covariance (k = C-1, the task rank)
  - overlap(U_i, U_j) = ||U_i^T U_j||_F^2 / k, all seed pairs
  - init-net control (must sit at chance) and chance = k/width

Interpretation: above-chance at depth ⇒ same-task training aligns actual
neuron bases (sharing = task/data). At chance ⇒ task alignment is a
weight-space (L0) phenomenon and depth geometry is rotation-private per net,
matching P4's LLM result.

Usage: python census/eigenspace_overlap.py --run-id probe_d32_c10_head [...]
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
from corpus_io import net_paths

sys.path.insert(0, str(REPO_ROOT / "census"))
from run_activation_census import get_task

CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"


def layer_eigvecs(weights: list[np.ndarray], x: np.ndarray, k: int) -> list[np.ndarray]:
    """Top-k eigenvectors of post-ReLU activation covariance, per layer."""
    h = np.asarray(x, dtype=np.float64)
    out = []
    for W in weights:
        h = np.maximum(h @ np.asarray(W, dtype=np.float64), 0.0)
        hc = h - h.mean(axis=0)
        _, _, Vt = np.linalg.svd(hc, full_matrices=False)
        out.append(Vt[:k].T)  # [width, k]
    return out


def pairwise_overlap(units: list[list[np.ndarray]], k: int) -> list[float]:
    """Mean pairwise ||U_i^T U_j||_F^2 / k per layer, across nets."""
    n = len(units)
    n_layers = len(units[0])
    per_layer = []
    for l in range(n_layers):
        vals = [float(np.sum((units[i][l].T @ units[j][l]) ** 2) / k)
                for i in range(n) for j in range(i + 1, n)]
        per_layer.append(float(np.mean(vals)))
    return per_layer


def main() -> None:
    rng = np.random.default_rng(args.sample_seed)
    results = {}
    for run_id in args.run_ids:
        run_dir = CORPUS_DIR / run_id
        paths = net_paths(run_id)[:args.max_nets]
        prov = json.loads((run_dir / f"{paths[0].stem}.json").read_text())
        width = 256
        task = get_task(prov["task"], width)
        C = prov["task"]["n_classes"]
        k = max(C - 1, 1)
        x, _ = task.sample(args.n_samples, rng)  # one shared sample per family
        x = x.astype(np.float64)

        trained_u, init_u = [], []
        for p in paths:
            d = np.load(p)
            n_layers = sum(1 for key in d.files if key.startswith("init_w"))
            trained_u.append(layer_eigvecs([d[f"w{i}"] for i in range(n_layers)], x, k))
            init_u.append(layer_eigvecs([d[f"init_w{i}"] for i in range(n_layers)], x, k))
            print(f"{run_id}/{p.stem} done", flush=True)

        results[run_id] = {
            "C": C, "k": k, "n_nets": len(paths), "chance": k / width,
            "trained_overlap": [round(v, 5) for v in pairwise_overlap(trained_u, k)],
            "init_overlap": [round(v, 5) for v in pairwise_overlap(init_u, k)],
        }

    out_path = CENSUS_DIR / "eigenspace_overlap.json"
    out_path.write_text(json.dumps(
        {"question": "P4 §2: does same-task sharing survive at depth in raw "
                     "activation eigenspace?",
         "n_samples": args.n_samples, "runs": results,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    sel = [0, 1, 2, 4, 8, 16, 24, 31]
    for run_id, r in results.items():
        lsel = [l for l in sel if l < len(r["trained_overlap"])]
        print(f"== {run_id} (C={r['C']}, k={r['k']}, chance={r['chance']:.4f}) ==")
        print("layer     " + " ".join(f"L{l:<7}" for l in lsel))
        print("trained   " + " ".join(f"{r['trained_overlap'][l]:<8.4f}" for l in lsel))
        print("init      " + " ".join(f"{r['init_overlap'][l]:<8.4f}" for l in lsel))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", dest="run_ids", action="append", required=True)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--sample-seed", type=int, default=99)
    p.add_argument("--max-nets", type=int, default=20)
    args = p.parse_args()
    main()
