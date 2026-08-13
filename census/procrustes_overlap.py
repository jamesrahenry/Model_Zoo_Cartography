"""
procrustes_overlap.py — Is the depth code the SAME code up to rotation?

Follow-up to eigenspace_overlap.py (raw overlap = chance at depth) and the
P4 exchange: P4 recovers all cross-model geometry through a fitted Procrustes
rotation. Here we test whether MZC's same-task twins do too, with honest
train/test separation:

  per twin pair (i, j), per layer:
    - shared input sample X, split into fit / test halves
    - R = orthogonal Procrustes fit on FIT-half activations
      (R = UVᵀ from SVD(A_i_fitᵀ A_j_fit))
    - recovered overlap on TEST half: ||U_iᵀ R U_j||²_F / k with U_* = top-k
      eigenvectors of each net's test-half activation covariance

  conditions: trained twins (task input) · init twins (same input) ·
  cross-task trained pairs (c10 GMM net vs MNIST net, shared noise input)

Reading: recovered(trained twins) >> recovered(init twins) at depth means
same-task training produces a shared code that is rotation-private per net;
recovered ≈ init means even the rotational content is generic input geometry.

Usage: python census/procrustes_overlap.py [--n-nets 8] [--n-samples 4096]
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

TWIN_RUN = "probe_d32_c10_head"
CROSS_RUN = "mnist_d32"
K = 9  # C-1 for both families


def forward_all_layers(weights: list[np.ndarray], x: np.ndarray) -> list[np.ndarray]:
    h = np.asarray(x, dtype=np.float64)
    acts = []
    for W in weights:
        h = np.maximum(h @ np.asarray(W, dtype=np.float64), 0.0)
        acts.append(h)
    return acts


def top_eigvecs(a: np.ndarray, k: int) -> np.ndarray:
    ac = a - a.mean(axis=0)
    _, _, Vt = np.linalg.svd(ac, full_matrices=False)
    return Vt[:k].T


def pair_layer_metrics(a_i: np.ndarray, a_j: np.ndarray, k: int) -> tuple[float, float]:
    """(raw, procrustes-recovered) top-k eigenspace overlap, fit/test split."""
    n = len(a_i) // 2
    fit_i, test_i = a_i[:n], a_i[n:]
    fit_j, test_j = a_j[:n], a_j[n:]
    u_svd, _, vt_svd = np.linalg.svd(
        (fit_i - fit_i.mean(0)).T @ (fit_j - fit_j.mean(0)))
    R = u_svd @ vt_svd
    U_i = top_eigvecs(test_i, k)
    U_j = top_eigvecs(test_j, k)
    raw = float(np.sum((U_i.T @ U_j) ** 2) / k)
    rec = float(np.sum(((R.T @ U_i).T @ U_j) ** 2) / k)
    return raw, rec


def load_weights(run_id: str, n: int, family: str) -> list[list[np.ndarray]]:
    key = "w" if family == "trained" else "init_w"
    out = []
    for p in net_paths(run_id)[:n]:
        d = np.load(p)
        n_layers = sum(1 for f in d.files if f.startswith("init_w"))
        out.append([d[f"{key}{i}"] for i in range(n_layers)])
    return out


def condition(acts: list[list[np.ndarray]], k: int) -> list[float]:
    """Mean recovered overlap per layer over all pairs."""
    n_layers = len(acts[0])
    n = len(acts)
    rec_layers = []
    for l in range(n_layers):
        recs = []
        for i in range(n):
            for j in range(i + 1, n):
                _, rec = pair_layer_metrics(acts[i][l], acts[j][l], k)
                recs.append(rec)
        rec_layers.append(float(np.mean(recs)))
    return rec_layers


def main() -> None:
    rng = np.random.default_rng(7)
    prov = json.loads(next((CORPUS_DIR / TWIN_RUN).glob("net_*.json")).read_text())
    task = get_task(prov["task"], 256)
    x_task, _ = task.sample(args.n_samples, rng)
    x_task = x_task.astype(np.float64)
    x_noise = rng.standard_normal((args.n_samples, 256))

    tw = load_weights(TWIN_RUN, args.n_nets, "trained")
    iw = load_weights(TWIN_RUN, args.n_nets, "init")
    mw = load_weights(CROSS_RUN, args.n_nets, "trained")

    results = {}
    # sequential compute-and-free: each condition's activations are ~2 GB
    print("trained twins...", flush=True)
    acts = [forward_all_layers(w, x_task) for w in tw]
    results["trained_twins"] = condition(acts, K)
    del acts
    print("init twins...", flush=True)
    acts = [forward_all_layers(w, x_task) for w in iw]
    results["init_twins"] = condition(acts, K)
    del acts
    # cross-task pairs: i from c10, j from mnist (no same-family pairs),
    # shared NOISE input so neither net sees its own task distribution
    print("cross-task...", flush=True)
    acts_tn = [forward_all_layers(w, x_noise) for w in tw]
    acts_mn = [forward_all_layers(w, x_noise) for w in mw]
    n_layers = len(acts_tn[0])
    cross = []
    for l in range(n_layers):
        recs = [pair_layer_metrics(acts_tn[i][l], acts_mn[j][l], K)[1]
                for i in range(len(acts_tn)) for j in range(len(acts_mn))]
        cross.append(float(np.mean(recs)))
    results["cross_task_noise"] = cross

    out_path = CENSUS_DIR / "procrustes_overlap.json"
    out_path.write_text(json.dumps(
        {"k": K, "n_nets": args.n_nets, "n_samples": args.n_samples,
         "chance": K / 256, "conditions": results,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"wrote {out_path}\n")

    sel = [0, 1, 2, 4, 8, 16, 24, 31]
    print(f"Procrustes-RECOVERED top-{K} eigenspace overlap (chance {K/256:.4f})")
    print("layer              " + " ".join(f"L{l:<7}" for l in sel))
    for name, vals in results.items():
        print(f"{name:<18} " + " ".join(f"{vals[l]:<8.4f}" for l in sel))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-nets", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=4096)
    args = p.parse_args()
    main()
