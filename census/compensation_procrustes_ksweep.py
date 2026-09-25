"""
compensation_procrustes_ksweep.py -- how much of the deep code is shared?

Follow-up to compensation_procrustes.py: that test used k=9 (C-1, the task-
label direction) and found cross_arm recovers ~baseline_twins-level overlap
at depth -- but 9 dimensions is a small slice of what's actually "active"
(effective_dim there is ~119-121 out of 256). This sweeps k to find where
shared structure ends and private/idiosyncratic structure begins: k in
{9, 20, 50, 100, 120, 150, 200}, same three conditions (baseline_twins,
freeze_l0_twins, cross_arm), same task input.

Efficiency note: neither the per-net top-eigenvector SVD nor the per-pair
Procrustes rotation R depends on k (numpy's full_matrices=False SVD already
returns all min(n,d) components; slicing to k is free). So the k-sweep
costs barely more than the original single-k=9 run -- compute the full-rank
SVD and R once per net/pair/layer, then slice for every k in one pass,
instead of repeating the expensive SVDs per k.

Usage: python census/compensation_procrustes_ksweep.py [--n-samples 4096]
"""

from __future__ import annotations

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

BASELINE_RUN = "comp_p1_baseline_c10"
FREEZE_RUN = "comp_p1_freeze_l0_c10"
WIDTH = 256
K_SWEEP = [9, 20, 50, 100, 120, 150, 200]
MAX_K = max(K_SWEEP)


def forward_all_layers(weights: list[np.ndarray], x: np.ndarray) -> list[np.ndarray]:
    h = np.asarray(x, dtype=np.float64)
    acts = []
    for W in weights:
        h = np.maximum(h @ np.asarray(W, dtype=np.float64), 0.0)
        acts.append(h)
    return acts


def top_eigvecs_full(a: np.ndarray, max_k: int) -> np.ndarray:
    """All min(n,d) eigvecs computed regardless (full_matrices=False), just
    keep the top max_k columns -- slicing to any k <= max_k afterward is free."""
    ac = a - a.mean(axis=0)
    _, _, Vt = np.linalg.svd(ac, full_matrices=False)
    return Vt[:max_k].T


def load_weights(run_id: str) -> list[list[np.ndarray]]:
    out = []
    for p in net_paths(run_id):
        d = np.load(p)
        n_layers = sum(1 for f in d.files if f.startswith("init_w"))
        out.append([d[f"w{i}"] for i in range(n_layers)])
    return out


def split_and_cache(acts: list[list[np.ndarray]]):
    """Per net, per layer: (fit half, top-MAX_K eigvecs of test half)."""
    n_layers = len(acts[0])
    out = []
    for net_acts in acts:
        fits, tests_U = [], []
        for l in range(n_layers):
            a = net_acts[l]
            n = len(a) // 2
            fit, test = a[:n], a[n:]
            fits.append(fit)
            tests_U.append(top_eigvecs_full(test, MAX_K))
        out.append((fits, tests_U))
    return out


def pair_overlaps_all_k(fit_i: np.ndarray, fit_j: np.ndarray,
                        U_i_full: np.ndarray, U_j_full: np.ndarray) -> dict[int, float]:
    """R doesn't depend on k -- compute once, slice U_i/U_j per k."""
    u_svd, _, vt_svd = np.linalg.svd(
        (fit_i - fit_i.mean(0)).T @ (fit_j - fit_j.mean(0)))
    R = u_svd @ vt_svd
    RtUi = R.T @ U_i_full  # (width, MAX_K); slicing columns is free
    out = {}
    for k in K_SWEEP:
        out[k] = float(np.sum((RtUi[:, :k].T @ U_j_full[:, :k]) ** 2) / k)
    return out


def same_arm_condition(acts: list[list[np.ndarray]]) -> dict[int, list[float]]:
    n_layers = len(acts[0])
    n = len(acts)
    cached = split_and_cache(acts)
    per_k_layers = {k: [] for k in K_SWEEP}
    for l in range(n_layers):
        per_k_sums = {k: [] for k in K_SWEEP}
        for i in range(n):
            for j in range(i + 1, n):
                pair_res = pair_overlaps_all_k(cached[i][0][l], cached[j][0][l],
                                               cached[i][1][l], cached[j][1][l])
                for k in K_SWEEP:
                    per_k_sums[k].append(pair_res[k])
        for k in K_SWEEP:
            per_k_layers[k].append(float(np.mean(per_k_sums[k])))
    return per_k_layers


def cross_arm_condition(acts_a: list[list[np.ndarray]],
                        acts_b: list[list[np.ndarray]]) -> dict[int, list[float]]:
    n_layers = len(acts_a[0])
    cached_a = split_and_cache(acts_a)
    cached_b = split_and_cache(acts_b)
    per_k_layers = {k: [] for k in K_SWEEP}
    for l in range(n_layers):
        per_k_sums = {k: [] for k in K_SWEEP}
        for i in range(len(acts_a)):
            for j in range(len(acts_b)):
                pair_res = pair_overlaps_all_k(cached_a[i][0][l], cached_b[j][0][l],
                                               cached_a[i][1][l], cached_b[j][1][l])
                for k in K_SWEEP:
                    per_k_sums[k].append(pair_res[k])
        for k in K_SWEEP:
            per_k_layers[k].append(float(np.mean(per_k_sums[k])))
    return per_k_layers


def main() -> None:
    rng = np.random.default_rng(7)
    prov = json.loads(next((CORPUS_DIR / BASELINE_RUN).glob("net_*.json")).read_text())
    task = get_task(prov["task"], WIDTH)
    x_task, _ = task.sample(args.n_samples, rng)
    x_task = x_task.astype(np.float64)

    print(f"loading {BASELINE_RUN}...", flush=True)
    bw = load_weights(BASELINE_RUN)
    print(f"loading {FREEZE_RUN}...", flush=True)
    fw = load_weights(FREEZE_RUN)

    print("forward pass, both arms...", flush=True)
    b_acts = [forward_all_layers(w, x_task) for w in bw]
    f_acts = [forward_all_layers(w, x_task) for w in fw]

    print("baseline twins (same-arm pairs, all k)...", flush=True)
    baseline_twins = same_arm_condition(b_acts)
    print("freeze_l0 twins (same-arm pairs, all k)...", flush=True)
    freeze_l0_twins = same_arm_condition(f_acts)
    print("cross-arm (all k, THE test)...", flush=True)
    cross_arm = cross_arm_condition(b_acts, f_acts)

    n_layers = len(baseline_twins[K_SWEEP[0]])
    out_path = CENSUS_DIR / "compensation_procrustes_ksweep_results.json"
    out_path.write_text(json.dumps(
        {"k_sweep": K_SWEEP, "width": WIDTH, "n_baseline": len(bw), "n_freeze_l0": len(fw),
         "conditions": {"baseline_twins": baseline_twins,
                        "freeze_l0_twins": freeze_l0_twins,
                        "cross_arm": cross_arm},
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    deep = list(range(4, n_layers))
    print(f"{'k':>5} {'chance':>8} {'baseline_twins(4+)':>19} "
          f"{'freeze_l0_twins(4+)':>20} {'cross_arm(4+)':>14}")
    for k in K_SWEEP:
        chance = k / WIDTH
        bt = float(np.mean([baseline_twins[k][l] for l in deep]))
        ft = float(np.mean([freeze_l0_twins[k][l] for l in deep]))
        ca = float(np.mean([cross_arm[k][l] for l in deep]))
        print(f"{k:>5} {chance:>8.4f} {bt:>19.4f} {ft:>20.4f} {ca:>14.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-samples", type=int, default=4096)
    args = p.parse_args()
    main()
