"""
compensation_procrustes.py -- does the deep code survive freezing L0, or
just its effective_dim?

Follow-up to the full-layer sweep (train/compensation_full_layer_sweep.py):
layers 4-31 showed matching effective_dim between the baseline and freeze_l0
arms (|z|<2 throughout), which is a PRH-shaped observation (James, 2026-09-24)
-- but effective_dim is a scalar (participation ratio of the eigenspectrum),
and two representations can share it while pointing in completely different
directions. This is the real test: adapts procrustes_overlap.py's method
(F5's own PRH check, same-seed twins) to a cross-ARM comparison instead of a
cross-seed one.

  per pair (i from one arm, j from the other), per layer:
    - shared task input X, split fit/test halves
    - R = orthogonal Procrustes fit on FIT-half activations
    - recovered overlap on TEST half: ||U_i^T R U_j||^2_F / k

  conditions:
    baseline_twins   -- same-arm pairs within the 16 baseline nets (the F5-style
                        reference: does this corpus's own seed-to-seed code match?)
    freeze_l0_twins  -- same-arm pairs within the 16 freeze_l0 nets (do THEY
                        share a code among themselves, despite worse accuracy?)
    cross_arm        -- baseline net i vs freeze_l0 net j, all 16x16 pairs
                        (THE test: does the same deep code form across arms?)

Reading: cross_arm ~ baseline_twins at layers 4+ means the deep code really
is the same regardless of the L0 perturbation -- strong PRH-style evidence,
sharper than F5's (that was seed noise; this is a real functional
perturbation, freeze_l0 caps at 62-64% acc vs baseline's ~90%). cross_arm
near chance means the effective_dim match was coincidental shape-matching,
not shared content.

Usage: python census/compensation_procrustes.py [--n-samples 4096]
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

BASELINE_RUN = "comp_p1_baseline_c10"
FREEZE_RUN = "comp_p1_freeze_l0_c10"
WIDTH = 256
K = 9  # C-1 for C=10


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


def pair_layer_metrics(fit_i: np.ndarray, fit_j: np.ndarray,
                       U_i: np.ndarray, U_j: np.ndarray, k: int) -> float:
    """Recovered top-k eigenspace overlap. fit_i/fit_j: FIT-half activations
    (raw, for the pairwise rotation fit). U_i/U_j: top-k eigvecs of the
    TEST-half activations -- precomputed once per net per layer by the
    caller (independent of which pair a net appears in), not recomputed
    per pair -- avoids O(n^2) redundant SVDs on the (n_samples/2, width)
    activation matrix, the expensive part of this computation."""
    u_svd, _, vt_svd = np.linalg.svd(
        (fit_i - fit_i.mean(0)).T @ (fit_j - fit_j.mean(0)))
    R = u_svd @ vt_svd
    return float(np.sum(((R.T @ U_i).T @ U_j) ** 2) / k)


def load_weights(run_id: str) -> list[list[np.ndarray]]:
    out = []
    for p in net_paths(run_id):
        d = np.load(p)
        n_layers = sum(1 for f in d.files if f.startswith("init_w"))
        out.append([d[f"w{i}"] for i in range(n_layers)])
    return out


def split_and_cache(acts: list[list[np.ndarray]], k: int):
    """Per net, per layer: (fit half, top-k eigvecs of test half) -- computed
    once, reused across every pair that net appears in."""
    n_layers = len(acts[0])
    out = []
    for net_acts in acts:
        fits, tests_U = [], []
        for l in range(n_layers):
            a = net_acts[l]
            n = len(a) // 2
            fit, test = a[:n], a[n:]
            fits.append(fit)
            tests_U.append(top_eigvecs(test, k))
        out.append((fits, tests_U))
    return out


def same_arm_condition(acts: list[list[np.ndarray]], k: int) -> list[float]:
    n_layers = len(acts[0])
    n = len(acts)
    cached = split_and_cache(acts, k)
    rec_layers = []
    for l in range(n_layers):
        recs = [pair_layer_metrics(cached[i][0][l], cached[j][0][l],
                                   cached[i][1][l], cached[j][1][l], k)
                for i in range(n) for j in range(i + 1, n)]
        rec_layers.append(float(np.mean(recs)))
    return rec_layers


def cross_arm_condition(acts_a: list[list[np.ndarray]], acts_b: list[list[np.ndarray]],
                        k: int) -> list[float]:
    n_layers = len(acts_a[0])
    cached_a = split_and_cache(acts_a, k)
    cached_b = split_and_cache(acts_b, k)
    rec_layers = []
    for l in range(n_layers):
        recs = [pair_layer_metrics(cached_a[i][0][l], cached_b[j][0][l],
                                   cached_a[i][1][l], cached_b[j][1][l], k)
                for i in range(len(acts_a)) for j in range(len(acts_b))]
        rec_layers.append(float(np.mean(recs)))
    return rec_layers


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

    results = {}
    print("baseline twins (same-arm pairs)...", flush=True)
    b_acts = [forward_all_layers(w, x_task) for w in bw]
    results["baseline_twins"] = same_arm_condition(b_acts, K)

    print("freeze_l0 twins (same-arm pairs)...", flush=True)
    f_acts = [forward_all_layers(w, x_task) for w in fw]
    results["freeze_l0_twins"] = same_arm_condition(f_acts, K)

    print("cross-arm (baseline vs freeze_l0, THE test)...", flush=True)
    results["cross_arm"] = cross_arm_condition(b_acts, f_acts, K)

    n_layers = len(results["baseline_twins"])
    out_path = CENSUS_DIR / "compensation_procrustes_results.json"
    out_path.write_text(json.dumps(
        {"k": K, "width": WIDTH, "n_baseline": len(bw), "n_freeze_l0": len(fw),
         "chance": K / WIDTH, "conditions": results,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    print(f"Procrustes-RECOVERED top-{K} eigenspace overlap (chance {K/WIDTH:.4f})")
    print(f"{'layer':>7} {'baseline_twins':>15} {'freeze_l0_twins':>16} {'cross_arm':>10}")
    for l in range(n_layers):
        marker = "  <- frozen" if l == 0 else ("  <- Phase1 shift zone" if l in (1, 2) else "")
        print(f"{l:>7} {results['baseline_twins'][l]:>15.4f} "
              f"{results['freeze_l0_twins'][l]:>16.4f} {results['cross_arm'][l]:>10.4f}{marker}")

    deep = list(range(4, n_layers))
    cross_deep_mean = float(np.mean([results["cross_arm"][l] for l in deep]))
    twins_deep_mean = float(np.mean([results["baseline_twins"][l] for l in deep]))
    print(f"\nLayers 4+ mean: baseline_twins={twins_deep_mean:.4f}  "
          f"cross_arm={cross_deep_mean:.4f}  chance={K/WIDTH:.4f}")
    if cross_deep_mean > 0.5 * twins_deep_mean:
        print("cross_arm recovers a substantial fraction of the twins-level overlap at "
              "depth -- the deep code looks like the SAME code, not just matching shape.")
    else:
        print("cross_arm does NOT recover twins-level overlap at depth -- the effective_dim "
              "match looks like coincidental shape-matching, not shared content.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-samples", type=int, default=4096)
    args = p.parse_args()
    main()
