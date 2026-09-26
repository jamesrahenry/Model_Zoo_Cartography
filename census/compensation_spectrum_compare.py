"""
compensation_spectrum_compare.py -- is the eigenvalue SPECTRUM shared even
where the eigenVECTORS aren't?

Follow-up to the k-sweep (compensation_procrustes_ksweep.py): shared
directional overlap decays past k~20-50, but that only tests whether a
single rigid rotation aligns the whole k-dim block at once. James's
hypothesis: the "recipe" (how much variance sits at each rank) might be
shared -- the same growth rule -- while the specific directions it's
embedded into are net-specific -- a different crystal each time. A rigid
Procrustes rotation requires the WHOLE block's mutual relationships to
align simultaneously; it can miss "same kinds of local structure recur,
different relative arrangement" even when real shared vocabulary exists.

This tests the recipe directly: per layer, per net, the FULL sorted
eigenvalue spectrum of the (centered) activation covariance, normalized to
sum to 1 (shape only, not absolute scale) -- averaged over the 16 seeds in
each arm, then compared baseline vs freeze_l0 curve-by-curve (Pearson r,
mean abs difference per rank). No pairwise fitting needed (no Procrustes
rotation, no fit/test split) -- much cheaper than the k-sweep, since
"does the shape match" doesn't require aligning WHICH directions carry it.

Usage: python census/compensation_spectrum_compare.py [--n-samples 4096]
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


def forward_all_layers(weights: list[np.ndarray], x: np.ndarray) -> list[np.ndarray]:
    h = np.asarray(x, dtype=np.float64)
    acts = []
    for W in weights:
        h = np.maximum(h @ np.asarray(W, dtype=np.float64), 0.0)
        acts.append(h)
    return acts


def normalized_spectrum(a: np.ndarray) -> np.ndarray:
    """Full sorted eigenvalue spectrum of the centered covariance,
    normalized to sum to 1 -- shape only, absolute scale divided out."""
    ac = a - a.mean(axis=0)
    eigs = np.linalg.eigvalsh(ac.T @ ac / (len(ac) - 1))[::-1]
    eigs = np.maximum(eigs, 0.0)
    total = eigs.sum()
    return eigs / total if total > 0 else eigs


def load_weights(run_id: str) -> list[list[np.ndarray]]:
    out = []
    for p in net_paths(run_id):
        d = np.load(p)
        n_layers = sum(1 for f in d.files if f.startswith("init_w"))
        out.append([d[f"w{i}"] for i in range(n_layers)])
    return out


def per_net_spectra(acts: list[list[np.ndarray]]) -> np.ndarray:
    """(n_nets, n_layers, width) normalized spectra."""
    n_layers = len(acts[0])
    out = []
    for net_acts in acts:
        out.append([normalized_spectrum(net_acts[l]) for l in range(n_layers)])
    return np.array(out)


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

    print("forward pass + spectra, both arms...", flush=True)
    b_acts = [forward_all_layers(w, x_task) for w in bw]
    f_acts = [forward_all_layers(w, x_task) for w in fw]
    b_spectra = per_net_spectra(b_acts)   # (16, n_layers, 256)
    f_spectra = per_net_spectra(f_acts)

    n_layers = b_spectra.shape[1]
    b_mean = b_spectra.mean(axis=0)   # (n_layers, 256)
    f_mean = f_spectra.mean(axis=0)

    results = {}
    print(f"\n{'layer':>7} {'pearson r':>10} {'mean |diff|':>12} {'max |diff|':>11}")
    for l in range(n_layers):
        r = float(np.corrcoef(b_mean[l], f_mean[l])[0, 1])
        diff = np.abs(b_mean[l] - f_mean[l])
        results[l] = {"pearson_r": round(r, 4), "mean_abs_diff": round(float(diff.mean()), 6),
                     "max_abs_diff": round(float(diff.max()), 6)}
        marker = "  <- frozen" if l == 0 else ("  <- shift zone" if l in (1, 2) else "")
        print(f"{l:>7} {r:>10.4f} {diff.mean():>12.6f} {diff.max():>11.6f}{marker}")

    out_path = CENSUS_DIR / "compensation_spectrum_compare_results.json"
    out_path.write_text(json.dumps(
        {"width": WIDTH, "n_baseline": len(bw), "n_freeze_l0": len(fw),
         "per_layer": results,
         "baseline_mean_spectrum": b_mean.tolist(),
         "freeze_l0_mean_spectrum": f_mean.tolist(),
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}")

    deep_r = np.mean([results[l]["pearson_r"] for l in range(4, n_layers)])
    print(f"\nLayers 4+ mean Pearson r (spectrum shape match): {deep_r:.4f}")
    print("Example curves at layer 16, ranks 0-14 (top of the spectrum):")
    print(f"  baseline: {[round(x,4) for x in b_mean[16][:15]]}")
    print(f"  freeze_l0: {[round(x,4) for x in f_mean[16][:15]]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-samples", type=int, default=4096)
    args = p.parse_args()
    main()
