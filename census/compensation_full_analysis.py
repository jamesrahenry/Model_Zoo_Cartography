"""
compensation_full_analysis.py -- analysis for the full-layer sweep
(train/compensation_full_layer_sweep.py).

Same method as census/compensation_analysis.py (Phase 1), generalized over
which layer was frozen: for each of the 32 freeze arms, z-score every
non-frozen layer's effective_dim against the baseline arm's (16 seeds,
comp_p1_baseline_c10) own null band. Assembles the full disruption-radius
matrix: rows = which layer was frozen, columns = which layer's z-score.

Question this answers: is Phase 1's finding (freezing L0 disrupts layers 1-2
only, nothing deeper) an L0-specific artifact, or does freezing ANY layer
produce the same short, bounded disruption radius?

Usage: python census/compensation_full_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_census import load_net, census_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from corpus_io import net_paths

BASELINE_RUN = "comp_p1_baseline_c10"
DEPTH = 32


def per_net_eff_dims(run_id: str) -> np.ndarray:
    rows = []
    for npz_path in net_paths(run_id):
        _, final_w = load_net(npz_path)
        census = census_weights(final_w)
        rows.append([layer["effective_dim"] for layer in census])
    return np.array(rows)  # (n_seeds, n_layers)


def main() -> None:
    print(f"loading baseline ({BASELINE_RUN})...", flush=True)
    baseline_arr = per_net_eff_dims(BASELINE_RUN)
    baseline_mean = baseline_arr.mean(axis=0)
    baseline_std = baseline_arr.std(axis=0, ddof=1)

    z_matrix = np.full((DEPTH, DEPTH), np.nan)  # [frozen_layer, measured_layer]
    disruption_radius = {}

    for frozen in range(DEPTH):
        run_id = f"comp_full_freeze_l{frozen:02d}_c10"
        print(f"loading {run_id}...", flush=True)
        try:
            arr = per_net_eff_dims(run_id)
        except FileNotFoundError:
            print(f"  !! {run_id} not found, skipping")
            continue
        mean = arr.mean(axis=0)
        for l in range(DEPTH):
            if l == frozen:
                continue
            z = (mean[l] - baseline_mean[l]) / baseline_std[l] if baseline_std[l] > 0 else np.nan
            z_matrix[frozen, l] = z
        # disruption radius: furthest layer (by |index - frozen|) with |z|>3
        shifted = [l for l in range(DEPTH) if l != frozen
                   and not np.isnan(z_matrix[frozen, l]) and abs(z_matrix[frozen, l]) > 3]
        radius = max((abs(l - frozen) for l in shifted), default=0)
        disruption_radius[frozen] = {"shifted_layers": shifted, "radius": radius}

    print(f"\n{'frozen':>6}  {'shifted layers (|z|>3)':<40} {'radius':>6}")
    for frozen in range(DEPTH):
        if frozen not in disruption_radius:
            continue
        info = disruption_radius[frozen]
        print(f"{frozen:>6}  {str(info['shifted_layers']):<40} {info['radius']:>6}")

    radii = [v["radius"] for v in disruption_radius.values()]
    print(f"\nradius stats: mean={np.mean(radii):.2f}  max={np.max(radii)}  "
          f"median={np.median(radii):.1f}")
    print("(radius 0 = no non-frozen layer cleared |z|>3 -- intervention fully absorbed locally)")

    out = {
        "baseline_run": BASELINE_RUN,
        "z_matrix": z_matrix.tolist(),
        "disruption_radius": disruption_radius,
    }
    out_path = REPO_ROOT / "census" / "compensation_full_layer_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
