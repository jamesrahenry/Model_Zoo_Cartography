"""
compensation_analysis.py -- Phase 1 analysis for the distributed/compensatory-
structure experiment (notes/2026-09-22_distributed-compensatory-structure-gem-solitaire.md).

Question: does freezing layer 0 (which prevents its usual collapse to task
rank, F1) cause OTHER layers to shift beyond normal seed-to-seed variance?

Method: baseline arm (16 unconstrained seeds) gives an empirical null band
per layer -- mean +/- std of effective_dim under pure seed noise at fixed
recipe. Z-score the freeze_l0 arm's same-layer effective_dim against that
band -- same convention as F2's q-clock (z(L31) = +3.2 to +4.2 against the
random-init null band), just pointed at a trained-baseline reference instead
of random-init, per the note's resolution of the "structure vs noise" gate.

A layer whose |z| stays small is noise (nothing beyond ordinary seed
variance). A layer whose |z| is large is a candidate real compensatory
shift. Layer 0 itself is excluded from the z-scoring (trivially "changed"
by construction -- frozen at init, not a candidate for compensation).

Usage: python census/compensation_analysis.py
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
FREEZE_RUN = "comp_p1_freeze_l0_c10"
FROZEN_LAYER = 0


def per_net_eff_dims(run_id: str) -> list[list[float]]:
    """Returns [n_layers] effective_dim per net, one row per net."""
    rows = []
    for npz_path in net_paths(run_id):
        _, final_w = load_net(npz_path)
        census = census_weights(final_w)
        rows.append([layer["effective_dim"] for layer in census])
    return rows


def main() -> None:
    baseline_rows = per_net_eff_dims(BASELINE_RUN)
    freeze_rows = per_net_eff_dims(FREEZE_RUN)
    n_layers = len(baseline_rows[0])

    baseline_arr = np.array(baseline_rows)   # (n_seeds, n_layers)
    freeze_arr = np.array(freeze_rows)

    baseline_mean = baseline_arr.mean(axis=0)
    baseline_std = baseline_arr.std(axis=0, ddof=1)
    freeze_mean = freeze_arr.mean(axis=0)
    freeze_std = freeze_arr.std(axis=0, ddof=1)

    results = []
    print(f"{'layer':>5} {'baseline eff_dim':>18} {'freeze_l0 eff_dim':>18} "
          f"{'z-score':>9}  note")
    for l in range(n_layers):
        z = ((freeze_mean[l] - baseline_mean[l]) / baseline_std[l]
             if baseline_std[l] > 0 else float("nan"))
        note = "FROZEN (excluded)" if l == FROZEN_LAYER else (
            "|z|>3 -- candidate real shift" if abs(z) > 3 else
            "|z|>2 -- borderline" if abs(z) > 2 else "")
        results.append({
            "layer": l,
            "baseline_mean": round(float(baseline_mean[l]), 2),
            "baseline_std": round(float(baseline_std[l]), 2),
            "freeze_l0_mean": round(float(freeze_mean[l]), 2),
            "freeze_l0_std": round(float(freeze_std[l]), 2),
            "z_score": round(float(z), 2) if not np.isnan(z) else None,
        })
        print(f"{l:>5} {baseline_mean[l]:>10.2f} +- {baseline_std[l]:<5.2f} "
              f"{freeze_mean[l]:>10.2f} +- {freeze_std[l]:<5.2f} "
              f"{z if not np.isnan(z) else 0:>9.2f}  {note}")

    n_candidates = sum(1 for r in results
                       if r["layer"] != FROZEN_LAYER and r["z_score"] is not None
                       and abs(r["z_score"]) > 3)
    print(f"\n{n_candidates} non-frozen layer(s) with |z| > 3 "
          f"(candidate real compensatory shifts).")
    if n_candidates == 0:
        print("Phase 1 gate: NOT cleared -- no layer outside the frozen one "
              "shows a shift beyond normal seed variance. Phase 2 (causal "
              "concept-ablation test) is not warranted on this evidence.")
    else:
        print("Phase 1 gate: CLEARED -- proceed to Phase 2 (concept-ablation "
              "recoverability test) per the note.")

    out_path = REPO_ROOT / "census" / "compensation_phase1_results.json"
    out_path.write_text(json.dumps({
        "baseline_run": BASELINE_RUN, "freeze_run": FREEZE_RUN,
        "frozen_layer": FROZEN_LAYER, "n_seeds_per_arm": len(baseline_rows),
        "layers": results,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
