"""
compensation_spectrum_sweep.py -- does the spectrum-shape match hold for
EVERY frozen layer, or only where the eff_dim test found real disruption?

Follow-up to compensation_spectrum_compare.py (froze L0 only) and the
full-layer sweep (train/compensation_full_layer_sweep.py, which trained all
32 freeze_l{0..31} arms and found real eff_dim disruption only in layers
0-3 -- 28 of 32 frozen layers show zero effect anywhere). Reuses that
existing corpus (comp_full_freeze_l00_c10 .. l31_c10, no new training) to
check: for a frozen layer with NO measured disruption (L4-31), does the
spectrum shape simply match baseline everywhere, including at the frozen
layer itself (since nothing detectably differs there either)? And for the
disruption zone (L0-3), does the same local-dip-then-recovery pattern
compensation_spectrum_compare.py found for L0 specifically also appear for
L1/L2/L3?

Per frozen arm: Pearson r at the frozen layer itself, r one and two layers
downstream, and the mean r over "far" layers (|layer - frozen| > 3,
matching the established disruption-radius boundary) as the summary.

Usage: python census/compensation_spectrum_sweep.py [--n-samples 4096]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compensation_spectrum_compare import forward_all_layers, load_weights, per_net_spectra

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))

sys.path.insert(0, str(REPO_ROOT / "census"))
from run_activation_census import get_task

CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"

BASELINE_RUN = "comp_p1_baseline_c10"
WIDTH = 256
DEPTH = 32


def main() -> None:
    rng = np.random.default_rng(7)
    prov = json.loads(next((CORPUS_DIR / BASELINE_RUN).glob("net_*.json")).read_text())
    task = get_task(prov["task"], WIDTH)
    x_task, _ = task.sample(args.n_samples, rng)
    x_task = x_task.astype(np.float64)

    print(f"loading {BASELINE_RUN}...", flush=True)
    bw = load_weights(BASELINE_RUN)
    b_acts = [forward_all_layers(w, x_task) for w in bw]
    b_spectra = per_net_spectra(b_acts)
    b_mean = b_spectra.mean(axis=0)  # (32, 256)

    all_results = {}
    summary_rows = []
    for frozen in range(DEPTH):
        run_id = f"comp_full_freeze_l{frozen:02d}_c10"
        print(f"loading {run_id}...", flush=True)
        try:
            fw = load_weights(run_id)
        except FileNotFoundError:
            print(f"  !! {run_id} not found, skipping")
            continue
        f_acts = [forward_all_layers(w, x_task) for w in fw]
        f_spectra = per_net_spectra(f_acts)
        f_mean = f_spectra.mean(axis=0)

        r_per_layer = [float(np.corrcoef(b_mean[l], f_mean[l])[0, 1]) for l in range(DEPTH)]
        far_layers = [l for l in range(DEPTH) if abs(l - frozen) > 3]
        far_mean_r = float(np.mean([r_per_layer[l] for l in far_layers]))
        r_at = lambda off: r_per_layer[frozen + off] if 0 <= frozen + off < DEPTH else None

        all_results[frozen] = {"r_per_layer": [round(x, 4) for x in r_per_layer],
                               "far_mean_r": round(far_mean_r, 4)}
        summary_rows.append((frozen, r_at(0), r_at(1), r_at(2), far_mean_r))

    out_path = CENSUS_DIR / "compensation_spectrum_sweep_results.json"
    out_path.write_text(json.dumps(
        {"baseline_run": BASELINE_RUN, "width": WIDTH, "depth": DEPTH,
         "per_frozen_layer": all_results,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}\n")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "--"

    print(f"{'frozen':>7} {'r@frozen':>9} {'r@+1':>8} {'r@+2':>8} {'far mean r':>11}")
    for frozen, r0, r1, r2, far in summary_rows:
        marker = "  <- disruption zone (F1)" if frozen <= 3 else ""
        print(f"{frozen:>7} {fmt(r0):>9} {fmt(r1):>8} {fmt(r2):>8} {far:>11.4f}{marker}")

    zone = [row for row in summary_rows if row[0] <= 3]
    modular = [row for row in summary_rows if row[0] > 3]
    print(f"\nDisruption zone (L0-3) mean r@frozen: {np.mean([r[1] for r in zone]):.4f}")
    print(f"Modular bulk (L4-31) mean r@frozen: {np.mean([r[1] for r in modular]):.4f}")
    print(f"Modular bulk (L4-31) mean far-r: {np.mean([r[4] for r in modular]):.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-samples", type=int, default=4096)
    args = p.parse_args()
    main()
