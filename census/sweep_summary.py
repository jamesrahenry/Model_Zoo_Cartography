"""
sweep_summary.py — Corpus-level test of the three C-sweep predictions.

Pulls from the per-run weight censuses (census/<run>_weight_census.json),
null_baseline/state_trajectories.json, and census/activation_census.json,
groups runs by task size C (from corpus provenance), and reports per C:

  P1  L0 weight-census effective dim (prediction: tracks C)
  P2  chain q plateau x width (mean over L8..L24; prediction: ~ C)
  P3  activation significant dims on task input (mid-net; prediction: = C)

plus accuracy context (mean val acc, Bayes, outcomes) so partially-converged
families (C=50) can be read against how much was actually learned.

Usage: python census/sweep_summary.py --run-id <id> [--run-id ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"
TRAJ_PATH = REPO_ROOT / "null_baseline" / "state_trajectories.json"

PLATEAU = range(8, 25)  # L8..L24 inclusive


def main() -> None:
    traj = json.loads(TRAJ_PATH.read_text())
    act = json.loads((CENSUS_DIR / "activation_census.json").read_text())
    width = traj["width"]

    rows = []
    for run_id in args.run_ids:
        run_dir = CORPUS_DIR / run_id
        provs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("net_*.json"))]
        C = provs[0]["task"]["n_classes"]
        bayes = provs[0]["task"]["bayes_accuracy"]
        accs = [p["final_val_acc"] for p in provs]
        outcomes = sorted({p["outcome"] for p in provs})

        wc = json.loads((CENSUS_DIR / f"{run_id}_weight_census.json").read_text())
        l0_eff = np.mean([n["trained_census"][0]["effective_dim"]
                          for n in wc["nets"].values()])
        l0_sig = np.mean([n["trained_census"][0]["significant_dims"]
                          for n in wc["nets"].values()])

        tr = traj["runs"][run_id]
        q_plateau = np.mean([[n["trained"]["q"][l] for l in PLATEAU]
                             for n in tr.values()])

        ar = act["runs"][run_id]
        sig_task = np.mean([[n["trained"]["task"][l]["significant_dims"]
                             for l in PLATEAU] for n in ar.values()])
        eff_task = np.mean([[n["trained"]["task"][l]["eff_dim"]
                             for l in PLATEAU] for n in ar.values()])
        eff_noise = np.mean([[n["trained"]["noise"][l]["eff_dim"]
                              for l in PLATEAU] for n in ar.values()])

        rows.append({
            "run_id": run_id, "C": C, "bayes": bayes,
            "acc_mean": float(np.mean(accs)), "outcomes": ",".join(outcomes),
            "P1_L0_eff_dim": float(l0_eff), "L0_sig_dims": float(l0_sig),
            "P2_q_plateau_x_w": float(q_plateau * width),
            "P3_sig_dims_task": float(sig_task),
            "eff_dim_task": float(eff_task), "eff_dim_noise": float(eff_noise),
        })

    rows.sort(key=lambda r: (r["C"], r["run_id"]))
    hdr = (f"{'run':<22} {'C':>4} {'acc':>7} {'bayes':>7} {'outcome':>10} "
           f"{'P1:L0eff':>9} {'L0sig':>6} {'P2:q*w':>7} {'P3:sig':>7} "
           f"{'eff(task)':>10} {'eff(noise)':>10}")
    print(hdr)
    for r in rows:
        print(f"{r['run_id']:<22} {r['C']:>4} {r['acc_mean']:>7.4f} "
              f"{r['bayes']:>7.4f} {r['outcomes']:>10} "
              f"{r['P1_L0_eff_dim']:>9.1f} {r['L0_sig_dims']:>6.1f} "
              f"{r['P2_q_plateau_x_w']:>7.1f} {r['P3_sig_dims_task']:>7.1f} "
              f"{r['eff_dim_task']:>10.1f} {r['eff_dim_noise']:>10.1f}")

    out_path = CENSUS_DIR / "c_sweep_summary.json"
    out_path.write_text(json.dumps({"plateau_layers": "L8-L24", "rows": rows},
                                   indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", dest="run_ids", action="append", required=True)
    args = p.parse_args()
    main()
