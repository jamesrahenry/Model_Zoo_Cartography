"""
run_activation_census.py — Post-inference census: real activations vs the
weight-only predictions (analytic vacuum + moment chain), on two input
distributions.

For every net in a corpus run (init and trained weights), forward-passes
n_samples inputs in numpy (x @ W, ReLU, float64) and records per layer:

  - post-ReLU spectral census (centered eff dim, self-calibrated MP sig dims)
  - empirical pre-activation q = tr(Σ)²/(w·‖Σ‖²_F) — directly comparable to
    the chain-predicted q in null_baseline/state_trajectories.json
  - dead-unit fraction (units that never fire)
  - analytic-vacuum comparison (noise input only): predicted vs empirical
    centered eigenspectrum (median relative error over top 20) and eff dim

Input distributions:
  - "noise": x ~ N(0, I) — the null's premise, and the vacuum/chain input model
  - "task":  the net's own GMM task samples (reconstructed from provenance).
    By construction the task aggregate matches N(0, I) in mean AND covariance,
    so any activation difference between the two is attributable to
    higher-order input structure (the class clusters) — the weight × input
    interaction isolated from second-moment effects.

Usage:
  python census/run_activation_census.py --run-id probe_d32_c10_head \
      [--run-id ...] [--n-samples 4096]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "census"))
sys.path.insert(0, str(REPO_ROOT / "null_baseline"))
sys.path.insert(0, str(REPO_ROOT / "train"))
from manifold_detector import _layer_census
from analytic_vacuum import predict_activation_cov_spectrum
from gmm_task import make_gmm_task

CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"


def empirical_q(z: np.ndarray) -> float:
    S = np.cov(z, rowvar=False)
    return float(np.trace(S) ** 2 / (z.shape[1] * np.sum(S * S) + 1e-300))


def forward_census(weights: list[np.ndarray], x: np.ndarray) -> list[dict]:
    per_layer = []
    h = np.asarray(x, dtype=np.float64)
    for l, W in enumerate(weights):
        z = h @ np.asarray(W, dtype=np.float64)
        h = np.maximum(z, 0.0)
        cen = _layer_census(h, l, {}, n_top_eigenvalues=20)
        per_layer.append({
            "layer": l,
            "q_pre": round(empirical_q(z), 6),
            "eff_dim": round(cen.effective_dim, 2),
            "significant_dims": cen.significant_dims,
            "dead_frac": round(float((h.max(axis=0) == 0.0).mean()), 4),
            "top_eigenvalues": [round(e, 8) for e in cen.top_eigenvalues],
        })
    return per_layer


def vacuum_compare(weights: list[np.ndarray], noise_census: list[dict]) -> list[dict]:
    pred = predict_activation_cov_spectrum(
        [np.asarray(W, dtype=np.float64) for W in weights], n_top=20)
    out = []
    for l, layer in enumerate(noise_census):
        emp = np.array(layer["top_eigenvalues"])
        prd = np.array(pred["eig_centered"][l][:len(emp)])
        rel = np.abs(prd - emp) / np.maximum(np.abs(emp), 1e-30)
        out.append({
            "layer": l,
            "eig_relerr_median_top20": round(float(np.median(rel)), 4),
            "eff_dim_pred": round(float(pred["eff_dim_centered"][l]), 2),
            "eff_dim_emp": layer["eff_dim"],
        })
    return out


def main() -> None:
    rng = np.random.default_rng(args.sample_seed)
    results = {}
    for run_id in args.run_ids:
        run_dir = CORPUS_DIR / run_id
        nets = {}
        for npz_path in sorted(run_dir.glob("net_*.npz")):
            name = npz_path.stem
            d = np.load(npz_path)
            n_layers = sum(1 for k in d.files if k.startswith("init_w"))
            width = d["init_w0"].shape[0]
            init = [d[f"init_w{i}"] for i in range(n_layers)]
            trained = [d[f"w{i}"] for i in range(n_layers)]

            prov = json.loads((run_dir / f"{name}.json").read_text())
            t = prov["task"]
            task = make_gmm_task(dim=width, n_classes=t["n_classes"],
                                 separation=t["separation"], seed=t["task_seed"])

            x_noise = rng.standard_normal((args.n_samples, width))
            x_task, _ = task.sample(args.n_samples, rng)
            x_task = x_task.astype(np.float64)

            entry = {}
            for fam, ws in (("init", init), ("trained", trained)):
                noise_c = forward_census(ws, x_noise)
                entry[fam] = {
                    "noise": noise_c,
                    "task": forward_census(ws, x_task),
                    "vacuum_vs_noise": vacuum_compare(ws, noise_c),
                }
            nets[name] = entry
            print(f"{run_id}/{name} done", flush=True)
        results[run_id] = nets

    out = {
        "n_samples": args.n_samples, "sample_seed": args.sample_seed,
        "notes": "task aggregate matches N(0,I) in mean+cov by construction; "
                 "task-vs-noise differences are higher-order input structure",
        "runs": results,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    out_path = CENSUS_DIR / "activation_census.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}\n")

    # Summary at selected layers
    sel = [0, 1, 2, 4, 8, 16, 24, 31]
    for run_id, nets in results.items():
        n_layers = len(next(iter(nets.values()))["init"]["noise"])
        lsel = [l for l in sel if l < n_layers]
        print(f"== {run_id} ==")
        hdr = " ".join(f"L{l:<7}" for l in lsel)
        print(f"{'metric':<34} {hdr}")

        def avg(fam, dist, key):
            return [float(np.mean([nets[n][fam][dist][l][key] for n in nets]))
                    for l in lsel]

        def avgv(fam, key):
            return [float(np.mean([nets[n][fam]["vacuum_vs_noise"][l][key] for n in nets]))
                    for l in lsel]

        for label, vals in [
            ("q_pre init/noise",        avg("init", "noise", "q_pre")),
            ("q_pre trained/noise",     avg("trained", "noise", "q_pre")),
            ("q_pre trained/task",      avg("trained", "task", "q_pre")),
            ("eff_dim init/noise",      avg("init", "noise", "eff_dim")),
            ("eff_dim trained/noise",   avg("trained", "noise", "eff_dim")),
            ("eff_dim trained/task",    avg("trained", "task", "eff_dim")),
            ("eff_dim vacuum(trained)", avgv("trained", "eff_dim_pred")),
            ("sig_dims trained/noise",  avg("trained", "noise", "significant_dims")),
            ("sig_dims trained/task",   avg("trained", "task", "significant_dims")),
            ("dead_frac trained/task",  avg("trained", "task", "dead_frac")),
            ("vac_relerr init/noise",   avgv("init", "eig_relerr_median_top20")),
            ("vac_relerr trained/noise", avgv("trained", "eig_relerr_median_top20")),
        ]:
            print(f"{label:<34} " + " ".join(f"{v:<8.4f}" for v in vals))
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", dest="run_ids", action="append", required=True)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--sample-seed", type=int, default=42)
    args = p.parse_args()
    main()
