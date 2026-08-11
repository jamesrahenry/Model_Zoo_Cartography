"""
run_census.py — Weight-space census for a corpus run: trained vs own-init
vs fresh-random, under the analytic He-Gaussian null.

For every net in corpus/<run_id>/ this loads init and final weights from the
.npz (ARC (in, out) convention, saved by train/train_mlp.py) and runs the AMC
spectral census on each weight matrix with the analytic null variance
σ² = 2/fan_in and no mean-centering (the mean-row direction may be learned
structure; see PROVENANCE.md). Reports, per layer:

  - significant dims above the analytic MP floor (init vs trained)
  - effective dim (participation ratio) (init vs trained)
  - Frobenius drift ‖W_trained − W_init‖_F / ‖W_init‖_F
  - spectral norm ratio σ_max(trained) / σ_max(init)

Writes census/<run_id>_weight_census.json and prints a per-layer summary
averaged over nets. Pre-inference only — activation census is a separate step.

Usage:
  python census/run_census.py --run-id probe_d32_c10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from manifold_detector import layer_manifold_census

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from corpus_io import net_paths
CORPUS_DIR = REPO_ROOT / "corpus"
CENSUS_DIR = REPO_ROOT / "census"


def load_net(npz_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (init_weights, final_weights), each a list ordered by layer."""
    data = np.load(npz_path)
    n_layers = sum(1 for k in data.files if k.startswith("init_w"))
    init = [data[f"init_w{i}"].astype(np.float64) for i in range(n_layers)]
    final = [data[f"w{i}"].astype(np.float64) for i in range(n_layers)]
    return init, final


def census_weights(weights: list[np.ndarray]) -> list[dict]:
    fan_in = weights[0].shape[0]
    res = layer_manifold_census(weights, noise_variance=2.0 / fan_in, center=False)
    return [
        {"layer": L.layer,
         "significant_dims": L.significant_dims,
         "effective_dim": round(L.effective_dim, 2),
         "total_variance": round(L.total_variance, 6),
         "top_eigenvalues": [round(e, 6) for e in L.top_eigenvalues[:10]]}
        for L in res.layers
    ]


def main() -> None:
    run_dir = CORPUS_DIR / args.run_id
    npz_files = net_paths(args.run_id)
    if not npz_files:
        raise SystemExit(f"no nets found in {run_dir}")

    nets = {}
    for npz_path in npz_files:
        name = npz_path.stem
        prov = json.loads((run_dir / f"{name}.json").read_text())
        init_w, final_w = load_net(npz_path)
        drift = [float(np.linalg.norm(f - i) / np.linalg.norm(i))
                 for i, f in zip(init_w, final_w)]
        spec_ratio = [float(np.linalg.svd(f, compute_uv=False)[0]
                            / np.linalg.svd(i, compute_uv=False)[0])
                      for i, f in zip(init_w, final_w)]
        nets[name] = {
            "outcome": prov["outcome"],
            "final_val_acc": prov["final_val_acc"],
            "init_census": census_weights(init_w),
            "trained_census": census_weights(final_w),
            "frobenius_drift": [round(d, 4) for d in drift],
            "spectral_norm_ratio": [round(r, 4) for r in spec_ratio],
        }
        print(f"{name}: {prov['outcome']} acc={prov['final_val_acc']:.4f}")

    out = {
        "run_id": args.run_id,
        "null": "MP edge at analytic sigma^2 = 2/fan_in, uncentered second moments",
        "n_nets": len(nets),
        "nets": nets,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    out_path = CENSUS_DIR / f"{args.run_id}_weight_census.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Per-layer summary averaged over nets: init vs trained
    n_layers = len(next(iter(nets.values()))["init_census"])
    print(f"\n{'layer':>5} {'sig(init)':>10} {'sig(trained)':>13} "
          f"{'eff(init)':>10} {'eff(trained)':>13} {'drift':>7} {'specx':>7}")
    for l in range(n_layers):
        si = np.mean([n["init_census"][l]["significant_dims"] for n in nets.values()])
        st = np.mean([n["trained_census"][l]["significant_dims"] for n in nets.values()])
        ei = np.mean([n["init_census"][l]["effective_dim"] for n in nets.values()])
        et = np.mean([n["trained_census"][l]["effective_dim"] for n in nets.values()])
        dr = np.mean([n["frobenius_drift"][l] for n in nets.values()])
        sr = np.mean([n["spectral_norm_ratio"][l] for n in nets.values()])
        print(f"{l:>5} {si:>10.1f} {st:>13.1f} {ei:>10.1f} {et:>13.1f} "
              f"{dr:>7.3f} {sr:>7.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    main()
