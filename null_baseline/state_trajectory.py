"""
state_trajectory.py — The four-number-per-layer statistical skeleton
(q, abar, asd, rbar), rolled analytically through weights. Zero forward passes.

Implements the ARC handoff's suggested first corpus measurement
(notes/2026-08-10_arc-state-descriptor-handoff.md §1-2): propagate
(mean, cov) = (0, I) through the uncorrected Hermite moment chain
(chain_state_keyed.step_np) and record, per layer, the measured state:

  q    — PR(Σ_pre)/w, the rank-collapse clock (1 = isotropic, →0 collapsed)
  abar — mean ReLU operating point μ/σ
  asd  — std of the operating point
  rbar — mean |off-diagonal correlation|

The null is a BAND, not a curve (handoff §3): we roll a fresh random-init
ensemble at matched architecture and report its per-layer distribution, then
place each corpus net's init and trained trajectories against it. The
headline question (handoff §2): does training arrest or accelerate the
rank collapse — is q(trained) above or below the random band at depth?

Usage:
  python null_baseline/state_trajectory.py --run-id probe_d32_c10_head \
      [--run-id probe_d32_c10] [--n-null 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chain_state_keyed import step_np, state_basis
from stage1_validate import build_mlp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from corpus_io import net_paths
CORPUS_DIR = REPO_ROOT / "corpus"
OUT_DIR = REPO_ROOT / "null_baseline"


def trajectory(weights: list[np.ndarray]) -> dict[str, list[float]]:
    """Roll the uncorrected analytic chain; return per-layer state descriptors."""
    w = weights[0].shape[0]
    m, S = np.zeros(w), np.eye(w)
    out = {"q": [], "abar": [], "asd": [], "rbar": []}
    for W in weights:
        m, S, _mu, _sig, Spre, a, rho = step_np(m, S, np.asarray(W, dtype=np.float64))
        g = state_basis(Spre, a, rho)
        out["q"].append(round(float(g[1]), 6))
        out["abar"].append(round(float(g[3]), 6))
        out["asd"].append(round(float(g[4]), 6))
        out["rbar"].append(round(float(g[5]), 6))
    return out


def main() -> None:
    # Establish architecture from the first corpus net.
    first_run = CORPUS_DIR / args.run_ids[0]
    first_npz = net_paths(args.run_ids[0])[0]
    d = np.load(first_npz)
    n_layers = sum(1 for k in d.files if k.startswith("init_w"))
    width = d["init_w0"].shape[0]

    print(f"null band: {args.n_null} fresh He nets at width {width} depth {n_layers}")
    null_trajs = [trajectory(build_mlp(width, n_layers, seed=10_000 + i))
                  for i in range(args.n_null)]
    null_band = {
        k: {"mean": np.mean([t[k] for t in null_trajs], axis=0).round(6).tolist(),
            "std": np.std([t[k] for t in null_trajs], axis=0).round(6).tolist()}
        for k in ("q", "abar", "asd", "rbar")
    }

    runs = {}
    for run_id in args.run_ids:
        run_dir = CORPUS_DIR / run_id
        nets = {}
        for npz_path in net_paths(run_id):
            d = np.load(npz_path)
            init = [d[f"init_w{i}"] for i in range(n_layers)]
            final = [d[f"w{i}"] for i in range(n_layers)]
            nets[npz_path.stem] = {"init": trajectory(init),
                                   "trained": trajectory(final)}
            print(f"  {run_id}/{npz_path.stem} done")
        runs[run_id] = nets

    out = {
        "descriptors": "q=PR(Sigma_pre)/w, abar/asd=mean/std of mu/sigma, "
                       "rbar=mean |off-diag rho|; uncorrected Hermite chain from (0, I)",
        "width": width, "depth": n_layers, "n_null": args.n_null,
        "null_band": null_band,
        "runs": runs,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    out_path = OUT_DIR / "state_trajectories.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}\n")

    # Summary: q (rank-collapse clock) at selected depths, trained vs band.
    sel = [0, 1, 2, 4, 8, 16, 24, 31]
    sel = [l for l in sel if l < n_layers]
    hdr = " ".join(f"L{l:<7}" for l in sel)
    print(f"{'q trajectory':<28} {hdr}")
    nm = null_band["q"]["mean"]; ns = null_band["q"]["std"]
    print(f"{'null mean':<28} " + " ".join(f"{nm[l]:<8.4f}" for l in sel))
    print(f"{'null std':<28} " + " ".join(f"{ns[l]:<8.4f}" for l in sel))
    for run_id, nets in runs.items():
        tq = np.mean([[n['trained']['q'][l] for l in sel] for n in nets.values()], axis=0)
        iq = np.mean([[n['init']['q'][l] for l in sel] for n in nets.values()], axis=0)
        print(f"{run_id + ' init':<28} " + " ".join(f"{v:<8.4f}" for v in iq))
        print(f"{run_id + ' trained':<28} " + " ".join(f"{v:<8.4f}" for v in tq))
        # sigma distance from the null band at final layer
        lf = sel[-1]
        z = (tq[-1] - nm[lf]) / (ns[lf] + 1e-12)
        print(f"{'':<28} final-layer q: {z:+.1f} sigma from null band")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", dest="run_ids", action="append", required=True)
    p.add_argument("--n-null", type=int, default=20)
    args = p.parse_args()
    main()
