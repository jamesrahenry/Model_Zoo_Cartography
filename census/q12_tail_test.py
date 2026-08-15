"""
q12_tail_test.py — AMC Q12, tested directly (X2): do ARC's hard-tail
challenge nets carry distinctive census / state-descriptor signatures?

ARC's E3 showed estimation difficulty correlates with state-trajectory
atypicality. Q12 asks whether AMC-style census observables (participation
ratio, MP counts, spectra) — computed from weights alone — also mark the
tail. Uses the 200 public-split nets whose per-net final-layer MSE is
committed in arc-whitebox-canary/moment_chain/tail_robustness_results.json
(nets 50–249 of the atlas, same ordering).

Per net, weights-only features:
  - state trajectory (q, abar, asd, rbar) via the uncorrected chain; per-net
    atypicality = mean |z| vs the 200-net ensemble, per descriptor
  - weight-census scalars: eff-dim mean/min across layers, total sig dims
    (fixed He floor), max spectral norm ratio across layers

Reports Spearman correlation of each feature vs log per-net MSE, plus a
top-decile (hard tail) vs rest comparison.

Usage: python census/q12_tail_test.py [--n-nets 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "null_baseline"))
from chain_state_keyed import load_nets, state_basis, step_np
sys.path.insert(0, str(REPO_ROOT / "census"))
from manifold_detector import _mp_upper_edge, _participation_ratio

CANARY = Path.home() / "Source/Rosetta_Program/arc-whitebox-canary"
CENSUS_DIR = REPO_ROOT / "census"
SKIP = 50  # tail_robustness_results used nets 50..249


def net_features(W: np.ndarray) -> dict:
    w = W.shape[1]
    m, S = np.zeros(w), np.eye(w)
    traj = {"q": [], "abar": [], "asd": [], "rbar": []}
    effs, sigs, specs = [], [], []
    analytic = 2.0 / w
    for l in range(W.shape[0]):
        m, S, _mu, _sig, Spre, a, rho = step_np(m, S, W[l])
        g = state_basis(Spre, a, rho)
        for k, idx in (("q", 1), ("abar", 3), ("asd", 4), ("rbar", 5)):
            traj[k].append(float(g[idx]))
        eigs = np.linalg.eigvalsh(W[l].T @ W[l] / (w - 1))[::-1]
        effs.append(_participation_ratio(eigs))
        sigs.append(int(np.sum(eigs > _mp_upper_edge(w, w, analytic))))
        specs.append(float(np.sqrt(max(eigs[0], 0.0)) / np.sqrt(analytic * w)))
    return {"traj": traj, "eff_mean": float(np.mean(effs)),
            "eff_min": float(np.min(effs)), "sig_total": int(np.sum(sigs)),
            "spec_max": float(np.max(specs))}


def spearman(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return round(float(r), 3), float(p)


def main() -> None:
    mse = json.load(open(CANARY / "moment_chain/tail_robustness_results.json"))
    y = np.log(np.array(mse["state_keyed"][:args.n_nets]))

    print("loading atlas nets...", flush=True)
    nets = load_nets(SKIP + args.n_nets)[SKIP:SKIP + args.n_nets]
    feats = []
    for i, (W, _fm) in enumerate(nets):
        feats.append(net_features(np.asarray(W, dtype=np.float64)))
        if i % 40 == 0:
            print(f"  net {i} done", flush=True)

    # ensemble z-scores for trajectory atypicality
    atyp = {}
    for k in ("q", "abar", "asd", "rbar"):
        M = np.array([f["traj"][k] for f in feats])       # [n, L]
        z = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-12)
        atyp[k] = np.abs(z).mean(axis=1)                  # per-net mean |z|

    rows = {}
    for name, x in [("atyp_q", atyp["q"]), ("atyp_abar", atyp["abar"]),
                    ("atyp_asd", atyp["asd"]), ("atyp_rbar", atyp["rbar"]),
                    ("eff_mean", np.array([f["eff_mean"] for f in feats])),
                    ("eff_min", np.array([f["eff_min"] for f in feats])),
                    ("sig_total", np.array([f["sig_total"] for f in feats])),
                    ("spec_max", np.array([f["spec_max"] for f in feats]))]:
        r, p = spearman(x, y)
        hard = y >= np.quantile(y, 0.9)
        delta = float((x[hard].mean() - x[~hard].mean())
                      / (x.std() + 1e-12))
        rows[name] = {"spearman_vs_logmse": r, "p": round(p, 5),
                      "tail_vs_rest_effect_size": round(delta, 3)}
        print(f"{name:<12} rho={r:+.3f}  p={p:.4f}  tail-effect={delta:+.2f}σ")

    out_path = CENSUS_DIR / "q12_tail_test.json"
    out_path.write_text(json.dumps(
        {"n_nets": args.n_nets, "difficulty": "log state_keyed final MSE",
         "features": rows,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-nets", type=int, default=200)
    args = p.parse_args()
    main()
