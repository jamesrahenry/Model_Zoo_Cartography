"""
refit_trained.py — Re-fit the state-keyed stabilizer on OUR trained population.

WRITEUP §8 said the random-ensemble corrections must be re-fit, not reused, for
any other population; the activation census quantified the failure (uncorrected
mean-field eigenspectrum rel err 80%+ by L16 on trained weights). This script
runs the ARC sequential-prefix DAgger fit (chain_state_keyed.py, canary
b6e75e1) against MC ground truth generated from our own corpus nets.

Population design:
  fit set      — converged families at C=2, 5, 25 (9 nets)
  held-out val — both C=10 families (6 nets; unseen task size)
  transfer     — C=50 partial learners (3 nets; reported, never fitted)

Truth per net: forward n_mc N(0,I) samples, record per-layer post-ReLU mean
and second moment M11 (the same quantities the ARC fit targets). Probe
cumulants use the vendored probe_cumulants (forward passes on the net itself —
this is a fitted instrument, not the zero-forward-pass null).

Metrics per layer, held-out, uncorrected (theta=0) vs refit theta:
  - mean-vector MSE (ARC's fit target)
  - centered-covariance eigenspectrum median rel err over top 20 (our census
    failure metric — directly comparable to vacuum_vs_noise numbers)
  - q = PR/w rel err (the rank-collapse clock)

Usage:
  python null_baseline/refit_trained.py [--n-mc 16384] [--n-probe 4096]
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chain_state_keyed import (DEPTH, N_BASIS, NMF, NOF, NVF, W_DIM, corrections,
                               mean_features, probe_cumulants, state_basis,
                               step_np, tcross_feats, var_features)


# ---- parametric-basis machinery (B = 8, or 10 with --edge-flags: the M1
# edge-aware variant appends [is_first, is_last] indicator dims — the layers
# where pure state-keying actively hurt) ----

def basis_dim() -> int:
    if args.basis == "const":
        return 1
    return N_BASIS + 2 if args.basis == "edge" else N_BASIS


def ext_basis(Spre, a, rho, l: int, depth: int) -> np.ndarray:
    if args.basis == "const":
        # factor-before-you-fit rung: per-feature CONSTANT coefficients
        # (16 params total) — no state dependence at all
        return np.array([1.0])
    g = state_basis(Spre, a, rho)
    if args.basis != "edge":
        return g
    return np.concatenate([g, [1.0 if l == 0 else 0.0,
                               1.0 if l == depth - 1 else 0.0]])


def zero_theta():
    B = basis_dim()
    return (np.zeros((NMF, B)), np.zeros((NVF, B)), np.zeros((NOF, B)))


def fresh_acc():
    B = basis_dim()
    return {"m_xtx": np.zeros((NMF * B,) * 2), "m_xty": np.zeros(NMF * B),
            "v_xtx": np.zeros((NVF * B,) * 2), "v_xty": np.zeros(NVF * B),
            "o_xtx": np.zeros((NOF * B,) * 2), "o_xty": np.zeros(NOF * B)}


def solve_theta(acc, ridge=1e-9):
    B = basis_dim()

    def solve(xtx, xty, nf):
        A = xtx + ridge * np.trace(xtx) / max(len(xtx), 1) * np.eye(len(xtx))
        return np.linalg.solve(A, xty).reshape(nf, B)

    return (solve(acc["m_xtx"], acc["m_xty"], NMF),
            solve(acc["v_xtx"], acc["v_xty"], NVF),
            solve(acc["o_xtx"], acc["o_xty"], NOF))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from corpus_io import net_paths
CORPUS_DIR = REPO_ROOT / "corpus"

# --population converged (default): the original design.
# --population mixed (M2): partial learners join the FIT set — tests whether
# one correction family can serve both regimes.
POPULATIONS = {
    "converged": {
        "fit": ["sweep_c2_head", "sweep_c5_head", "sweep_c25_head"],
        "val": ["probe_d32_c10", "probe_d32_c10_head"],
        "transfer": ["sweep_c50_head"],
    },
    "mixed": {
        "fit": ["sweep_c2_head", "sweep_c5_head", "sweep_c25_head",
                "sweep_c40_head", "sweep_c50_head"],
        "val": ["probe_d32_c10", "probe_d32_c10_head"],
        "transfer": ["b3_c44", "sweep_c50_head_60k"],
    },
}


def load_trained(run_ids: list[str]) -> list[tuple[str, np.ndarray]]:
    """Population is EXPLICIT: the first --max-per-run nets (seed order) of
    each run. The original committed results used what was then a 3-seed
    local corpus (audit: reproducibility gap); regenerations must state
    max_per_run in the output."""
    nets = []
    for run_id in run_ids:
        for npz_path in net_paths(run_id)[:args.max_per_run]:
            d = np.load(npz_path)
            n_layers = sum(1 for k in d.files if k.startswith("init_w"))
            W = np.stack([d[f"w{i}"].astype(np.float64) for i in range(n_layers)])
            nets.append((f"{run_id}/{npz_path.stem}", W))
    return nets


def mc_truth(W: np.ndarray, n_mc: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n_mc, W.shape[1]))
    mean, M11 = [], []
    for l in range(W.shape[0]):
        h = np.maximum(h @ W[l], 0.0)
        mean.append(h.mean(axis=0))
        M11.append(h.T @ h / n_mc)
    return {"mean": np.stack(mean), "M11": np.stack(M11)}


def roll_eval(W: np.ndarray, probe, theta) -> tuple[list, list, list]:
    """Corrected chain; returns per-layer (mean, top-20 eigs of S, q)."""
    w = W.shape[1]
    m, S = np.zeros(w), np.eye(w)
    means, eigs, qs = [], [], []
    for l in range(W.shape[0]):
        m_hat, S_hat, mu_pre, sig_pre, Spre, a, rho = step_np(m, S, W[l])
        k3, k4, kiij = probe[l]
        Fm = mean_features(a, sig_pre, k3, k4)
        Fv = var_features(a, sig_pre, k3, k4)
        Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho)
        g = ext_basis(Spre, a, rho, l, W.shape[0])
        dm, dv, dS = corrections(theta, g, Fm, Fv, Fo)
        m = m_hat + dm
        S = S_hat + dS
        S[np.diag_indices(w)] = np.clip(np.diag(S_hat) + dv, 1e-30, None)
        means.append(np.maximum(m, 0.0))
        ev = np.clip(np.linalg.eigvalsh(0.5 * (S + S.T))[::-1], 0.0, None)
        eigs.append(ev[:20])
        qs.append(float(np.trace(S) ** 2 / (w * np.sum(S * S) + 1e-300)))
    return means, eigs, qs


def metrics(nets, probes, truths, theta) -> dict:
    """Per-layer metrics averaged over nets."""
    mmse = np.zeros(DEPTH)
    relerr = np.zeros(DEPTH)
    qerr = np.zeros(DEPTH)
    for (name, W), probe, tr in zip(nets, probes, truths):
        means, eigs, qs = roll_eval(W, probe, theta)
        for l in range(DEPTH):
            tm = tr["mean"][l]
            tS = tr["M11"][l] - np.outer(tm, tm)
            tev = np.clip(np.linalg.eigvalsh(tS)[::-1][:20], 1e-30, None)
            tq = float(np.trace(tS) ** 2 / (W_DIM * np.sum(tS * tS) + 1e-300))
            mmse[l] += np.mean((means[l] - tm) ** 2)
            relerr[l] += np.median(np.abs(eigs[l] - tev) / tev)
            qerr[l] += abs(qs[l] - tq) / tq
    n = len(nets)
    return {"mean_mse": (mmse / n).tolist(),
            "eig_relerr_top20": (relerr / n).tolist(),
            "q_relerr": (qerr / n).tolist()}


def main() -> None:
    t0 = time.time()
    pop = POPULATIONS[args.population]
    fit_nets = load_trained(pop["fit"])
    val_nets = load_trained(pop["val"])
    tra_nets = load_trained(pop["transfer"])
    print(f"fit {len(fit_nets)}, val {len(val_nets)}, transfer {len(tra_nets)}")

    def prep(nets, tag):
        probes, truths = [], []
        for i, (name, W) in enumerate(nets):
            probes.append(probe_cumulants(W, args.n_probe, seed=7000 + i))
            truths.append(mc_truth(W, args.n_mc, seed=8000 + i))
            print(f"  {tag} {name} probed ({time.time()-t0:.0f}s)", flush=True)
        return probes, truths

    fit_p, fit_t = prep(fit_nets, "fit")
    val_p, val_t = prep(val_nets, "val")
    tra_p, tra_t = prep(tra_nets, "tra")

    # ---- sequential-prefix DAgger fit (structure from chain_state_keyed) ----
    theta = zero_theta()
    acc = fresh_acc()
    states = [(np.zeros(W_DIM), np.eye(W_DIM)) for _ in fit_nets]
    iu = np.triu_indices(W_DIM, k=1)
    for l in range(DEPTH):
        cache = []
        for t, (name, W) in enumerate(fit_nets):
            m_hat, S_hat, mu_pre, sig_pre, Spre, a, rho = step_np(*states[t], W[l])
            k3, k4, kiij = fit_p[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho)
            g = ext_basis(Spre, a, rho, l, DEPTH)
            tm = fit_t[t]["mean"][l]
            tS = fit_t[t]["M11"][l] - np.outer(tm, tm)
            Xm = (Fm[:, :, None] * g[None, None, :]).reshape(W_DIM, NMF * basis_dim())
            acc["m_xtx"] += Xm.T @ Xm; acc["m_xty"] += Xm.T @ (tm - m_hat)
            Xv = (Fv[:, :, None] * g[None, None, :]).reshape(W_DIM, NVF * basis_dim())
            acc["v_xtx"] += Xv.T @ Xv; acc["v_xty"] += Xv.T @ (np.diag(tS) - np.diag(S_hat))
            Fo_iu = np.stack([f[iu] for f in Fo], axis=1)
            Xo = (Fo_iu[:, :, None] * g[None, None, :]).reshape(len(iu[0]), NOF * basis_dim())
            acc["o_xtx"] += Xo.T @ Xo; acc["o_xty"] += Xo.T @ (tS - S_hat)[iu]
            cache.append((m_hat, S_hat, Fm, Fv, Fo, g))
        theta = solve_theta(acc)
        for t in range(len(fit_nets)):
            m_hat, S_hat, Fm, Fv, Fo, g = cache[t]
            dm, dv, dS = corrections(theta, g, Fm, Fv, Fo)
            K2 = S_hat + dS
            K2[np.diag_indices(W_DIM)] = np.clip(np.diag(S_hat) + dv, 1e-30, None)
            states[t] = (m_hat + dm, K2)
        if l in (0, 7, 15, 23, 31):
            print(f"seq layer {l+1} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- damped self-consistency polish, accept on held-out final-layer MSE ----
    def val_final_mse(th):
        return float(np.mean(metrics(val_nets, val_p, val_t, th)["mean_mse"][-1]))

    best_theta, best_val = theta, val_final_mse(theta)
    print(f"sequential fit: val final mean-mse {best_val:.3e} ({time.time()-t0:.0f}s)")
    eta = 0.5
    for it in range(args.n_iter):
        acc = fresh_acc()
        for t, (name, W) in enumerate(fit_nets):
            # full-trajectory collect under current theta
            w = W.shape[1]
            m, S = np.zeros(w), np.eye(w)
            for l in range(DEPTH):
                m_hat, S_hat, mu_pre, sig_pre, Spre, a, rho = step_np(m, S, W[l])
                k3, k4, kiij = fit_p[t][l]
                Fm = mean_features(a, sig_pre, k3, k4)
                Fv = var_features(a, sig_pre, k3, k4)
                Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a, rho)
                g = ext_basis(Spre, a, rho, l, DEPTH)
                tm = fit_t[t]["mean"][l]
                tS = fit_t[t]["M11"][l] - np.outer(tm, tm)
                Xm = (Fm[:, :, None] * g[None, None, :]).reshape(W_DIM, NMF * basis_dim())
                acc["m_xtx"] += Xm.T @ Xm; acc["m_xty"] += Xm.T @ (tm - m_hat)
                Xv = (Fv[:, :, None] * g[None, None, :]).reshape(W_DIM, NVF * basis_dim())
                acc["v_xtx"] += Xv.T @ Xv; acc["v_xty"] += Xv.T @ (np.diag(tS) - np.diag(S_hat))
                Fo_iu = np.stack([f[iu] for f in Fo], axis=1)
                Xo = (Fo_iu[:, :, None] * g[None, None, :]).reshape(len(iu[0]), NOF * basis_dim())
                acc["o_xtx"] += Xo.T @ Xo; acc["o_xty"] += Xo.T @ (tS - S_hat)[iu]
                dm, dv, dS = corrections(best_theta, g, Fm, Fv, Fo)
                m = m_hat + dm
                S = S_hat + dS
                S[np.diag_indices(w)] = np.clip(np.diag(S_hat) + dv, 1e-30, None)
        fitted = solve_theta(acc)
        cand = tuple((1 - eta) * b + eta * f for b, f in zip(best_theta, fitted))
        cand_val = val_final_mse(cand)
        ok = cand_val < best_val
        print(f"polish {it+1}: val {cand_val:.3e} ({'accepted' if ok else 'rejected'}, "
              f"eta={eta}) ({time.time()-t0:.0f}s)", flush=True)
        if ok:
            best_theta, best_val = cand, cand_val
        else:
            eta *= 0.5
            if eta < 0.1:
                break
    theta = best_theta

    # ---- final evaluation ----
    results = {}
    for tag, nets, probes, truths in [("val_c10", val_nets, val_p, val_t),
                                      ("transfer", tra_nets, tra_p, tra_t)]:
        results[tag] = {"uncorrected": metrics(nets, probes, truths, zero_theta()),
                        "refit": metrics(nets, probes, truths, theta)}
    # transfer eval printed too when mixed
    print("\n=== transfer set: eigenspectrum median rel err (top 20) ===")
    sel2 = [0, 1, 3, 7, 15, 23, 31]
    for k in ("uncorrected", "refit"):
        v = results["transfer"][k]["eig_relerr_top20"]
        print(f"{k:<12} " + " ".join(f"{v[l]:<8.4f}" for l in sel2))

    sel = [0, 1, 3, 7, 15, 23, 31]
    print("\n=== held-out C=10 (6 nets): eigenspectrum median rel err (top 20) ===")
    print("layer        " + " ".join(f"L{l:<7}" for l in sel))
    for k in ("uncorrected", "refit"):
        v = results["val_c10"][k]["eig_relerr_top20"]
        print(f"{k:<12} " + " ".join(f"{v[l]:<8.4f}" for l in sel))
    print("=== q rel err ===")
    for k in ("uncorrected", "refit"):
        v = results["val_c10"][k]["q_relerr"]
        print(f"{k:<12} " + " ".join(f"{v[l]:<8.4f}" for l in sel))
    print("=== mean-vector MSE ===")
    for k in ("uncorrected", "refit"):
        v = results["val_c10"][k]["mean_mse"]
        print(f"{k:<12} " + " ".join(f"{v[l]:<8.2e}" for l in sel))

    n_params = (NMF + NVF + NOF) * basis_dim()
    out = {
        "population": {**pop, "name": args.population,
                       "max_per_run": args.max_per_run},
        "config": {"n_mc": args.n_mc, "n_probe": args.n_probe,
                   "n_iter": args.n_iter, "n_params": n_params,
                   "basis": args.basis},
        "results": results,
        "theta_b64": base64.b64encode(struct.pack(
            "<%dd" % n_params,
            *np.concatenate([t.ravel() for t in theta]))).decode(),
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path = Path(__file__).with_name(args.out_name)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-mc", type=int, default=16384)
    p.add_argument("--n-probe", type=int, default=4096)
    p.add_argument("--n-iter", type=int, default=4)
    p.add_argument("--max-per-run", type=int, default=8)
    p.add_argument("--basis", default="state",
                   choices=["const", "state", "edge"],
                   help="const=16 params (factor rung), state=128, edge=160")
    p.add_argument("--out-name", default="refit_trained_results.json")
    args = p.parse_args()
    main()
