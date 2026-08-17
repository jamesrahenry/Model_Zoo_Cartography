"""
crossover_map.py — Where does sampling-free (analytic) propagation beat
FLOP-matched sampling, as a function of (width, depth)?

The measurement keenanpepper's Phase-1 write-up asked for ("Wu et al. beat
sampling at sufficient width to depth 12; I lose at depth 32. Width effect,
depth effect, or depth/width?"). MZC is positioned to answer: random He nets
at any (w, d), the vendored k=2 Hermite chain, batched GPU ground truth.

Per grid cell (w, d), K nets:
  truth      final-layer post-ReLU mean per neuron, MC at N_truth (GPU)
  analytic   uncorrected k=2 chain (step_np; zero fitted parameters)
  sampling   MC at a FLOP-matched budget: chain ≈ 4·d·w³ FLOPs, a sample
             ≈ 2·d·w² FLOPs → n_matched = 2·w samples (and 10× that as a
             robustness band)
  metric     MSE across neurons vs truth, averaged over nets

Output: per-cell MSEs + the crossover ratio analytic/sampling; the d*(w)
contour answers the question.

Usage: python null_baseline/crossover_map.py [--k-nets 12] [--n-truth 2000000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chain_state_keyed import step_np
from stage1_validate import build_mlp

OUT = Path(__file__).with_name("crossover_map.json")  # overridden by --out

WIDTHS = [64, 128, 256, 512, 1024]
DEPTHS = [4, 8, 12, 16, 24, 32, 48]


def analytic_final_mean(Ws: list[np.ndarray]) -> np.ndarray:
    w = Ws[0].shape[0]
    m, S = np.zeros(w), np.eye(w)
    for W in Ws:
        m, S, *_ = step_np(m, S, np.asarray(W, dtype=np.float64))
    return np.maximum(m, 0.0)


def mc_final_mean(Ws, n: int, device, gen) -> np.ndarray:
    import torch
    w = Ws[0].shape[0]
    total = torch.zeros(w, dtype=torch.float64, device=device)
    chunk = min(n, max(1, int(2e8 // (w * 4))))  # VRAM-bounded chunks
    done = 0
    Wt = [torch.from_numpy(np.asarray(W, dtype=np.float32)).to(device) for W in Ws]
    while done < n:
        b = min(chunk, n - done)
        h = torch.randn(b, w, device=device, generator=gen)
        for W in Wt:
            h = torch.relu(h @ W)
        total += h.sum(dim=0, dtype=torch.float64)
        done += b
    return (total / n).cpu().numpy()


def main() -> None:
    global DEPTHS, OUT
    if args.depths:
        DEPTHS = args.depths
    if args.out:
        OUT = Path(__file__).with_name(args.out)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()
    cells = {}
    for w in WIDTHS:
        for d in DEPTHS:
            n_match = 2 * w
            mse_a, mse_s, mse_s10 = [], [], []
            for k in range(args.k_nets):
                Ws = build_mlp(w, d, seed=20_000 + 97 * k)
                truth = mc_final_mean(Ws, args.n_truth, device, gen)
                a = analytic_final_mean(Ws)
                s = mc_final_mean(Ws, n_match, device, gen)
                s10 = mc_final_mean(Ws, 10 * n_match, device, gen)
                mse_a.append(float(np.mean((a - truth) ** 2)))
                mse_s.append(float(np.mean((s - truth) ** 2)))
                mse_s10.append(float(np.mean((s10 - truth) ** 2)))
            cell = {
                "mse_analytic": float(np.mean(mse_a)),
                "mse_sampling_matched": float(np.mean(mse_s)),
                "mse_sampling_10x": float(np.mean(mse_s10)),
                "n_matched": n_match,
                "ratio": float(np.mean(mse_a) / np.mean(mse_s)),
                # per-net records + median stats: per-net MSEs are heavy-tailed
                # (MC max/median 10-22x at depth), so mean-based cell ratios
                # are unstable at small K — medians are the robust headline
                "per_net_analytic": [float(v) for v in mse_a],
                "per_net_sampling": [float(v) for v in mse_s],
                "ratio_median": float(np.median(mse_a) / np.median(mse_s)),
            }
            cells[f"w{w}_d{d}"] = cell
            winner = "ANALYTIC" if cell["ratio"] < 1 else "sampling"
            print(f"w={w:<5} d={d:<3} analytic {cell['mse_analytic']:.3e}  "
                  f"matched-MC {cell['mse_sampling_matched']:.3e}  "
                  f"ratio {cell['ratio']:.3f}  -> {winner}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    OUT.write_text(json.dumps(
        {"widths": WIDTHS, "depths": DEPTHS, "k_nets": args.k_nets,
         "n_truth": args.n_truth,
         "flop_convention": "chain=4dw^3, sample=2dw^2 -> n_matched=2w",
         "cells": cells,
         "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=2))
    print(f"\nwrote {OUT}")

    # crossover contour: largest d where analytic wins, per width
    print("\nd*(w) = deepest depth where analytic beats FLOP-matched sampling:")
    for w in WIDTHS:
        dstar = max([d for d in DEPTHS if cells[f"w{w}_d{d}"]["ratio"] < 1],
                    default=None)
        print(f"  w={w:<5} d* = {dstar}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--k-nets", type=int, default=12)
    p.add_argument("--n-truth", type=int, default=2_000_000)
    p.add_argument("--depths", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    main()
