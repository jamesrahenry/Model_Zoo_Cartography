"""
weekend_sweep.py — Autonomous weekend driver: the lr-tuned w=512 transition
program + wall fill-ins, on the local RTX 500.

Stages (lr choices propagate automatically between stages):
  A. lr micro-tune at w=512: C=32 × lr {1e-4, 3e-5}, 8 seeds each
     (C=32 @ 3e-4 and C=64 @ {3e-4, 1e-4, 3e-5} already exist in the corpus).
  B. tuned transition sweep: C ∈ {32, 40, 48, 56, 64, 72} × 16 seeds at the
     per-C best lr (log-interpolated between the C=32 and C=64 winners).
  C. wall-or-budget arms: 60k steps at C=48 and C=64 (tuned lr), 16 seeds.
  D. fill-ins: w64 C∈{30, 32} and w128 C∈{32, 36} (32 seeds, default lr) —
     completes the per-width wall logistic fits; d64 @ lr 1e-4 60k (16 seeds)
     — do the near-miss partials converge with budget?

Per config: batched train → weight census → directional consistency →
upload-verify-prune. Heartbeats Hopper before each config. Resumable.

  python train/weekend_sweep.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def heartbeat() -> None:
    subprocess.run(["hopper", "task", "heartbeat", "t4b9971d", "--expect", "4h"],
                   cwd=str(Path.home() / "Source" / "Rosetta_Program"),
                   check=False)


def train_config(run_id: str, extra: list[str], seeds: list[int],
                 net_batch: int) -> None:
    missing = [s for s in seeds
               if not (CORPUS_DIR / run_id / f"net_{s:04d}.json").exists()]
    heartbeat()
    for lo in range(0, len(missing), net_batch):
        chunk = missing[lo:lo + net_batch]
        if chunk:
            run([sys.executable, "train/train_mlp_batched.py", "--run-id",
                 run_id, "--tf32", "--seeds"] + [str(s) for s in chunk] + extra)
    if missing:
        run([sys.executable, "census/run_census.py", "--run-id", run_id])
        run([sys.executable, "census/directional_consistency.py",
             "--run-id", run_id])
    run([sys.executable, "-c",
         f"import sys; sys.path.insert(0, 'train'); "
         f"from corpus_io import upload_run; upload_run('{run_id}', prune=True)"])


def mean_acc(run_id: str) -> float:
    accs = [json.loads(p.read_text())["final_val_acc"]
            for p in sorted((CORPUS_DIR / run_id).glob("net_*.json"))]
    return float(np.mean(accs)) if accs else float("nan")


LR_TAG = {3e-4: "3e-4", 1e-4: "1e-4", 3e-5: "3e-5"}


def main() -> None:
    t0 = time.time()
    plan_only = args.dry_run

    # ---- Stage A: complete the lr grid at C=32 (C=64 grid already exists) --
    stage_a = [("b1d_w512_c32_lr1e4", 1e-4), ("b1d_w512_c32_lr3e5", 3e-5)]
    for rid, lr in stage_a:
        print(f"\n=== A: {rid} ===", flush=True)
        if not plan_only:
            train_config(rid, ["--width", "512", "--classes", "32",
                               "--lr", f"{lr:g}"], list(range(8)), 8)

    # pick best lr per anchor C from all available data
    def best_lr(c: int, candidates: list[tuple[float, str]]) -> float:
        scored = [(mean_acc(rid), lr) for lr, rid in candidates
                  if (CORPUS_DIR / rid).exists()]
        scored = [(a, lr) for a, lr in scored if not np.isnan(a)]
        return max(scored)[1] if scored else 1e-4

    lr32 = best_lr(32, [(3e-4, "b1b_w512_c32"),
                        (1e-4, "b1d_w512_c32_lr1e4"),
                        (3e-5, "b1d_w512_c32_lr3e5")])
    lr64 = best_lr(64, [(3e-4, "b1_w512_c64"),
                        (1e-4, "b1c_w512_c64_lr1e4"),
                        (3e-5, "b1c_w512_c64_lr3e5")])
    print(f"\ntuned lr: C=32 -> {lr32:g}, C=64 -> {lr64:g}", flush=True)

    def lr_for(c: int) -> float:
        # log-interpolate between the two anchors, snap to the grid
        if lr32 == lr64:
            return lr32
        f = (np.log(c) - np.log(32)) / (np.log(64) - np.log(32))
        target = np.exp((1 - f) * np.log(lr32) + f * np.log(lr64))
        grid = np.array([3e-4, 1e-4, 3e-5])
        return float(grid[np.argmin(np.abs(np.log(grid) - np.log(target)))])

    # ---- Stage B: tuned transition sweep, 16 seeds ----
    for c in (32, 40, 48, 56, 64, 72):
        lr = lr_for(c)
        rid = f"b1d_w512_c{c}_t{LR_TAG[lr]}"
        print(f"\n=== B: {rid} (lr {lr:g}) ===", flush=True)
        if not plan_only:
            train_config(rid, ["--width", "512", "--classes", str(c),
                               "--lr", f"{lr:g}"], list(range(16)), 16)

    # ---- Stage C: wall-or-budget, 60k steps ----
    for c in (48, 64):
        lr = lr_for(c)
        rid = f"b1d_w512_c{c}_60k"
        print(f"\n=== C: {rid} (lr {lr:g}, 60k) ===", flush=True)
        if not plan_only:
            train_config(rid, ["--width", "512", "--classes", str(c),
                               "--lr", f"{lr:g}", "--steps", "60000",
                               "--eval-every", "10000"], list(range(16)), 16)

    # ---- Stage D: fill-ins ----
    fillins = [
        ("b1d_w64_c30",  ["--width", "64", "--classes", "30"], 32, 32),
        ("b1d_w64_c32",  ["--width", "64", "--classes", "32"], 32, 32),
        ("b1d_w128_c32", ["--width", "128", "--classes", "32"], 32, 32),
        ("b1d_w128_c36", ["--width", "128", "--classes", "36"], 32, 32),
        ("b1d_d64_lr1e4_60k", ["--depth", "64", "--lr", "1e-4",
                               "--steps", "60000", "--eval-every", "10000"],
         16, 16),
    ]
    for rid, extra, n_seeds, nb in fillins:
        print(f"\n=== D: {rid} ===", flush=True)
        if not plan_only:
            train_config(rid, extra, list(range(n_seeds)), nb)

    print(f"\nWEEKEND SWEEP COMPLETE ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main()
