"""
phase2_wall_c50_w1024.py — C50(width=1024) wall measurement (NOT YET RUN).

Plan: notes/2026-09-08_c50_w1024_wall_plan.md. FINDINGS.md F6 flags that the
Phase-2 MVP's C=40/50 partial outcomes are only a qualitative wall replication
(8 seeds, no lr tuning) -- not a real width=1024 point for the log-frontier
table (F6's existing w=64/128/256/512 entries used 16-32 seeds and per-width
lr tuning; w=512 in particular had a "wall" that was largely a fixed-lr
optimization artifact, dissolving at lr=1e-4).

Grid brackets the pre-registered prediction C50(1024) in [40, 43] (two
extrapolations off the existing table's decelerating per-doubling increment:
naive average slope -> ~42.5, continued deceleration -> ~40-41): C in
{35, 40, 45, 50, 55}, 16 seeds/config, wd=0 (matches the base C-sweep
convention, not the F7 weight-decay axis), lr=3e-4 first -- gate a tuned
lr=1e-4 rerun on whether convergence looks anomalous (see the plan note).

  python train/phase2_wall_c50_w1024.py --dry-run
  python train/phase2_wall_c50_w1024.py
  python train/phase2_wall_c50_w1024.py --only p2_w1024_d16_wall_c45
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
SEEDS = list(range(16))

# (run_id, n_classes, est. GPU-min per net -- measured on this box)
CONFIGS: list[tuple[str, int, float]] = [
    ("p2_w1024_d16_wall_c35", 35, 16.0),
    ("p2_w1024_d16_wall_c40", 40, 16.0),
    ("p2_w1024_d16_wall_c45", 45, 16.0),
    ("p2_w1024_d16_wall_c50", 50, 16.0),
    ("p2_w1024_d16_wall_c55", 55, 16.0),
]

LR = "3e-4"  # default pass; gate a tuned lr=1e-4 rerun per the plan note
BASE_ARGS = ["--task", "gmm", "--separation", "3.0", "--weight-decay", "0",
             "--width", "1024", "--depth", "16", "--steps", "20000",
             "--lr", LR, "--eval-every", "4000", "--seeds"] + [str(s) for s in SEEDS]


def missing_seeds(run_id: str) -> list[int]:
    run_dir = CORPUS_DIR / run_id
    return [s for s in SEEDS if not (run_dir / f"net_{s:04d}.json").exists()]


def run(cmd: list[str], check: bool = True) -> bool:
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode == 0


def main() -> None:
    configs = [c for c in CONFIGS if not args.only or c[0] in args.only]
    todo = [(rid, c, est, missing_seeds(rid)) for rid, c, est in configs]
    total_nets = sum(len(m) for *_, m in todo)
    total_h = sum(len(m) * est for _, _, est, m in todo) / 60
    print(f"C50(w=1024) wall plan (lr={LR}): {len(todo)} configs, {total_nets} "
          f"nets to train, ~{total_h:.1f} GPU-hours\n")
    for rid, c, est, m in todo:
        print(f"  {rid:<26} C={c:<3} {len(m):>2} nets to train "
              f"(~{len(m) * est / 60:.1f} h)")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (rid, c, est, m) in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {rid} ({time.time()-t0:.0f}s elapsed) ===",
              flush=True)
        if args.hopper_task:
            subprocess.run(["hopper", "task", "heartbeat", args.hopper_task,
                            "--expect", "3h"],
                           cwd=str(REPO_ROOT), check=False)
        if m:
            run([sys.executable, "train/train_mlp.py", "--run-id", rid,
                 "--classes", str(c)] + BASE_ARGS)
        for attempt in range(3):
            ok = run([sys.executable, "-c",
                     f"import sys; sys.path.insert(0, 'train'); "
                     f"from corpus_io import upload_run; upload_run('{rid}', prune=True)"],
                     check=False)
            if ok:
                break
            print(f"!! upload failed for {rid} (attempt {attempt+1}/3)", flush=True)
        else:
            print(f"!! {rid} left un-pruned locally; re-run with --only {rid} to retry",
                  flush=True)
    print(f"\nC50(w=1024) wall pass complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hopper-task", default=None)
    args = p.parse_args()
    main()
