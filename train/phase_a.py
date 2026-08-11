"""
phase_a.py — Phase A overnight orchestrator: statistical hardening of the
corpus (~20 seeds per config).

Per config, sequentially:
  1. train missing seeds (train_mlp.py skips seeds whose provenance exists)
  2. weight census + directional consistency (while .npz are local)
  3. upload to james-ra-henry/MZC-Corpus, verify, prune local .npz

Disk stays bounded at ~one config of .npz (~300 MB). Cross-run analyses
(state trajectories, activation census, sweep summary) are NOT run here —
run them afterwards on whichever runs matter; corpus_io re-downloads on
demand. If a hopper task id is given, heartbeats before each config.

Existing 3-seed families are extended in place (same run_id, seeds 0-19).

  python train/phase_a.py --dry-run          # show the plan + time estimate
  python train/phase_a.py                    # run everything
  python train/phase_a.py --only sweep_c3_head sweep_c8_head
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
SEEDS = list(range(20))

# (run_id, extra train_mlp args, est. GPU-min per net)
CONFIGS: list[tuple[str, list[str], float]] = [
    # C sweep at sep 3.0 (existing 3-seed families extended in place)
    ("sweep_c2_head",       ["--classes", "2"], 5.5),
    ("sweep_c3_head",       ["--classes", "3"], 5.5),
    ("sweep_c5_head",       ["--classes", "5"], 5.5),
    ("sweep_c8_head",       ["--classes", "8"], 5.5),
    ("probe_d32_c10_head",  ["--classes", "10"], 5.5),
    ("sweep_c15_head",      ["--classes", "15"], 5.5),
    ("sweep_c20_head",      ["--classes", "20"], 5.5),
    ("sweep_c25_head",      ["--classes", "25"], 5.5),
    ("sweep_c32_head",      ["--classes", "32"], 5.5),
    ("sweep_c40_head",      ["--classes", "40"], 5.5),
    ("sweep_c50_head",      ["--classes", "50"], 5.5),
    # separation sweep at C=10
    ("sweep_c10_sep15",     ["--classes", "10", "--separation", "1.5"], 5.5),
    ("sweep_c10_sep20",     ["--classes", "10", "--separation", "2.0"], 5.5),
    ("sweep_c10_sep45",     ["--classes", "10", "--separation", "4.5"], 5.5),
    ("sweep_c10_sep60",     ["--classes", "10", "--separation", "6.0"], 5.5),
    # real data
    ("mnist_d32",           ["--task", "mnist"], 2.8),
    # weight decay at meaningful wd x lr x steps (1.8 and 6.0 nats of decay)
    ("sweep_c10_wd03",      ["--classes", "10", "--weight-decay", "0.3"], 5.5),
    ("sweep_c10_wd1",       ["--classes", "10", "--weight-decay", "1.0"], 5.5),
]

BASE_ARGS = ["--depth", "32", "--steps", "20000", "--lr", "3e-4",
             "--eval-every", "2000", "--seeds"] + [str(s) for s in SEEDS]


def missing_seeds(run_id: str) -> list[int]:
    run_dir = CORPUS_DIR / run_id
    return [s for s in SEEDS if not (run_dir / f"net_{s:04d}.json").exists()]


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    configs = [c for c in CONFIGS if not args.only or c[0] in args.only]
    todo = [(rid, extra, est, missing_seeds(rid)) for rid, extra, est in configs]
    total_nets = sum(len(m) for *_, m in todo)
    total_h = sum(len(m) * est for _, _, est, m in todo) / 60
    print(f"Phase A plan: {len(todo)} configs, {total_nets} nets to train, "
          f"~{total_h:.1f} GPU-hours\n")
    for rid, _, est, m in todo:
        print(f"  {rid:<22} {len(m):>3} nets to train "
              f"(~{len(m) * est / 60:.1f} h)")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (rid, extra, est, m) in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {rid} ({time.time()-t0:.0f}s elapsed) ===",
              flush=True)
        if args.hopper_task:
            subprocess.run(["hopper", "task", "heartbeat", args.hopper_task,
                            "--expect", "3h"],
                           cwd=str(Path.home() / "Source" / "Rosetta_Program"),
                           check=False)
        if m:
            run([sys.executable, "train/train_mlp.py", "--run-id", rid]
                + BASE_ARGS + extra)
        run([sys.executable, "census/run_census.py", "--run-id", rid])
        run([sys.executable, "census/directional_consistency.py",
             "--run-id", rid])
        run([sys.executable, "-c",
             f"import sys; sys.path.insert(0, 'train'); "
             f"from corpus_io import upload_run; upload_run('{rid}', prune=True)"])
    print(f"\nPhase A complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hopper-task", default="t4b9971d")
    args = p.parse_args()
    main()
