"""
phase2_wall_c50_w1024_lr1e4.py -- C50(width=1024) wall, lr=1e-4 rerun.

Plan: notes/2026-09-08_c50_w1024_wall_plan.md ("lr=1e-4 rerun" section).
The lr=3e-4 pass (train/phase2_wall_c50_w1024.py, Hopper te44c55be, complete
2026-09-12) measured C35:16/16, C40:16/16, C45:11/16, C50:8/16, C55:10/16 --
falsifying the pre-registered [40,43] crossing (it sits at/above C=50
instead) and, per the plan's own gate condition, C=55 not reading cleanly
collapsed triggers this rerun. Matches the w=512 precedent exactly: that
architecture's fixed-lr=3e-4 "wall" (C=64 total stall, 0.05 acc) was largely
an optimization artifact -- lr=1e-4 alone took it to 16/16 partial at 0.43.

Same full grid as the lr=3e-4 pass (not just the affected configs) so the
result is a clean, directly-comparable "tuned lr" curve, matching how the
existing w<=512 table rows were themselves built at per-width tuned lr from
the start -- a patchwork of tuned-only-where-needed would mix methodologies
within one table entry. Only --lr changes (1e-4 instead of 3e-4); same 16
seeds, wd=0, width=1024, depth=16, 20k steps. Distinct run-id prefix
(p2_w1024_d16_wall_lr1e4_c*) so it doesn't collide with the lr=3e-4 corpus.

Launch is manual (James's call), not scheduled via cron:
  python train/phase2_wall_c50_w1024_lr1e4.py --dry-run
  python train/phase2_wall_c50_w1024_lr1e4.py --hopper-task <id>
  python train/phase2_wall_c50_w1024_lr1e4.py --only p2_w1024_d16_wall_lr1e4_c50
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
    ("p2_w1024_d16_wall_lr1e4_c35", 35, 16.0),
    ("p2_w1024_d16_wall_lr1e4_c40", 40, 16.0),
    ("p2_w1024_d16_wall_lr1e4_c45", 45, 16.0),
    ("p2_w1024_d16_wall_lr1e4_c50", 50, 16.0),
    ("p2_w1024_d16_wall_lr1e4_c55", 55, 16.0),
]

LR = "1e-4"  # tuned pass -- matches the w=512 precedent that dissolved its wall
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
        print(f"  {rid:<30} C={c:<3} {len(m):>2} nets to train "
              f"(~{len(m) * est / 60:.1f} h)")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (rid, c, est, m) in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {rid} ({time.time()-t0:.0f}s elapsed) ===",
              flush=True)
        if args.hopper_task:
            # Heartbeat is a nice-to-have, not essential -- must never crash
            # the run (seen: cron's minimal PATH has no pyenv shims, so bare
            # "hopper" raises FileNotFoundError and would kill the whole
            # sweep before training a single net, 2026-09-08).
            try:
                subprocess.run(["hopper", "task", "heartbeat", args.hopper_task,
                                "--expect", "3h"],
                               cwd=str(REPO_ROOT), check=False)
            except OSError as e:
                print(f"!! hopper heartbeat failed (non-fatal): {e}", flush=True)
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
    print(f"\nC50(w=1024) lr=1e-4 pass complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hopper-task", default=None)
    args = p.parse_args()
    main()
