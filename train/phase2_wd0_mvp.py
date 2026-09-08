"""
phase2_wd0_mvp.py — wd=0 arm at ARC Phase-2 architecture (1024x16).

Sibling of phase2_mvp.py (which ran the same 11-config C-sweep at wd=0.3).
FINDINGS.md F1 flags the gap this closes: every existing Phase-2 net used
wd=0.3, which per F7 puts the corpus in the bulk-annihilated regime where
`significant_dims` is unreliable -- so the existing 88-net corpus only shows
the *qualitative* rank collapse (via effective_dim), not a re-verification of
the *exact* C-1 integer law at the new architecture the way Phase-1's
original (wd=0) C-sweep did. This arm is that re-verification: same 11 C
values, same seed count, same everything except --weight-decay 0 and a
distinct run-id prefix (p2_w1024_d16_wd0_c*) so it doesn't collide with the
existing wd=0.3 corpus already on HF.

  python train/phase2_wd0_mvp.py --dry-run
  python train/phase2_wd0_mvp.py
  python train/phase2_wd0_mvp.py --only p2_w1024_d16_wd0_c10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
SEEDS = list(range(8))

# (run_id, n_classes, est. GPU-min per net -- measured on this box, wd=0.3 arm)
CONFIGS: list[tuple[str, int, float]] = [
    ("p2_w1024_d16_wd0_c2", 2, 15.0),
    ("p2_w1024_d16_wd0_c3", 3, 15.0),
    ("p2_w1024_d16_wd0_c5", 5, 15.0),
    ("p2_w1024_d16_wd0_c8", 8, 15.0),
    ("p2_w1024_d16_wd0_c10", 10, 15.0),
    ("p2_w1024_d16_wd0_c15", 15, 15.0),
    ("p2_w1024_d16_wd0_c20", 20, 15.0),
    ("p2_w1024_d16_wd0_c25", 25, 15.0),
    ("p2_w1024_d16_wd0_c32", 32, 15.0),
    ("p2_w1024_d16_wd0_c40", 40, 15.0),
    ("p2_w1024_d16_wd0_c50", 50, 15.0),
]

BASE_ARGS = ["--task", "gmm", "--separation", "3.0", "--weight-decay", "0",
             "--width", "1024", "--depth", "16", "--steps", "20000",
             "--lr", "3e-4", "--eval-every", "4000", "--seeds"] + [str(s) for s in SEEDS]


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
    print(f"Phase 2 wd=0 arm plan: {len(todo)} configs, {total_nets} nets to train, "
          f"~{total_h:.1f} GPU-hours\n")
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
        # Same upload-resilience pattern as phase2_mvp.py: a failed upload
        # must not kill the whole sweep -- retry 3x, then move on. Local
        # .npz are only ever deleted after upload_run's own verify passes.
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
    print(f"\nPhase 2 wd=0 arm complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hopper-task", default=None)
    args = p.parse_args()
    main()
