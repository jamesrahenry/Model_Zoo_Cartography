"""
compensation_full_layer_sweep.py -- full-depth follow-up to Phase 1
(notes/2026-09-22_distributed-compensatory-structure-gem-solitaire.md).

Phase 1 froze only L0 and found the disruption stays local (layers 1-2
shift, z=-17.4/-8.9; layers 3-31 all |z|<2). Open question that raised:
is that localization an L0-specific artifact (it's the one layer with a
known, exact task role -- F1's C-1 law) or does freezing ANY layer produce
the same short, bounded disruption radius? This sweeps every layer, not a
sparse sample -- James's call ("the full layer check, not the cheap
version").

32 arms (freeze_layer=0..31), 16 seeds each, same config as Phase 1
(C=10, sep=3.0, wd=0, width=256, depth=32) so every arm is directly
comparable to the SAME baseline (comp_p1_baseline_c10, already trained,
not retrained here). Analysis (census/compensation_full_analysis.py, run
separately): for each arm, z-score every non-frozen layer against the
baseline's own null band, same method as Phase 1 -- assembles a full
32x32 disruption-radius matrix (which layer was frozen x which layer
shifted).

  python train/compensation_full_layer_sweep.py --dry-run
  python train/compensation_full_layer_sweep.py
  python train/compensation_full_layer_sweep.py --only 15 20 25
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
DEPTH = 32
EST_MIN_PER_ARM = 17.0  # measured rate at this architecture (Phase 1)

BASE_ARGS = ["--task", "gmm", "--classes", "10", "--separation", "3.0",
             "--weight-decay", "0", "--width", "256", "--depth", str(DEPTH),
             "--steps", "20000", "--lr", "3e-4", "--eval-every", "4000",
             "--tf32", "--seeds"] + [str(s) for s in SEEDS]


def run_id_for(layer: int) -> str:
    return f"comp_full_freeze_l{layer:02d}_c10"


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
    layers = args.only if args.only else list(range(DEPTH))
    todo = [(l, run_id_for(l), missing_seeds(run_id_for(l))) for l in layers]
    total_nets = sum(len(m) for *_, m in todo)
    total_h = sum(len(m) > 0 for *_, m in todo) * EST_MIN_PER_ARM / 60
    print(f"Full-layer sweep plan: {len(todo)} arms, {total_nets} nets to train, "
          f"~{total_h:.1f} GPU-hours\n")
    for l, rid, m in todo:
        print(f"  layer {l:>2}  {rid:<28} {len(m):>2} nets to train")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (l, rid, m) in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] freeze_layer={l} -> {rid} "
              f"({time.time()-t0:.0f}s elapsed) ===", flush=True)
        if args.hopper_task:
            try:
                subprocess.run(["hopper", "task", "heartbeat", args.hopper_task,
                                "--expect", "3h"],
                               cwd=str(REPO_ROOT), check=False)
            except OSError as e:
                print(f"!! hopper heartbeat failed (non-fatal): {e}", flush=True)
        if m:
            run([sys.executable, "train/train_mlp_batched.py", "--run-id", rid,
                 "--freeze-layer", str(l)] + BASE_ARGS)
        for attempt in range(3):
            ok = run([sys.executable, "-c",
                     f"import sys; sys.path.insert(0, 'train'); "
                     f"from corpus_io import upload_run; upload_run('{rid}', prune=True)"],
                     check=False)
            if ok:
                break
            print(f"!! upload failed for {rid} (attempt {attempt+1}/3)", flush=True)
        else:
            print(f"!! {rid} left un-pruned locally; re-run with --only {l} to retry",
                  flush=True)
    print(f"\nFull-layer sweep complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="only these layer indices (default: all 0-31)")
    p.add_argument("--hopper-task", default=None)
    args = p.parse_args()
    main()
