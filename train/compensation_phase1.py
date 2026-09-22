"""
compensation_phase1.py -- Phase 1 of the distributed/compensatory-structure
experiment (notes/2026-09-22_distributed-compensatory-structure-gem-solitaire.md).

Question: does freezing one layer (preventing its usual collapse-to-task-rank)
cause OTHER layers to shift beyond normal seed-to-seed variance? Cheap,
falsifiable gate before any causal (concept-ablation) follow-up.

Two arms, same config otherwise (C=10, sep=3.0, wd=0, width=256, depth=32 --
the original ARC-spec architecture, cheap and already fully instrumented):
  - baseline: unconstrained, 16 seeds
  - freeze_l0: layer 0 (input) frozen at init via --freeze-layer 0, 16 seeds

L0 chosen because it's the layer MZC's own F1 law (input rank = C-1) says
should collapse; freezing it is the most direct echo of the Solitaire
LayerNorm result (which protected net.0 and saw the readout collapse harder
in exchange). Analysis (run separately, after training): z-score every
non-frozen layer's freeze_l0-arm effective_dim against the baseline arm's
own seed-to-seed null band (F2's q-clock convention, pointed at a
trained-baseline reference instead of random-init).

  python train/compensation_phase1.py --dry-run
  python train/compensation_phase1.py
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

BASE_ARGS = ["--task", "gmm", "--classes", "10", "--separation", "3.0",
             "--weight-decay", "0", "--width", "256", "--depth", "32",
             "--steps", "20000", "--lr", "3e-4", "--eval-every", "4000",
             "--tf32", "--seeds"] + [str(s) for s in SEEDS]

# (run_id, extra args, est. GPU-min for the WHOLE batched arm -- measured
# rate at this architecture, ~64s/net x 16 seeds)
ARMS: list[tuple[str, list[str], float]] = [
    ("comp_p1_baseline_c10", [], 17.0),
    ("comp_p1_freeze_l0_c10", ["--freeze-layer", "0"], 17.0),
]


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
    todo = [(rid, extra, est, missing_seeds(rid)) for rid, extra, est in ARMS]
    total_h = sum(est for _, _, est, m in todo if m) / 60
    print(f"Compensation Phase 1 plan: {len(todo)} arms, ~{total_h:.1f} GPU-hours\n")
    for rid, extra, est, m in todo:
        print(f"  {rid:<24} {len(m):>2} nets to train (~{est:.0f} min)")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (rid, extra, est, m) in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {rid} ({time.time()-t0:.0f}s elapsed) ===",
              flush=True)
        if args.hopper_task:
            try:
                subprocess.run(["hopper", "task", "heartbeat", args.hopper_task,
                                "--expect", "2h"],
                               cwd=str(REPO_ROOT), check=False)
            except OSError as e:
                print(f"!! hopper heartbeat failed (non-fatal): {e}", flush=True)
        if m:
            run([sys.executable, "train/train_mlp_batched.py", "--run-id", rid]
                + BASE_ARGS + extra)
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
    print(f"\nCompensation Phase 1 complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hopper-task", default=None)
    args = p.parse_args()
    main()
