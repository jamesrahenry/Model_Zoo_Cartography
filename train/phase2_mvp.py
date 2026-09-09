"""
phase2_mvp.py — MVP overnight sweep at ARC Phase-2 architecture (1024x16).

Extends MZC (currently 1,569 nets / 69 families, all at ARC Phase-1 spec
256x32) with a parallel population at ARC's actual Phase-2 spec: width 1024,
depth 16 (resolved 2026-08-23, see arc-whitebox-canary/phase2/SPEC_WATCH.md
U1). Scoped per notes/2026-09-06_mzc-phase2-sweep-scope.md: a reduced C sweep
(not full Phase-1 parity — that's 1-2+ weeks of overnights) at 8 seeds/config
instead of 16-32, using the sequential trainer (train_mlp.py) rather than the
batched one.

Sequential over batched, deliberately: a smoke test on this box (RTX 500 Ada,
4GB VRAM) showed train_mlp_batched.py at B=8/1024x16 sits at ~93% VRAM with
no headroom, and batched only writes output after ALL steps finish — an OOM
or crash anywhere in an hours-long run loses the whole batch. Sequential
writes one net at a time (resumable net-by-net, like phase_a.py already
relies on) and uses a trivial fraction of VRAM per net. Measured ~2x slower
in aggregate (15 min/net sequential vs ~5.5 min/net-equivalent batched at
this width) — worth it for an unattended run.

Per config, sequentially (same shape as phase_a.py):
  1. train missing seeds (train_mlp.py skips seeds whose provenance exists)
  2. upload to james-ra-henry/MZC-Corpus, verify, prune local .npz
No census step here — the MVP is corpus-building; analysis is a follow-on
once nets exist. If a hopper task id is given, heartbeats before each config.

  python train/phase2_mvp.py --dry-run          # show the plan + time estimate
  python train/phase2_mvp.py                    # run everything
  python train/phase2_mvp.py --only p2_w1024_d16_c10
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

# (run_id, n_classes, est. GPU-min per net -- measured on this box)
CONFIGS: list[tuple[str, int, float]] = [
    ("p2_w1024_d16_c2", 2, 15.0),
    ("p2_w1024_d16_c3", 3, 15.0),
    ("p2_w1024_d16_c5", 5, 15.0),
    ("p2_w1024_d16_c8", 8, 15.0),
    ("p2_w1024_d16_c10", 10, 15.0),
    ("p2_w1024_d16_c15", 15, 15.0),
    ("p2_w1024_d16_c20", 20, 15.0),
    ("p2_w1024_d16_c25", 25, 15.0),
    ("p2_w1024_d16_c32", 32, 15.0),
    ("p2_w1024_d16_c40", 40, 15.0),
    ("p2_w1024_d16_c50", 50, 15.0),
]

BASE_ARGS = ["--task", "gmm", "--separation", "3.0", "--weight-decay", "0.3",
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
    print(f"Phase 2 MVP plan: {len(todo)} configs, {total_nets} nets to train, "
          f"~{total_h:.1f} GPU-hours\n")
    for rid, c, est, m in todo:
        print(f"  {rid:<22} C={c:<3} {len(m):>2} nets to train "
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
        # Upload failures (seen: a transient xet/drvfs I/O error) must not
        # kill the whole multi-hour sweep -- local .npz are never deleted
        # unless upload_run's own verify step passes, so a failed upload
        # just means this config stays un-pruned locally; retry it by
        # rerunning with --only <rid> later.
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
    print(f"\nPhase 2 MVP sweep complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hopper-task", default=None)
    args = p.parse_args()
    main()
