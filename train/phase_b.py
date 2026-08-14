"""
phase_b.py — Wave-1 Phase B orchestrator (batched trainer).

Config blocks (notes/2026-08-13_next-steps.md):
  B1 width sweep  — does the C−1 law hold across width; does the mid-net code
                    ceiling scale with width (wall-theory prediction)
  B2 depth sweep  — q-plateau persistence past the random fixed point; wall vs depth
  B3 wall core    — C ∈ {36, 44} at the standard architecture/budget

Per config: batched-train missing seeds (chunked to fit VRAM), weight census +
directional consistency while local, upload-verify-prune. Heartbeats Hopper
before each config. Resumable: completed seeds are skipped, completed configs
fly through.

  python train/phase_b.py --dry-run
  python train/phase_b.py [--only b2_d8_c10 ...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
SEEDS = list(range(32))

# (run_id, extra trainer args, net_batch, est. minutes per full config)
CONFIGS: list[tuple[str, list[str], int, float]] = [
    # B1 width sweep (wall probes chosen around ceiling ~ w/18 heuristic)
    ("b1_w64_c5",    ["--width", "64",  "--classes", "5"],  32, 10),
    ("b1_w64_c10",   ["--width", "64",  "--classes", "10"], 32, 10),
    ("b1_w64_c16",   ["--width", "64",  "--classes", "16"], 32, 10),
    ("b1_w128_c10",  ["--width", "128", "--classes", "10"], 32, 14),
    ("b1_w128_c20",  ["--width", "128", "--classes", "20"], 32, 14),
    ("b1_w128_c28",  ["--width", "128", "--classes", "28"], 32, 14),
    ("b1_w512_c10",  ["--width", "512", "--classes", "10"], 16, 140),
    ("b1_w512_c64",  ["--width", "512", "--classes", "64"], 16, 140),
    # B2 depth sweep at w=256, C=10
    ("b2_d8_c10",    ["--depth", "8"],  32, 10),
    ("b2_d16_c10",   ["--depth", "16"], 32, 19),
    ("b2_d48_c10",   ["--depth", "48"], 32, 55),
    ("b2_d64_c10",   ["--depth", "64"], 24, 100),
    # B3 wall core at standard architecture/budget
    ("b3_c36",       ["--classes", "36"], 32, 35),
    ("b3_c44",       ["--classes", "44"], 32, 35),
    # B1b (added mid-wave-1): locate the w=512 transition. C=64 STALLED at
    # ~0.05 (hard plateau, never lifted off — a different failure mode from
    # the soft wall at w=256, where partials reach 0.5-0.7). These probes
    # determine whether w=512 shows the same soft wall somewhere in (10, 64)
    # or a cliff straight to stall. 16 seeds each (transition location, not
    # fine statistics).
    ("b1b_w512_c32", ["--width", "512", "--classes", "32"], 16, 70),
    ("b1b_w512_c44", ["--width", "512", "--classes", "44"], 16, 70),
    ("b1b_w512_c56", ["--width", "512", "--classes", "56"], 16, 70),
]

# b1b probes run 16 seeds (transition location, not fine statistics)
SEEDS_PER_CONFIG = {"b1b_w512_c32": 16, "b1b_w512_c44": 16, "b1b_w512_c56": 16}


def missing_seeds(run_id: str) -> list[int]:
    run_dir = CORPUS_DIR / run_id
    seeds = SEEDS[:SEEDS_PER_CONFIG.get(run_id, len(SEEDS))]
    return [s for s in seeds if not (run_dir / f"net_{s:04d}.json").exists()]


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    configs = [c for c in CONFIGS if not args.only or c[0] in args.only]
    todo = [(rid, extra, nb, est, missing_seeds(rid))
            for rid, extra, nb, est in configs]
    total = sum(len(m) for *_, m in todo)
    total_h = sum(est * len(m) / len(SEEDS) for _, _, _, est, m in todo) / 60
    print(f"Phase B wave 1: {len(todo)} configs, {total} nets to train, "
          f"~{total_h:.1f} GPU-hours\n")
    for rid, _, nb, est, m in todo:
        print(f"  {rid:<16} {len(m):>3} nets (net_batch {nb}, ~{est} min full)")
    if args.dry_run:
        return

    t0 = time.time()
    for i, (rid, extra, nb, est, m) in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {rid} ({time.time()-t0:.0f}s elapsed) ===",
              flush=True)
        subprocess.run(["hopper", "task", "heartbeat", args.hopper_task,
                        "--expect", "4h"],
                       cwd=str(Path.home() / "Source" / "Rosetta_Program"),
                       check=False)
        for lo in range(0, len(m), nb):
            chunk = m[lo:lo + nb]
            if chunk:
                run([sys.executable, "train/train_mlp_batched.py",
                     "--run-id", rid, "--tf32",
                     "--seeds"] + [str(s) for s in chunk] + extra)
        run([sys.executable, "census/run_census.py", "--run-id", rid])
        run([sys.executable, "census/directional_consistency.py",
             "--run-id", rid])
        run([sys.executable, "-c",
             f"import sys; sys.path.insert(0, 'train'); "
             f"from corpus_io import upload_run; upload_run('{rid}', prune=True)"])
    print(f"\nPhase B wave 1 complete ({(time.time()-t0)/3600:.1f} h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hopper-task", default="t4b9971d")
    args = p.parse_args()
    main()
