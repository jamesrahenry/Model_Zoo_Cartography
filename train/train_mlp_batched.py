"""
train_mlp_batched.py — Train many ARC-spec MLPs simultaneously as one stacked
batched-matmul computation.

The nets are tiny (256×256 matmuls); one net leaves the GPU almost idle. This
trainer stacks B nets into (B, w, w) parameter tensors per layer and trains
them in one process: forward is bmm over the net axis, per-net parameters are
disjoint so gradients never mix, and Adam is elementwise so each net follows
exactly the trajectory it would follow alone (with its own data stream —
per-net RNGs match train_mlp.py's data_seed convention; loss is the SUM of
per-net mean-CE so each net's gradient equals its solo gradient).

Init is bit-identical to train_mlp.ArcMLP for the same seed (same generator
draw sequence), so batched and sequential corpora pair exactly.

Output schema (npz + provenance JSON per net) matches train_mlp.py; provenance
records trainer="batched". Readout: head mode only (the corpus default).

Usage:
  python train/train_mlp_batched.py --run-id phase_b_x --classes 10 \
      --seeds 0 1 2 ... 31 --steps 20000 [--upload --prune]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from gmm_task import GMMTask, make_gmm_task

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"


def init_stacked(width: int, depth: int, n_classes: int,
                 seeds: list[int]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """He-Gaussian init, bit-identical per net to train_mlp.ArcMLP(seed)."""
    layers = [torch.empty(len(seeds), width, width) for _ in range(depth)]
    heads = torch.empty(len(seeds), width, n_classes)
    std = (2.0 / width) ** 0.5
    for b, seed in enumerate(seeds):
        gen = torch.Generator().manual_seed(seed)
        for l in range(depth):
            # ArcMLP draws nn.Linear.weight of shape (out, in) = (w, w);
            # our stacked forward uses x @ W with W = weight.T
            wt = torch.empty(width, width).normal_(0.0, std, generator=gen)
            layers[l][b] = wt.T
        head = torch.empty(n_classes, width).normal_(0.0, std, generator=gen)
        heads[b] = head.T
    return layers, heads


def evaluate(layers, heads, xv, yv, batch: int = 4096) -> list[float]:
    """Per-net val accuracy. xv: (B, n, w) tensor, yv: (B, n)."""
    B, n, _ = xv.shape
    correct = torch.zeros(B, device=xv.device)
    with torch.no_grad():
        for i in range(0, n, batch):
            h = xv[:, i:i + batch, :]
            for W in layers:
                h = F.relu(torch.bmm(h, W))
            logits = torch.bmm(h, heads)
            correct += (logits.argmax(dim=2) == yv[:, i:i + batch]).sum(dim=1)
    return (correct / n).tolist()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    if args.tf32 and device == "cuda":
        # TF32 bmm on Ada: ~10-bit mantissa in the matmul inner loop. Weights,
        # activations, and Adam state stay fp32; recorded in provenance.
        torch.set_float32_matmul_precision("high")
    run_dir = CORPUS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = [s for s in args.seeds
             if not (run_dir / f"net_{s:04d}.json").exists() or args.overwrite]
    if not seeds:
        print("all seeds already trained")
        return
    B = len(seeds)

    if args.task == "gmm":
        task = make_gmm_task(dim=args.width, n_classes=args.classes,
                             separation=args.separation, seed=args.task_seed)
    else:
        from mnist_task import make_mnist_task
        task = make_mnist_task(dim=args.width, seed=args.task_seed,
                               dataset=args.task)
    print(f"task: {task.describe()}")
    print(f"batched: {B} nets, depth {args.depth}, device {device}")

    layers, heads = init_stacked(args.width, args.depth, task.n_classes, seeds)
    init_weights = [[layers[l][b].numpy().copy() for l in range(args.depth)]
                    for b in range(B)]
    layers = [torch.nn.Parameter(W.to(device)) for W in layers]
    heads = torch.nn.Parameter(heads.to(device))

    if args.weight_decay > 0:
        opt = torch.optim.AdamW(list(layers) + [heads], lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(list(layers) + [heads], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    # GPU-side sampler: CPU numpy sampling for B nets starves the GPU (it was
    # the throughput bottleneck at B=32). One device generator; streams are
    # independent across nets but do NOT replicate train_mlp.py's per-net
    # numpy streams — recorded in provenance as sampler="gpu_batched".
    gen_dev = torch.Generator(device=device).manual_seed(args.task_seed + 42)
    if args.task == "gmm":
        means_t = torch.from_numpy(task.means).float().to(device)
        whiten_t = torch.from_numpy(task.whiten).float().to(device)

        def sample_gpu(n: int) -> tuple[torch.Tensor, torch.Tensor]:
            y = torch.randint(task.n_classes, (B, n), device=device,
                              generator=gen_dev)
            z = torch.randn(B, n, args.width, device=device, generator=gen_dev)
            return means_t[y] + z @ whiten_t, y
    else:  # mnist: index-select from the device-resident whitened train split
        xtr_t = torch.from_numpy(task.x_train).to(device)
        ytr_t = torch.from_numpy(task.y_train).to(device)

        def sample_gpu(n: int) -> tuple[torch.Tensor, torch.Tensor]:
            idx = torch.randint(len(xtr_t), (B, n), device=device,
                                generator=gen_dev)
            return xtr_t[idx], ytr_t[idx]

    sample_val = getattr(task, "sample_val", task.sample)
    val = [sample_val(20_000, np.random.default_rng(seed + 1_500_000))
           for seed in seeds]
    xv = torch.from_numpy(np.stack([v[0] for v in val])).to(device)
    yv = torch.from_numpy(np.stack([v[1] for v in val])).to(device)

    trajectories: list[list[tuple[int, float]]] = [[] for _ in range(B)]
    t0 = time.time()
    for step in range(args.steps):
        x, y = sample_gpu(args.batch_size)
        h = x
        for W in layers:
            h = F.relu(torch.bmm(h, W))
        logits = torch.bmm(h, heads)
        # sum of per-net mean-CE: each net's gradient equals its solo gradient
        loss = F.cross_entropy(logits.reshape(-1, task.n_classes),
                               y.reshape(-1), reduction="mean") * B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            accs = evaluate(layers, heads, xv, yv)
            for b in range(B):
                trajectories[b].append((step, round(accs[b], 4)))
            print(f"step {step:6d}  loss/net {loss.item()/B:.4f}  "
                  f"val_acc mean {np.mean(accs):.4f} "
                  f"[{min(accs):.4f}, {max(accs):.4f}]  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if not np.isfinite(loss.item()):
                raise RuntimeError("diverged")

    wall = time.time() - t0
    chance = 1.0 / task.n_classes
    for b, seed in enumerate(seeds):
        name = f"net_{seed:04d}"
        final_acc = trajectories[b][-1][1]
        outcome = ("converged" if final_acc >= task.bayes_accuracy - 0.03
                   else "partial" if final_acc >= chance + 0.10
                   else "stalled")
        arrays = {f"init_w{l}": init_weights[b][l] for l in range(args.depth)}
        arrays |= {f"w{l}": layers[l][b].detach().cpu().numpy()
                   for l in range(args.depth)}
        arrays["head_w"] = heads[b].detach().cpu().numpy()
        np.savez_compressed(run_dir / f"{name}.npz", **arrays)
        provenance = {
            "run_id": args.run_id, "net": name,
            "architecture": {"width": args.width, "depth": args.depth,
                             "bias": False, "activation": "relu_all_layers",
                             "init": "he_gaussian_2_over_fanin",
                             "weight_convention": "(in, out); forward = x @ W"},
            "task": task.describe(),
            "training": {"trainer": "batched", "net_batch": B,
                         "loss": "cross_entropy_head_post_relu",
                         "readout": f"bias_free_head_{args.width}x{task.n_classes}",
                         "optimizer": ("adamw" if args.weight_decay > 0 else "adam"),
                         "weight_decay": args.weight_decay,
                         "lr": args.lr, "warmup_steps": args.warmup,
                         "steps": args.steps, "batch_size": args.batch_size,
                         "init_seed": seed,
                         "sampler": "gpu_batched",
                         "sampler_seed": args.task_seed + 42,
                         "matmul_precision": ("tf32" if args.tf32 and device == "cuda"
                                              else "fp32"),
                         "val_seed": seed + 1_500_000, "device": device},
            "outcome": outcome, "final_val_acc": final_acc,
            "val_acc_trajectory": trajectories[b],
            "wall_seconds_batch_total": round(wall, 1),
            "git_commit": git_commit(),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        (run_dir / f"{name}.json").write_text(json.dumps(provenance, indent=2))
        print(f"{name}: {outcome} val_acc={final_acc:.4f}")

    print(f"\n{B} nets in {wall:.0f}s = {wall/B:.1f}s/net "
          f"(sequential baseline ~330s/net)")
    if args.upload:
        from corpus_io import upload_run
        upload_run(args.run_id, prune=args.prune)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", required=True)
    p.add_argument("--task", default="gmm", choices=["gmm", "mnist", "fashion"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--separation", type=float, default=3.0)
    p.add_argument("--task-seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tf32", action="store_true",
                   help="TF32 matmul (Ada): faster, ~10-bit matmul mantissa")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()
    main()
