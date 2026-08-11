"""
train_mlp.py — Train MLP classifiers at ARC's exact Phase-1 architecture.

Architecture (must stay analytically comparable to the null baseline — never
change these to rescue convergence; scale the task instead):
  - depth x width square MLP, no input embedding, no output head
  - bias-free Linear everywhere
  - He-Gaussian init N(0, 2/fan_in) per element
  - ReLU after every layer, INCLUDING the last

Readout — two corpus families (--readout, recorded in provenance):
  - "head" (default): a bias-free Linear(width, C) head on top of the full
    32-layer stack, reading the POST-ReLU final-layer output (ARC's measured
    quantity). The head sits outside the censused stack — it never touches
    the stack's forward pass or the null — but routes gradient to every
    column of every square matrix, matching how nets are trained in use.
    Head weights are stored separately (head_w) and are not census objects.
  - "columns": the first C of the width output units are class scores, CE on
    the PRE-ReLU final layer. Leaves the final layer's other columns with
    exactly zero gradient — bit-frozen at init — which makes them a built-in
    negative control (a guaranteed-null patch inside a trained net).

Weights are saved in ARC's (in, out) storage convention (forward = x @ W), so
corpus files are directly consumable by null_baseline/analytic_vacuum.py and
census/manifold_detector.py. Both init and final weights are stored: deviation
from own-init and deviation from the analytic null are separate quantities.

Each net gets a provenance JSON next to its .npz: task, hyperparameters,
seeds, convergence outcome, accuracy trajectory, git commit, timestamp.

Usage:
  python train/train_mlp.py --run-id probe_d32 --depth 32 --classes 10 \
      --separation 3.0 --seeds 0 1 2 --steps 20000 --lr 3e-4
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


class ArcMLP(torch.nn.Module):
    """ARC Phase-1 spec: square, bias-free, He-Gaussian, ReLU every layer.

    Optionally carries a bias-free linear head (readout="head"). The head is
    initialized AFTER the stack from the same generator, so two nets with the
    same init_seed have bit-identical stack inits across readout modes.
    """

    def __init__(self, width: int, depth: int, init_seed: int,
                 head_classes: int | None = None):
        super().__init__()
        self.width, self.depth = width, depth
        gen = torch.Generator().manual_seed(init_seed)
        layers = []
        for _ in range(depth):
            lin = torch.nn.Linear(width, width, bias=False)
            with torch.no_grad():
                lin.weight.normal_(0.0, (2.0 / width) ** 0.5, generator=gen)
            layers.append(lin)
        self.layers = torch.nn.ModuleList(layers)
        if head_classes is not None:
            self.head = torch.nn.Linear(width, head_classes, bias=False)
            with torch.no_grad():
                self.head.weight.normal_(0.0, (2.0 / width) ** 0.5, generator=gen)
        else:
            self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """head mode: class logits via head(post-ReLU final layer).
        columns mode: PRE-ReLU values of the final layer (slice [:, :C])."""
        for lin in self.layers[:-1]:
            x = F.relu(lin(x))
        if self.head is not None:
            return self.head(F.relu(self.layers[-1](x)))
        return self.layers[-1](x)

    def weights_arc_convention(self) -> list[np.ndarray]:
        """(in, out) storage, float32 — matches build_mlp / analytic_vacuum."""
        return [lin.weight.detach().cpu().numpy().T.copy() for lin in self.layers]


def evaluate(model: ArcMLP, x: torch.Tensor, y: torch.Tensor, n_classes: int,
             batch: int = 4096) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = model(x[i:i + batch])[:, :n_classes]  # no-op slice in head mode
            correct += (logits.argmax(dim=1) == y[i:i + batch]).sum().item()
    model.train()
    return correct / len(x)


def train_one(task: GMMTask, init_seed: int, data_seed: int, steps: int,
              batch_size: int, lr: float, warmup: int, device: str,
              eval_every: int, readout: str, val_size: int = 20_000) -> dict:
    head_classes = task.n_classes if readout == "head" else None
    model = ArcMLP(task.dim, args.depth, init_seed,
                   head_classes=head_classes).to(device)
    init_weights = model.weights_arc_convention()

    if args.weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))

    rng = np.random.default_rng(data_seed)
    sample_val = getattr(task, "sample_val", task.sample)
    xv, yv = sample_val(val_size, np.random.default_rng(data_seed + 500_000))
    xv = torch.from_numpy(xv).to(device)
    yv = torch.from_numpy(yv).to(device)

    trajectory = []  # (step, val_acc)
    t0 = time.time()
    for step in range(steps):
        x, y = task.sample(batch_size, rng)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        logits = model(x)[:, :task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            acc = evaluate(model, xv, yv, task.n_classes)
            trajectory.append((step, round(acc, 4)))
            print(f"  step {step:6d}  loss {loss.item():.4f}  val_acc {acc:.4f}",
                  flush=True)
            if not np.isfinite(loss.item()):
                return {"outcome": "diverged", "trajectory": trajectory,
                        "final_val_acc": acc, "wall_seconds": time.time() - t0,
                        "init_weights": init_weights, "final_weights": None}

    final_acc = trajectory[-1][1]
    chance = 1.0 / task.n_classes
    outcome = ("converged" if final_acc >= task.bayes_accuracy - 0.03
               else "partial" if final_acc >= chance + 0.10
               else "stalled")
    head_w = (model.head.weight.detach().cpu().numpy().T.copy()
              if model.head is not None else None)
    return {"outcome": outcome, "trajectory": trajectory,
            "final_val_acc": final_acc, "wall_seconds": time.time() - t0,
            "init_weights": init_weights, "head_weights": head_w,
            "final_weights": model.weights_arc_convention()}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    run_dir = CORPUS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "gmm":
        task = make_gmm_task(dim=args.width, n_classes=args.classes,
                             separation=args.separation, seed=args.task_seed)
    else:
        from mnist_task import make_mnist_task
        task = make_mnist_task(dim=args.width, seed=args.task_seed)
    print(f"task: {task.describe()}")

    for seed in args.seeds:
        name = f"net_{seed:04d}"
        if (run_dir / f"{name}.json").exists() and not args.overwrite:
            print(f"{name}: exists, skipping")
            continue
        print(f"{name}: depth={args.depth} readout={args.readout} lr={args.lr} "
              f"steps={args.steps} batch={args.batch_size} device={device}")
        result = train_one(task, init_seed=seed, data_seed=seed + 1_000_000,
                           steps=args.steps, batch_size=args.batch_size,
                           lr=args.lr, warmup=args.warmup, device=device,
                           eval_every=args.eval_every, readout=args.readout)

        arrays = {f"init_w{i}": w for i, w in enumerate(result["init_weights"])}
        if result["final_weights"] is not None:
            arrays |= {f"w{i}": w for i, w in enumerate(result["final_weights"])}
        if result.get("head_weights") is not None:
            arrays["head_w"] = result["head_weights"]
        np.savez_compressed(run_dir / f"{name}.npz", **arrays)

        provenance = {
            "run_id": args.run_id,
            "net": name,
            "architecture": {"width": args.width, "depth": args.depth,
                             "bias": False, "activation": "relu_all_layers",
                             "init": "he_gaussian_2_over_fanin",
                             "weight_convention": "(in, out); forward = x @ W"},
            "task": task.describe(),
            "training": {"loss": ("cross_entropy_head_post_relu"
                                  if args.readout == "head"
                                  else "cross_entropy_pre_relu_final_layer"),
                         "readout": (f"bias_free_head_{args.width}x{task.n_classes}"
                                     if args.readout == "head"
                                     else f"first_{task.n_classes}_of_{args.width}_units"),
                         "optimizer": ("adamw" if args.weight_decay > 0 else "adam"),
                         "weight_decay": args.weight_decay, "lr": args.lr,
                         "warmup_steps": args.warmup, "steps": args.steps,
                         "batch_size": args.batch_size,
                         "init_seed": seed, "data_seed": seed + 1_000_000,
                         "device": device},
            "outcome": result["outcome"],
            "final_val_acc": result["final_val_acc"],
            "val_acc_trajectory": result["trajectory"],
            "wall_seconds": round(result["wall_seconds"], 1),
            "git_commit": git_commit(),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        (run_dir / f"{name}.json").write_text(json.dumps(provenance, indent=2))
        print(f"{name}: {result['outcome']} val_acc={result['final_val_acc']:.4f} "
              f"({result['wall_seconds']:.0f}s)")

    if args.upload:
        from corpus_io import upload_run
        upload_run(args.run_id, prune=args.prune)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-id", required=True)
    p.add_argument("--task", default="gmm", choices=["gmm", "mnist"])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--separation", type=float, default=3.0)
    p.add_argument("--task-seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--readout", default="head", choices=["head", "columns"])
    p.add_argument("--upload", action="store_true",
                   help="upload run to the MZC-Corpus HF dataset after training")
    p.add_argument("--prune", action="store_true",
                   help="after verified upload, delete local .npz (JSONs kept)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main()
