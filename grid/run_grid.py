"""Train one weight-decay row of the preregistered grid, analyse every net, write per-net JSON (resumable).

usage: run_grid.py --wd 0.03 --classes 2 5 10 25 50 --seeds 0 1 2 3 4 5 --out OUT/results --weights-out OUT/weights [--resume DIR]
Per net writes  results/<run_id>/net_<seed>.json  and  results/<run_id>/net_<seed>_truth.npz;  for the first seed of each
cell also weights-out/<run_id>/net_<seed>.npz (w layers + head only; init is regenerable from the seed).
Cells already present in --resume (or --out) are skipped.  Weights of other seeds are never written out.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--wd", type=float, nargs="+", required=True); ap.add_argument("--classes", type=int, nargs="+", default=[10])
ap.add_argument("--task", default="gmm", choices=["gmm", "mnist", "fashion"]); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--weights-out"); ap.add_argument("--resume")
ap.add_argument("--width", type=int, default=1024); ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--truth-n", type=int, default=4_000_000)
a = ap.parse_args()
tg = lambda v: ("%g" % v).replace(".", "p").replace("-", "m").replace("+", "")
DEFAULT = (a.task == "gmm" and a.width == 1024 and a.depth == 16 and a.lr == 3e-4 and a.steps == 20000)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if a.resume and Path(a.resume).is_dir():
    src = Path(a.resume)
    for f in src.rglob("*"):
        if f.is_file():
            dst = out / f.relative_to(src); dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copy2(f, dst)
from analyze_net import analyze
for wd_, C in [(w, c) for w in a.wd for c in a.classes]:
    run_id = (f"grid_wd{tg(wd_)}_c{C}" if DEFAULT else
              f"grid_{a.task}_w{a.width}d{a.depth}_lr{tg(a.lr)}_s{a.steps}_wd{tg(wd_)}_c{C}")
    cell = out / run_id; cell.mkdir(parents=True, exist_ok=True)
    todo = [s for s in a.seeds if not (cell / f"net_{s:04d}.json").exists()]
    if not todo:
        print(f"[{run_id}] complete, skipping", flush=True); continue
    t0 = time.time()
    cmd = [sys.executable, "-u", str(REPO / "train" / "train_mlp_batched.py"), "--run-id", run_id, "--task", a.task,
           "--width", str(a.width), "--depth", str(a.depth), "--classes", str(C), "--weight-decay", str(wd_),
           "--separation", "3.0", "--seeds", *map(str, todo), "--steps", str(a.steps), "--lr", str(a.lr),
           "--eval-every", str(max(1, a.steps // 5))]
    print(f"[{run_id}] training {len(todo)} nets: {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))
    print(f"[{run_id}] trained in {time.time()-t0:.0f}s", flush=True)
    run_dir = REPO / "corpus" / run_id
    for s in todo:
        t1 = time.time()
        npz, js = run_dir / f"net_{s:04d}.npz", run_dir / f"net_{s:04d}.json"
        res, mean, se2 = analyze(npz, js, truth_n=a.truth_n)
        np.savez(cell / f"net_{s:04d}_truth.npz", mean=mean.astype(np.float32), se2=se2.astype(np.float32), N=a.truth_n)
        (cell / f"net_{s:04d}.json").write_text(json.dumps(res))        # written last: marks the net complete
        if a.weights_out and s == a.seeds[0]:
            wd = Path(a.weights_out) / run_id; wd.mkdir(parents=True, exist_ok=True)
            z = np.load(npz)
            np.savez(wd / f"net_{s:04d}.npz", head_w=z["head_w"], **{f"w{l}": z[f"w{l}"] for l in range(a.depth)})
        print(f"[{run_id}] net {s}: acc {res['final_val_acc']:.3f} rerandom {res['ablation']['rerandom']['acc']:.3f} "
              f"({time.time()-t1:.0f}s)", flush=True)
        npz.unlink()                                                      # scratch weights are not kept
    print(f"[{run_id}] done in {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
