"""Does the trained net use the random remnant?  Function-level ablation of W = alpha*W0 + Delta (MZC nets).

Written 2026-10-21.  Per net, per condition, on the task's own validation draw (make_gmm_task + sample):
  full        W                      (reproduces the corpus final_val_acc)
  rerandom    Delta + alpha*W0'      remnant swapped for a fresh independent He draw of the same scale
  delta_only  Delta                  remnant removed
  remnant     alpha*W0               task update removed
  rerand_L<k> single-layer swap, k in {0, 1, 8, 15}
Reports accuracy, and the cosine between the final post-ReLU activations of the condition and of `full`.
Also the pre-activation energy split on real inputs: ||h aW0||^2 : ||h Delta||^2 per layer (+ their cosine).
usage: mzc_random_remnant_ablation.py OUT.json NPZ [NPZ ...]     (needs the sibling Model_Zoo_Cartography checkout)
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/Source/Rosetta_Program/Model_Zoo_Cartography/train"))
from gmm_task import make_gmm_task
torch.backends.cuda.matmul.allow_tf32 = False
out, files = sys.argv[1], sys.argv[2:]
dev = "cuda"; res = {}
for f in files:
    j = json.load(open(f.replace(".npz", ".json"))); z = np.load(f)
    C, sep, ts = j["task"]["n_classes"], j["task"]["separation"], j["task"]["task_seed"]
    task = make_gmm_task(dim=1024, n_classes=C, separation=sep, seed=ts)
    x, y = task.sample(20000, np.random.default_rng(j["training"]["data_seed"] + 500_000))
    x = torch.tensor(x, device=dev); y = torch.tensor(y, device=dev)
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(16)]; W0 = [torch.tensor(z[f"init_w{l}"], device=dev) for l in range(16)]
    head = torch.tensor(z["head_w"], device=dev)
    al = [float((a * b).sum() / (b * b).sum()) for a, b in zip(W, W0)]
    De = [W[l] - al[l] * W0[l] for l in range(16)]
    g = torch.Generator(device=dev); g.manual_seed(12345)
    Wn = [torch.randn(1024, 1024, device=dev, generator=g) * (2.0 / 1024) ** 0.5 for _ in range(16)]   # fresh He draw
    def run(layers):
        h = x
        for Wl in layers: h = torch.relu(h @ Wl)
        return h, (h @ head).argmax(1)
    hf, pf = run(W); acc = lambda p: float((p == y).float().mean())
    conds = {"full": W,
             "rerandom": [De[l] + al[l] * Wn[l] for l in range(16)],
             "delta_only": De, "remnant": [al[l] * W0[l] for l in range(16)]}
    for k in (0, 1, 8, 15):
        conds[f"rerand_L{k}"] = [(De[l] + al[l] * Wn[l]) if l == k else W[l] for l in range(16)]
    r = {}
    for name, layers in conds.items():
        h, p = run(layers)
        cos = float(torch.nn.functional.cosine_similarity(h.flatten(), hf.flatten(), dim=0))
        r[name] = dict(acc=acc(p), cos_final=cos)
    # pre-activation energy split along the true trajectory
    h = x; split = []
    for l in range(16):
        zr = h @ (al[l] * W0[l]); zd = h @ De[l]
        er, ed = float((zr ** 2).sum()), float((zd ** 2).sum())
        split.append(dict(rand_share=er / (er + ed), cos=float((zr * zd).sum() / (er * ed) ** 0.5)))
        h = torch.relu(zr + zd)
    res[os.path.relpath(f).split("corpus/")[-1]] = dict(C=C, chance=1.0 / C, bayes=j["task"]["bayes_accuracy"],
                                                         corpus_val_acc=j["final_val_acc"], conds=r, split=split)
    print(os.path.relpath(f).split("corpus/")[-1], "full %.3f rerandom %.3f delta_only %.3f remnant %.3f" % tuple(r[k]["acc"] for k in ("full", "rerandom", "delta_only", "remnant")), flush=True)
json.dump(res, open(out, "w"), indent=1)
