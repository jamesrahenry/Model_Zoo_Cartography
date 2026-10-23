"""Per-net analysis for the preregistered grid (see PREREG.md).

Weight decomposition W = alpha*W0 + Delta; function-level ablation of the random remnant; aligned-slice search with a
random-subspace control; Monte Carlo per-layer truth under N(0,I).  GPU, torch only.  Ported from the pilot scripts in
arc-whitebox-canary/phase2/harness (mzc_weight_decomposition.py, mzc_random_remnant_ablation.py,
mzc_remnant_structure_search.py, mzc_mc_truth.py).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from gmm_task import make_gmm_task

_TASKS = {}
KS_ALL = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _load(npz, dev):
    z = np.load(npz)
    depth = sum(1 for k in z.files if k.startswith("w") and k[1:].isdigit())
    W = [torch.tensor(z[f"w{l}"], device=dev) for l in range(depth)]
    W0 = [torch.tensor(z[f"init_w{l}"], device=dev) for l in range(depth)]
    return W, W0, torch.tensor(z["head_w"], device=dev), depth


def mc_truth(W, n_samples, seed, dev, batch=32768):
    width = W[0].shape[0]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    s1 = torch.zeros(len(W), width, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1)
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        h = torch.randn(b, width, device=dev, generator=g)
        for l, Wl in enumerate(W):
            h = torch.relu(h @ Wl)
            s1[l] += h.sum(0, dtype=torch.float64); s2[l] += (h.double() ** 2).sum(0)
        done += b
    mean = s1 / n_samples
    return mean.cpu().numpy(), ((s2 / n_samples - mean ** 2) / n_samples).cpu().numpy()


def analyze(npz, prov_path, dev="cuda", truth_n=4_000_000, val_n=20000):
    torch.backends.cuda.matmul.allow_tf32 = False
    prov = json.load(open(prov_path)); tr = prov["training"]
    W, W0, head, D = _load(npz, dev); width = W[0].shape[0]
    C = prov["task"]["n_classes"]; fam = prov["task"]["family"]
    val_seed = tr.get("val_seed", tr.get("data_seed", 0) + 500_000)
    if fam == "gmm":
        task = make_gmm_task(dim=width, n_classes=C, separation=prov["task"]["separation"], seed=prov["task"]["task_seed"])
        x, y = task.sample(val_n, np.random.default_rng(val_seed))
    else:                                              # whitened MNIST / Fashion-MNIST (validation = held-out test split)
        from mnist_task import make_mnist_task
        key = (fam, width, prov["task"]["projection_seed"])
        if key not in _TASKS: _TASKS[key] = make_mnist_task(dim=width, seed=key[2], dataset=fam.split("_")[0])
        task = _TASKS[key]
        x, y = task.sample_val(val_n, np.random.default_rng(val_seed))
    x = torch.tensor(x, device=dev); y = torch.tensor(y, device=dev)

    # --- decomposition (float64)
    al = [float((a.double() * b.double()).sum() / (b.double() ** 2).sum()) for a, b in zip(W, W0)]
    De = [W[l] - al[l] * W0[l] for l in range(D)]
    dec = []
    for l in range(D):
        Dd = De[l].double(); s = torch.linalg.svdvals(Dd); e = s ** 2
        sW = torch.linalg.svdvals(W[l].double())
        sig = float((al[l] * W0[l]).double().std()); edge = sig * 2 * np.sqrt(width)
        out = sW[sW > 1.02 * edge]
        dec.append(dict(alpha=al[l], delta_rel=float(Dd.norm() / (al[l] * W0[l]).double().norm()),
                        delta_pr=float(e.sum() ** 2 / (e ** 2).sum()), delta_top8=float(e[:8].sum() / e.sum()),
                        n_outliers=int(len(out)), outlier_energy=float((out ** 2).sum() / (sW ** 2).sum())))

    g = torch.Generator(device=dev); g.manual_seed(2024)
    Wn = [torch.randn(width, width, device=dev, generator=g) * (2.0 / width) ** 0.5 for _ in range(D)]

    def run(layers):
        h = x
        for Wl in layers: h = torch.relu(h @ Wl)
        return h, (h @ head).argmax(1)
    hf, pf = run(W)
    acc = lambda p: float((p == y).float().mean())

    # --- ablation
    conds = {"full": W, "rerandom": [De[l] + al[l] * Wn[l] for l in range(D)],
             "delta_only": De, "remnant": [al[l] * W0[l] for l in range(D)]}
    for k in sorted({0, 1, D // 2, D - 1}):
        conds[f"rerand_L{k}"] = [(De[l] + al[l] * Wn[l]) if l == k else W[l] for l in range(D)]
    abl = {}
    for name, layers in conds.items():
        h, p = run(layers)
        abl[name] = dict(acc=acc(p), cos_final=float(torch.nn.functional.cosine_similarity(h.flatten(), hf.flatten(), dim=0)))
    split = []; h = x
    for l in range(D):
        zr = h @ (al[l] * W0[l]); zd = h @ De[l]
        er, ed = float((zr ** 2).sum()), float((zd ** 2).sum())
        split.append(dict(rand_share=er / (er + ed), cos=float((zr * zd).sum() / (er * ed) ** 0.5)))
        h = torch.relu(zr + zd)

    # --- aligned-slice search
    KS = [k for k in KS_ALL if k <= width]
    svd = [torch.linalg.svd(Dl) for Dl in De]
    rnd = [torch.linalg.qr(torch.randn(width, width, device=dev, generator=g))[0] for _ in range(2 * D)]

    def build(l, k, kind):
        if kind == "random_rerand": U, V = rnd[l][:, :k], rnd[D + l][:, :k]
        else: U, V = svd[l][0][:, :k], svd[l][2][:k].T
        R = U @ (U.T @ W0[l]) + (W0[l] @ V) @ V.T - U @ ((U.T @ W0[l] @ V) @ V.T)
        if kind == "aligned_zero": return De[l] + al[l] * R
        rest = Wn[l] - U @ (U.T @ Wn[l]) - (Wn[l] @ V) @ V.T + U @ ((U.T @ Wn[l] @ V) @ V.T)
        return De[l] + al[l] * (R + rest)
    slc = {kind: {str(k): acc(run([build(l, k, kind) for l in range(D)])[1]) for k in KS}
           for kind in ("aligned_rerand", "aligned_zero", "random_rerand")}

    mean, se2 = mc_truth(W, truth_n, seed=7, dev=dev)
    wd_, lr_, st_ = tr["weight_decay"], tr["lr"], tr["steps"]
    return dict(run_id=prov["run_id"], net=prov["net"], C=C, wd=wd_, lr=lr_, steps=st_, task=fam,
                pred_alpha=float(np.exp(-lr_ * wd_ * max(st_ - 250, 0))),
                outcome=prov["outcome"], final_val_acc=prov["final_val_acc"], bayes=prov["task"]["bayes_accuracy"],
                chance=1.0 / C, width=width, depth=D, truth_n=truth_n, val_n=len(y), decomposition=dec,
                ablation=abl, split=split, slice=slc), mean, se2
